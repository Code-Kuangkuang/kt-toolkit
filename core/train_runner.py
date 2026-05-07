import copy
import datetime
import os
import uuid

import torch
from rich import print

from core.factory import build_dataset, build_model, build_trainer
from core.run_support import (
    aggregate_fold_metrics,
    apply_overrides,
    build_hooks,
    build_optimizer,
    print_cv_summary,
    print_run_overview,
    save_cv_summary,
    save_run_config,
    set_seed,
)
from strategies import apply_dkt_pebg_strategy


def train_one_fold(
    *,
    dataset_name,
    model_name,
    emb_type,
    fold_id,
    root_dir,
    kt_cfg_raw,
    data_config_raw,
    seed,
    save_root,
    add_uuid,
    wandb_cfg,
    kt_config_path,
    data_config_path,
    wandb_config_path,
    cv_run_name,
    overrides,
    gpu_id=0,
):
    kt_cfg = copy.deepcopy(kt_cfg_raw)
    train_cfg_local = kt_cfg["train_config"]
    model_cfg_local = kt_cfg[model_name]

    resolved_emb_type = emb_type or model_cfg_local.get("emb_type", "qid")
    model_cfg_local["emb_type"] = resolved_emb_type

    apply_overrides(train_cfg_local, model_cfg_local, overrides)

    if train_cfg_local.get("patience") == -1:
        train_cfg_local["patience"] = None

    data_config = copy.deepcopy(data_config_raw)
    dataset_cfg_local = data_config[dataset_name]
    if "dpath" in dataset_cfg_local:
        dpath = dataset_cfg_local["dpath"]
        if not os.path.isabs(dpath):
            dpath = os.path.join(root_dir, dpath)
        dataset_cfg_local["dpath"] = os.path.normpath(dpath)

    booster_info = None
    if model_name == "dkt_pebg":
        model_cfg_local, booster_info = apply_dkt_pebg_strategy(
            model_cfg=model_cfg_local,
            dataset_name=dataset_name,
            dataset_cfg=dataset_cfg_local,
            root_dir=root_dir,
            fold_id=fold_id,
        )
        print(
            "DKT-PEBG booster strategy resolved: "
            f"strategy={booster_info.get('strategy')} "
            f"enabled={booster_info.get('enabled')} "
            f"emb_path={booster_info.get('emb_path', '')}"
        )

    # Filter out learning_rate and other_config parameters for model
    other_config_keys = {"loss_c_all_lambda", "loss_q_all_lambda", "loss_c_next_lambda", "loss_q_next_lambda",
                         "output_mode", "output_c_all_lambda", "output_c_next_lambda", "output_q_all_lambda",
                         "output_q_next_lambda", "emb_type", "learning_rate", "use_timestamps", "dpath",
                         "num_at", "num_it", "booster_strategy", "require_fold_embedding"}
    model_kwargs = {k: v for k, v in model_cfg_local.items() if k not in other_config_keys}

    set_seed(seed)

    # Set device with specified GPU ID
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
        torch.cuda.set_device(gpu_id)
    else:
        device = "cpu"

    model = build_model(
        model_name,
        num_c=dataset_cfg_local["num_c"],
        num_q=dataset_cfg_local["num_q"],
        emb_type=resolved_emb_type,
        seq_len=train_cfg_local.get("seq_len"),
        device=device,
        dpath=dataset_cfg_local.get("dpath", ""),
        num_at=model_cfg_local.get("num_at"),
        num_it=model_cfg_local.get("num_it"),
        **model_kwargs,
    ).to(device)

    # Apply weight init for specific models (same as pykt)
    if model_name == "hawkes":
        model.apply(model.init_weights)
        model = model.double()

    # Resolve timestamp loading before any run overview/logging.
    model_use_timestamps = model_cfg_local.get("use_timestamps", False)
    use_timestamps = bool(train_cfg_local.get("use_timestamps", False) or model_use_timestamps)
    train_cfg_local["use_timestamps"] = use_timestamps

    print_run_overview(device, model, model_cfg_local, dataset_cfg_local, train_cfg_local)

    # Get dataset_mode from overrides (if specified)
    dataset_mode = train_cfg_local.get("dataset_mode")

    train_loader, valid_loader = build_dataset(
        "kt_default",
        dataset_name=dataset_name,
        data_config=data_config,
        fold=fold_id,
        batch_size=train_cfg_local["batch_size"],
        model_name=model_name,
        dataset_mode=dataset_mode,
        use_timestamps=use_timestamps,
    )

    # Build test dataloader if data exists. Peiyou's official test file has
    # hidden targets marked as -1, so it is for prediction/submission only.
    test_loader = None
    is_peiyou = dataset_name.lower() == "peiyou"
    test_path = os.path.join(dataset_cfg_local["dpath"], dataset_cfg_local.get("test_file_quelevel", dataset_cfg_local.get("test_file", "")))
    if is_peiyou:
        print(
            "Peiyou test evaluation disabled: pykt_test.csv contains hidden "
            "targets marked as -1. Use scripts/predict_peiyou.py to generate prediction.csv."
        )
    elif os.path.exists(dataset_cfg_local["dpath"]) and dataset_cfg_local.get("test_file"):
        try:
            test_loader = build_dataset(
                "kt_test",
                dataset_name=dataset_name,
                data_config=data_config,
                batch_size=train_cfg_local["batch_size"],
                model_name=model_name,
                dataset_mode=dataset_mode,
                use_timestamps=use_timestamps,
            )
            print(f"Test loader built from: {test_path}")
        except Exception as e:
            print(f"Warning: Could not build test loader: {e}")
            test_loader = None

    opt = build_optimizer(train_cfg_local, model_cfg_local, model)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{dataset_name}-{model_name}-fold{fold_id}-{ts}"
    if add_uuid == 1:
        run_name = f"{run_name}-{uuid.uuid4()}"
    ckpt_dir = os.path.join(save_root, run_name)

    run_config = {
        "run_name": run_name,
        "timestamp": ts,
        "save_dir": save_root,
        "ckpt_dir": ckpt_dir,
        "device": device,
        "dataset_name": dataset_name,
        "model_name": model_name,
        "emb_type": resolved_emb_type,
        "fold": fold_id,
        "seed": seed,
        "use_wandb": bool(wandb_cfg),
        "add_uuid": bool(add_uuid),
        "cv": {
            "enabled": bool(cv_run_name),
            "cv_run_name": cv_run_name,
        },
        "config_paths": {
            "kt_config": kt_config_path,
            "data_config": data_config_path,
            "wandb_config": wandb_config_path if wandb_cfg else None,
        },
        "train_config": train_cfg_local,
        "model_config": model_cfg_local,
        "dataset_config": dataset_cfg_local,
        "booster_info": booster_info,
        "wandb": {
            "enabled": bool(wandb_cfg),
            "project": wandb_cfg.get("project") if wandb_cfg else None,
            "entity": wandb_cfg.get("entity") if wandb_cfg else None,
            "mode": wandb_cfg.get("mode") if wandb_cfg else None,
        },
    }
    save_run_config(os.path.join(ckpt_dir, "run_config.json"), run_config)

    hooks = build_hooks(
        wandb_cfg,
        run_name,
        dataset_name,
        model_name,
        fold_id,
        train_cfg_local,
        model_cfg_local,
        cv_run_name,
        ckpt_dir,
        resolved_emb_type,
    )

    trainer_kwargs = {
        "model": model,
        "train_loader": train_loader,
        "valid_loader": valid_loader,
        "optimizer": opt,
        "num_epochs": train_cfg_local["num_epochs"],
        "device": device,
        "hooks": hooks,
        "other_config": model_cfg_local,
        "test_loader": test_loader,
    }
    if "patience" in train_cfg_local:
        trainer_kwargs["patience"] = train_cfg_local["patience"]

    trainer = build_trainer(model_name, **trainer_kwargs)
    trainer.run()

    best_path = getattr(trainer, "best_path", None)



    last_epoch_path = os.path.join(ckpt_dir, "last_epoch_model.pt")
    torch.save(trainer.model.state_dict(), last_epoch_path)


    best_test_metrics = None
    if best_path and os.path.exists(best_path):
        trainer.model.load_state_dict(torch.load(best_path, weights_only=True))
        if test_loader is not None:
            print(f"Loaded best model from epoch {trainer.best_metrics.get('epoch', '?')} for test evaluation")
            best_test_metrics = trainer.evaluate_test()
            if best_test_metrics:
                print(f"[Best-Valid Epoch] Test AUC={best_test_metrics.get('test_auc', -1):.4f}, ACC={best_test_metrics.get('test_acc', -1):.4f}")
        else:
            print(f"Loaded best model from epoch {trainer.best_metrics.get('epoch', '?')}")


    last_test_metrics = None
    if test_loader is not None:
        trainer.model.load_state_dict(torch.load(last_epoch_path, weights_only=True))
        print(f"Loaded last epoch model for test evaluation")
        last_test_metrics = trainer.evaluate_test()
        if last_test_metrics:
            print(f"[Last Epoch]        Test AUC={last_test_metrics.get('test_auc', -1):.4f}, ACC={last_test_metrics.get('test_acc', -1):.4f}")

    best_metrics = getattr(trainer, "best_metrics", None)
    if best_metrics is not None:
        # Rename keys so best_metric dict carries unambiguous names
        if best_test_metrics:
            best_metrics["best_test_auc"] = best_test_metrics.get("test_auc", -1)
            best_metrics["best_test_acc"] = best_test_metrics.get("test_acc", -1)
        if last_test_metrics:
            best_metrics["last_test_auc"] = last_test_metrics.get("test_auc", -1)
            best_metrics["last_test_acc"] = last_test_metrics.get("test_acc", -1)
        save_run_config(os.path.join(ckpt_dir, "best_metrics.json"), best_metrics)

    return {
        "fold": fold_id,
        "run_name": run_name,
        "ckpt_dir": ckpt_dir,
        "emb_type": resolved_emb_type,
        "best_metrics": best_metrics,
        "best_path": getattr(trainer, "best_path", None),
    }
