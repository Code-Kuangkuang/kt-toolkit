import copy
import datetime
import json
import os
import uuid
import statistics
import csv

import torch
from rich import print

from core.device_info import get_device_info
from core.factory import build_dataset, build_model, build_trainer
from core.hooks import BestMetricsHook, SaveBestHook, WandbHook


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_run_config(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def apply_overrides(train_cfg, model_cfg, overrides):
    if overrides.get("batch_size") is not None:
        train_cfg["batch_size"] = overrides["batch_size"]
    if overrides.get("num_epochs") is not None:
        train_cfg["num_epochs"] = overrides["num_epochs"]
    if overrides.get("learning_rate") is not None:
        model_cfg["learning_rate"] = overrides["learning_rate"]
    if overrides.get("emb_size") is not None:
        model_cfg["emb_size"] = overrides["emb_size"]
    if overrides.get("dropout") is not None:
        model_cfg["dropout"] = overrides["dropout"]
    if overrides.get("dataset_mode") is not None:
        train_cfg["dataset_mode"] = overrides["dataset_mode"]
    if overrides.get("patience") is not None:
        train_cfg["patience"] = overrides["patience"]


def build_optimizer(train_cfg, model_cfg, model):
    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    weight_decay = model_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(), lr=model_cfg["learning_rate"], momentum=0.9, weight_decay=weight_decay
        )
    return torch.optim.Adam(model.parameters(), lr=model_cfg["learning_rate"], weight_decay=weight_decay)


def build_hooks(wandb_cfg, run_name, dataset_name, model_name, fold_id, train_cfg, model_cfg, cv_run_name, ckpt_dir, emb_type):
    hooks = [
        BestMetricsHook(metric_key="valid_auc", mode="max"),
        SaveBestHook(save_dir=ckpt_dir, filename=f"{emb_type}_model.pt"),
    ]

    if wandb_cfg:
        wandb_kwargs = {
            "enabled": True,
            "run_name": run_name,
            "config": {
                "dataset_name": dataset_name,
                "model_name": model_name,
                "fold": fold_id,
                "train_config": train_cfg,
                "model_config": model_cfg,
                "cv_run_name": cv_run_name,
            },
            "project": wandb_cfg.get("project"),
            "entity": wandb_cfg.get("entity") or wandb_cfg.get("uid"),
            "api_key": wandb_cfg.get("api_key"),
        }
        if cv_run_name is not None:
            wandb_kwargs["group"] = cv_run_name
        mode = wandb_cfg.get("mode")
        if mode is not None:
            wandb_kwargs["mode"] = mode
        hooks.append(WandbHook(**wandb_kwargs))

    return hooks


def print_run_overview(device, model, model_cfg, dataset_cfg, train_cfg):
    info = get_device_info(device)
    print("Training on device:\n" + "\n".join(info.format_lines()))
    print(f"Model_Info:\n[green][bold]{model}[/bold][/green]\n")
    print("Model_Config:\n" + json.dumps(model_cfg, indent=2, ensure_ascii=True) + "\n")
    print("Dataset_Config:\n" + json.dumps(dataset_cfg, indent=2, ensure_ascii=True) + "\n")
    print("Train_Config:\n" + json.dumps(train_cfg, indent=2, ensure_ascii=True) + "\n")


def aggregate_fold_metrics(fold_results):
    numeric_keys = set()
    for r in fold_results:
        bm = r.get("best_metrics") or {}
        for k, v in bm.items():
            if isinstance(v, (int, float)):
                numeric_keys.add(k)

    summary = {}
    for k in sorted(numeric_keys):
        values = []
        for r in fold_results:
            bm = r.get("best_metrics") or {}
            v = bm.get(k)
            if isinstance(v, (int, float)):
                values.append(float(v))
        if not values:
            continue
        mean_v = statistics.mean(values)
        std_v = statistics.pstdev(values) if len(values) > 1 else 0.0
        summary[k] = {
            "mean": mean_v,
            "std": std_v,
            "values": values,
        }
    return summary


def save_cv_summary(cv_dir, cv_payload, fold_results):
    save_run_config(os.path.join(cv_dir, "cv_summary.json"), cv_payload)

    csv_path = os.path.join(cv_dir, "cv_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["fold", "valid_auc", "valid_acc", "best_epoch", "ckpt_dir", "best_path", "run_name"],
        )
        writer.writeheader()
        for r in fold_results:
            bm = r.get("best_metrics") or {}
            writer.writerow(
                {
                    "fold": r.get("fold"),
                    "valid_auc": bm.get("valid_auc"),
                    "valid_acc": bm.get("valid_acc"),
                    "best_epoch": bm.get("epoch"),
                    "ckpt_dir": r.get("ckpt_dir"),
                    "best_path": r.get("best_path"),
                    "run_name": r.get("run_name"),
                }
            )


def print_cv_summary(agg, cv_dir):
    if "valid_auc" in agg:
        m = agg["valid_auc"]["mean"]
        s = agg["valid_auc"]["std"]
        print(f"\n[green][bold]CV valid_auc mean={m:.6f} std={s:.6f}[/bold][/green]")
    if "valid_acc" in agg:
        m = agg["valid_acc"]["mean"]
        s = agg["valid_acc"]["std"]
        print(f"[green][bold]CV valid_acc mean={m:.6f} std={s:.6f}[/bold][/green]\n")
    print(f"CV summary saved to: [bold]{cv_dir}[/bold]")


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
):
    kt_cfg = copy.deepcopy(kt_cfg_raw)
    train_cfg_local = kt_cfg["train_config"]
    model_cfg_local = kt_cfg[model_name]

    resolved_emb_type = emb_type or model_cfg_local.get("emb_type", "qid")
    model_cfg_local["emb_type"] = resolved_emb_type

    apply_overrides(train_cfg_local, model_cfg_local, overrides)

    if train_cfg_local.get("patience") == -1:
        train_cfg_local["patience"] = None

    # Filter out learning_rate and other_config parameters for model
    other_config_keys = {"loss_c_all_lambda", "loss_q_all_lambda", "loss_c_next_lambda", "loss_q_next_lambda",
                         "output_mode", "output_c_all_lambda", "output_c_next_lambda", "output_q_all_lambda",
                         "output_q_next_lambda", "emb_type", "learning_rate", "use_timestamps", "dpath",
                         "num_at", "num_it"}
    model_kwargs = {k: v for k, v in model_cfg_local.items() if k not in other_config_keys}

    data_config = copy.deepcopy(data_config_raw)
    dataset_cfg_local = data_config[dataset_name]
    if "dpath" in dataset_cfg_local:
        dpath = dataset_cfg_local["dpath"]
        if not os.path.isabs(dpath):
            dpath = os.path.join(root_dir, dpath)
        dataset_cfg_local["dpath"] = os.path.normpath(dpath)

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    # Build test dataloader if data exists
    test_loader = None
    test_path = os.path.join(dataset_cfg_local["dpath"], dataset_cfg_local.get("test_file_quelevel", dataset_cfg_local.get("test_file", "")))
    if os.path.exists(dataset_cfg_local["dpath"]) and dataset_cfg_local.get("test_file"):
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

    # Evaluate on test set if available
    test_metrics = None
    if test_loader is not None:
        test_metrics = trainer.evaluate_test()
        if test_metrics:
            print(f"Test results: AUC={test_metrics.get('test_auc', -1):.4f}, ACC={test_metrics.get('test_acc', -1):.4f}")

    best_metrics = getattr(trainer, "best_metrics", None)
    if best_metrics is not None:
        # Merge test metrics into best_metrics if available
        if test_metrics:
            best_metrics.update(test_metrics)
        save_run_config(os.path.join(ckpt_dir, "best_metrics.json"), best_metrics)

    return {
        "fold": fold_id,
        "run_name": run_name,
        "ckpt_dir": ckpt_dir,
        "emb_type": resolved_emb_type,
        "best_metrics": best_metrics,
        "best_path": getattr(trainer, "best_path", None),
    }
