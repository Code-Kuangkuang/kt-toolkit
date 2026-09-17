import copy
import datetime
import os
import uuid

import numpy as np
import torch
from rich import print

from core.factory import build_dataset, build_model, build_trainer
from core.dataset_names import is_hidden_label_dataset, normalize_dataset_name
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
from datasets.init_dataset import protocol_stamp
from datasets.lpkt_utils import generate_time2idx
from datasets.feature_utils import (
    compute_dkt_forget_stats,
    compute_dimkt_difficulty_maps,
    compute_hqaf_feature_maps,
)
from models.gkt_utils import get_gkt_graph
from models.dgekt_utils import build_dgekt_graphs
from strategies import apply_dkt_pebg_strategy


MODEL_NAME_ALIASES = {
    "dkt-forget": "dkt_forget",
    "hd-kt": "hdkt",
    "hd_lpkt": "hdkt",
    "hd-lpkt": "hdkt",
    "hd-dkt": "hd_dkt",
    "hd-akt": "hd_akt",
    "hd-simplekt": "hd_simplekt",

    "lefokt": "lefokt_akt",
    "hqaf-kt": "hqaf",
    "hqaf_kt": "hqaf",
}
QUESTION_REQUIRED_MODELS = {"atdkt", "dimkt", "stablekt", "sparsekt", "robustkt", "dtransformer", "rekt", "lefokt_akt", "hqaf", "keenkt", "dgekt", "lpkt", "hdkt"}
ALL_IN_ONE_MODELS = {
    "lpkt",
    "hdkt",
    "hd_dkt",
    "hd_akt",
    "hd_simplekt",
    "atdkt",
    "dimkt",
    "stablekt",
    "sparsekt",
    "robustkt",
    "dtransformer",
    "dkt_forget",
    "skvmn",
    "rekt",
    "lefokt_akt",
    "hqaf",
    "keenkt",
    "dgekt",
}
ONE_BY_ONE_MODELS = {"hawkes"}


def _resolve_dataset_mode(model_name, train_cfg, model_cfg, overrides=None):
    """Resolve data mode with CLI > model config > compatibility defaults."""
    overrides = overrides or {}
    if overrides.get("dataset_mode") is not None:
        mode = overrides["dataset_mode"]
    elif model_cfg.get("dataset_mode") is not None:
        mode = model_cfg["dataset_mode"]
    elif model_name in ALL_IN_ONE_MODELS:
        mode = "all_in_one"
    elif model_name in ONE_BY_ONE_MODELS:
        mode = "one_by_one"
    else:
        mode = train_cfg.get("dataset_mode", "one_by_one")
    if mode not in {"one_by_one", "all_in_one"}:
        raise ValueError(
            f"Unsupported dataset_mode={mode!r} for {model_name}; "
            "expected 'one_by_one' or 'all_in_one'."
        )
    return mode


def _resolve_existing_sequence_filename(dataset_cfg, primary_key, fallback_key):
    """Return the filename that the dataset builder will actually load."""
    primary_name = dataset_cfg.get(primary_key) or dataset_cfg.get(fallback_key)
    fallback_name = dataset_cfg.get(fallback_key)
    for filename in (primary_name, fallback_name):
        if not filename:
            continue
        if os.path.exists(os.path.join(dataset_cfg["dpath"], filename)):
            return filename
    return primary_name or fallback_name


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
    train_label_flip_ratio=0.0,
    train_label_flip_seed=None,
):
    dataset_name = normalize_dataset_name(dataset_name)
    model_name = MODEL_NAME_ALIASES.get(model_name.lower(), model_name.lower())
    kt_cfg = copy.deepcopy(kt_cfg_raw)
    train_cfg_local = kt_cfg["train_config"]
    model_cfg_local = kt_cfg[model_name]

    resolved_emb_type = emb_type or model_cfg_local.get("emb_type", "qid")
    model_cfg_local["emb_type"] = resolved_emb_type

    apply_overrides(train_cfg_local, model_cfg_local, overrides)

    train_label_flip_ratio = float(train_label_flip_ratio)
    if not 0.0 <= train_label_flip_ratio <= 1.0:
        raise ValueError(
            "train_label_flip_ratio must be in [0, 1], "
            f"got {train_label_flip_ratio}."
        )
    if train_label_flip_seed is None:
        train_label_flip_seed = seed
    train_label_flip_seed = int(train_label_flip_seed)
    train_cfg_local["train_label_flip_ratio"] = train_label_flip_ratio
    train_cfg_local["train_label_flip_seed"] = train_label_flip_seed

    if train_cfg_local.get("patience") == -1:
        train_cfg_local["patience"] = None

    data_config = copy.deepcopy(data_config_raw)
    dataset_cfg_local = data_config[dataset_name]
    if "dpath" in dataset_cfg_local:
        dpath = dataset_cfg_local["dpath"]
        if not os.path.isabs(dpath):
            dpath = os.path.join(root_dir, dpath)
        dataset_cfg_local["dpath"] = os.path.normpath(dpath)

    resolved_dataset_mode = _resolve_dataset_mode(
        model_name, train_cfg_local, model_cfg_local, overrides
    )
    train_cfg_local["dataset_mode"] = resolved_dataset_mode
    # A per-model opt-out for the windowed test set, declared in the model's
    # config block but consumed as a training-run setting.
    if model_cfg_local.get("eval_window") is not None:
        train_cfg_local["eval_window"] = bool(model_cfg_local["eval_window"])
    model_cfg_local["dataset_mode"] = resolved_dataset_mode

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

    lpkt_time_idx_maps = None
    if model_name in {"lpkt", "hdkt"}:
        train_time_folds = (
            sorted(set(dataset_cfg_local.get("folds", [])) - {int(fold_id)})
            if model_name == "hdkt" or resolved_dataset_mode == "all_in_one"
            else None
        )
        at2idx, it2idx = generate_time2idx(
            dataset_cfg_local, folds=train_time_folds
        )
        lpkt_time_idx_maps = {"at2idx": at2idx, "it2idx": it2idx}
        dataset_cfg_local["num_at"] = len(at2idx) + 1
        dataset_cfg_local["num_it"] = len(it2idx) + 1
        model_cfg_local["num_at"] = dataset_cfg_local["num_at"]
        model_cfg_local["num_it"] = dataset_cfg_local["num_it"]
        if model_name == "hdkt" or resolved_dataset_mode == "all_in_one":
            dataset_cfg_local["time_index_scope"] = "train_folds_only"
            dataset_cfg_local["time_index_folds"] = train_time_folds
    if model_name == "hawkes" and dataset_cfg_local.get("num_q", 0) <= 0:
        raise ValueError(
            f"Hawkes requires question ids, but dataset {dataset_name} has num_q={dataset_cfg_local.get('num_q')}."
        )
    if model_name in QUESTION_REQUIRED_MODELS and (
        "questions" not in dataset_cfg_local.get("input_type", []) or dataset_cfg_local.get("num_q", 0) <= 0
    ):
        raise ValueError(
            f"{model_name} requires question ids, but dataset {dataset_name} has "
            f"input_type={dataset_cfg_local.get('input_type')} and num_q={dataset_cfg_local.get('num_q')}."
        )
    dimkt_difficulty_maps = None
    hqaf_feature_maps = None
    if model_name == "dkt_forget":
        train_valid_key = "train_valid_file_quelevel" if train_cfg_local.get("dataset_mode") == "all_in_one" else "train_valid_file"
        test_key = "test_file_quelevel" if train_cfg_local.get("dataset_mode") == "all_in_one" else "test_file"
        gap_files = [
            _resolve_existing_sequence_filename(dataset_cfg_local, train_valid_key, "train_valid_file"),
            _resolve_existing_sequence_filename(dataset_cfg_local, test_key, "test_file"),
        ]
        gap_stats = compute_dkt_forget_stats(
            dataset_cfg_local["dpath"],
            gap_files,
            dataset_cfg_local["input_type"],
        )
        model_cfg_local.update(gap_stats)
        dataset_cfg_local.update(gap_stats)
        model_cfg_local["use_timestamps"] = True
    elif model_name == "dimkt":
        difficult_levels = int(model_cfg_local.get("difficult_levels", model_cfg_local.get("diff_level", 100)))
        model_cfg_local["difficult_levels"] = difficult_levels
        model_cfg_local["batch_size"] = train_cfg_local["batch_size"]
        model_cfg_local["num_steps"] = train_cfg_local.get("seq_len", 200)
        train_folds = sorted(set(dataset_cfg_local.get("folds", [])) - {fold_id})
        difficulty_file_key = "train_valid_file_quelevel" if train_cfg_local.get("dataset_mode") == "all_in_one" else "train_valid_file"
        difficulty_file = _resolve_existing_sequence_filename(
            dataset_cfg_local,
            difficulty_file_key,
            "train_valid_file",
        )
        dimkt_difficulty_maps = compute_dimkt_difficulty_maps(
            dataset_cfg_local["dpath"],
            difficulty_file,
            difficult_levels,
            folds=train_folds,
        )
    elif model_name == "hqaf":
        diff_level = int(model_cfg_local.get("diff_level", model_cfg_local.get("difficult_levels", 50)))
        num_time_bins = int(model_cfg_local.get("num_time_bins", 20))
        model_cfg_local["diff_level"] = diff_level
        model_cfg_local["num_time_bins"] = num_time_bins
        model_cfg_local["num_type"] = int(dataset_cfg_local.get("num_type", model_cfg_local.get("num_type", 16)))
        train_file_key = "train_valid_file_quelevel" if train_cfg_local.get("dataset_mode") == "all_in_one" else "train_valid_file"
        train_folds = sorted(set(dataset_cfg_local.get("folds", [])) - {fold_id})
        train_file = _resolve_existing_sequence_filename(
            dataset_cfg_local,
            train_file_key,
            "train_valid_file",
        )
        hqaf_feature_maps = compute_hqaf_feature_maps(
            dataset_cfg_local["dpath"],
            train_file,
            diff_level=diff_level,
            num_time_bins=num_time_bins,
            folds=train_folds,
        )
        if not hqaf_feature_maps.get("has_usetimes", False):
            print("Warning: HQAF source data has no 'usetimes' column; using default time bucket 0.")
        if not hqaf_feature_maps.get("has_type", False):
            print("Warning: HQAF source data has no 'type' column; using default question type 0.")
        dimkt_difficulty_maps = {
            "skills": hqaf_feature_maps.get("skills", {}),
            "questions": hqaf_feature_maps.get("questions", {}),
        }

    dgekt_graph_info = None

    # Filter out learning_rate and other_config parameters for model
    other_config_keys = {"loss_c_all_lambda", "loss_q_all_lambda", "loss_c_next_lambda", "loss_q_next_lambda",
                          "output_mode", "output_c_all_lambda", "output_c_next_lambda", "output_q_all_lambda",
                          "output_q_next_lambda", "emb_type", "learning_rate", "use_timestamps", "dpath",
                           "num_at", "num_it", "booster_strategy", "require_fold_embedding",
                           "lambda_item_difficulty", "lambda_item_l2", "lambda_rel", "lambda_kl",
                           "lambda_prior", "kl_warmup_epochs", "clean_prior",
                           "lambda_move", "lambda_item", "dataset_mode",
                           # Routed to the dataset builder, not the model.
                           "concept_mode", "eval_window"}
    model_kwargs = {k: v for k, v in model_cfg_local.items() if k not in other_config_keys}
    if model_name == "lpkt":
        model_kwargs["use_runtime_concepts"] = resolved_dataset_mode == "all_in_one"
    if model_name in {"simplekt", "ukt", "stablekt", "sparsekt", "robustkt", "dtransformer", "lefokt_akt", "hqaf"}:
        model_kwargs.setdefault("num_pid", dataset_cfg_local.get("num_q", 0))
    if model_name == "hqaf":
        model_kwargs.setdefault("num_type", dataset_cfg_local.get("num_type", model_cfg_local.get("num_type", 16)))
    if model_name == "gkt":
        graph_type = model_cfg_local.get("graph_type", "dense")
        graph_file = f"gkt_graph_{graph_type}.npz"
        graph_path = os.path.join(dataset_cfg_local["dpath"], graph_file)
        if os.path.exists(graph_path):
            graph = np.load(graph_path, allow_pickle=True)["matrix"]
        else:
            graph = get_gkt_graph(
                dataset_cfg_local["num_c"],
                dataset_cfg_local["dpath"],
                dataset_cfg_local.get("train_valid_original_file", dataset_cfg_local.get("train_valid_file")),
                dataset_cfg_local.get("test_original_file", dataset_cfg_local.get("test_file")),
                graph_type=graph_type,
                tofile=graph_file,
            )
        model_kwargs["graph"] = graph.float() if torch.is_tensor(graph) else torch.tensor(graph).float()
    elif model_name == "dgekt":
        if "concepts" not in dataset_cfg_local.get("input_type", []):
            raise ValueError(
                f"DGEKT requires question-concept associations, but dataset {dataset_name} "
                f"has input_type={dataset_cfg_local.get('input_type')}."
            )
        graph_file_key = (
            "train_valid_file_quelevel"
            if train_cfg_local.get("dataset_mode") == "all_in_one"
            else "train_valid_file"
        )
        graph_file = _resolve_existing_sequence_filename(
            dataset_cfg_local, graph_file_key, "train_valid_file"
        )
        test_graph_file_key = (
            "test_file_quelevel"
            if train_cfg_local.get("dataset_mode") == "all_in_one"
            else "test_file"
        )
        test_graph_file = _resolve_existing_sequence_filename(
            dataset_cfg_local, test_graph_file_key, "test_file"
        )
        association_files = []
        if model_cfg_local.get("include_test_question_metadata", True) and test_graph_file and os.path.exists(
            os.path.join(dataset_cfg_local["dpath"], test_graph_file)
        ):
            association_files.append(test_graph_file)
        train_folds = sorted(set(dataset_cfg_local.get("folds", [])) - {int(fold_id)})
        hypergraph, transition_out, transition_in, dgekt_graph_info = build_dgekt_graphs(
            dataset_cfg_local["dpath"],
            graph_file,
            num_q=dataset_cfg_local["num_q"],
            num_c=dataset_cfg_local["num_c"],
            train_folds=train_folds,
            association_files=association_files,
        )
        model_kwargs.update(
            {
                "hypergraph": hypergraph,
                "transition_out": transition_out,
                "transition_in": transition_in,
            }
        )

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
    dataset_feature_kwargs = {
        "include_dkt_forget": model_name == "dkt_forget",
        "difficulty_maps": dimkt_difficulty_maps,
        "include_history": model_name == "atdkt" and "his" in resolved_emb_type,
        "include_hqaf_attrs": model_name == "hqaf",
        "hqaf_feature_maps": hqaf_feature_maps,
        # Optional `concept_mode` in the model's config block forces multi/first
        # instead of taking it from MULTI_CONCEPT_MODELS, so the cost of
        # truncation can be measured without editing code.
        "concept_mode": model_cfg_local.get("concept_mode"),
    }

    train_loader, valid_loader = build_dataset(
        "kt_default",
        dataset_name=dataset_name,
        data_config=data_config,
        fold=fold_id,
        batch_size=train_cfg_local["batch_size"],
        model_name=model_name,
        dataset_mode=dataset_mode,
        use_timestamps=use_timestamps,
        time_idx_maps=lpkt_time_idx_maps,
        train_label_flip_ratio=train_label_flip_ratio,
        train_label_flip_seed=train_label_flip_seed,
        **dataset_feature_kwargs,
    )
    train_label_flip_info = train_loader.dataset.label_flip_info

    # Build test dataloader if data exists. AAAI2023's official test file has
    # hidden targets marked as -1, so it is for prediction/submission only.
    test_loader = None
    has_hidden_test_labels = is_hidden_label_dataset(dataset_name)
    test_file_key = (
        "test_file_quelevel" if dataset_mode == "all_in_one" else "test_file"
    )
    test_filename = _resolve_existing_sequence_filename(
        dataset_cfg_local,
        test_file_key,
        "test_file",
    )
    test_path = os.path.join(dataset_cfg_local["dpath"], test_filename or "")
    if has_hidden_test_labels:
        print(
            "AAAI2023 test evaluation disabled: pykt_test.csv contains hidden "
            "targets marked as -1. Use scripts/predict_aaai2023.py to generate prediction.csv."
        )
    elif test_filename and os.path.exists(test_path):
        try:
            test_loader = build_dataset(
                "kt_test",
                dataset_name=dataset_name,
                data_config=data_config,
                batch_size=train_cfg_local["batch_size"],
                model_name=model_name,
                dataset_mode=dataset_mode,
                use_timestamps=use_timestamps,
                time_idx_maps=lpkt_time_idx_maps,
                **dataset_feature_kwargs,
            )
            print(f"Test loader built from: {test_path}")
        except Exception as e:
            print(f"Warning: Could not build test loader: {e}")
            test_loader = None

    # The windowed test file is what pykt reports on: one row per position, each
    # with a full-length history, instead of non-overlapping chunks that leave
    # boundary positions with almost none.  It is ~20x larger, so it is scored
    # once at the end rather than during training.
    window_test_loader = None
    # Models whose forward runs a per-timestep Python loop (SKVMN's hop-LSTM is
    # the worst) take hours on the windowed file, which has ~20x the rows.  They
    # can opt out with `eval_window: false` in their config block; the choice is
    # recorded in the protocol stamp so a table cannot silently mix rows that
    # have a windowed number with rows that do not.
    eval_window = bool(train_cfg_local.get("eval_window", True))
    window_file_key = (
        "test_window_file_quelevel" if dataset_mode == "all_in_one" else "test_window_file"
    )
    window_filename = _resolve_existing_sequence_filename(
        dataset_cfg_local,
        window_file_key,
        "test_window_file",
    )
    window_path = os.path.join(dataset_cfg_local["dpath"], window_filename or "")
    if not eval_window:
        print("Windowed test evaluation disabled for this model (eval_window=false).")
    elif test_loader is not None and window_filename and os.path.exists(window_path):
        try:
            window_test_loader = build_dataset(
                "kt_test",
                dataset_name=dataset_name,
                data_config=data_config,
                batch_size=train_cfg_local["batch_size"],
                model_name=model_name,
                dataset_mode=dataset_mode,
                use_timestamps=use_timestamps,
                time_idx_maps=lpkt_time_idx_maps,
                window=True,
                **dataset_feature_kwargs,
            )
            print(f"Windowed test loader built from: {window_path}")
        except Exception as e:
            print(f"Warning: Could not build windowed test loader: {e}")
            window_test_loader = None

    opt = build_optimizer(train_cfg_local, model_cfg_local, model)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_name = f"{dataset_name}-{model_name}-fold{fold_id}-{ts}"
    if train_label_flip_ratio > 0:
        ratio_tag = f"{train_label_flip_ratio:g}".replace(".", "p")
        run_name = f"{run_name}-flip{ratio_tag}"
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
        "train_label_flip": train_label_flip_info,
        # Two runs are comparable only when this block matches.
        "protocol": protocol_stamp(
            model_name,
            train_cfg_local.get("dataset_mode"),
            dataset_cfg_local.get("max_concepts"),
            model_cfg_local.get("concept_mode"),
            eval_window=bool(train_cfg_local.get("eval_window", True)),
        ),
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
        "dgekt_graph": dgekt_graph_info,
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
    # Attached rather than passed through __init__: every trainer subclass
    # declares its own constructor, so a new keyword would break all of them.
    trainer.window_test_loader = window_test_loader
    trainer.run()

    best_path = getattr(trainer, "best_path", None)



    last_epoch_path = os.path.join(ckpt_dir, "last_epoch_model.pt")
    torch.save(trainer.model.state_dict(), last_epoch_path)


    best_test_metrics = None
    best_window_metrics = None
    if best_path and os.path.exists(best_path):
        trainer.model.load_state_dict(torch.load(best_path, weights_only=True))
        if test_loader is not None:
            print(f"Loaded best model from epoch {trainer.best_metrics.get('epoch', '?')} for test evaluation")
            best_test_metrics = trainer.evaluate_test()
            if best_test_metrics:
                print(f"[Best-Valid Epoch] Test AUC={best_test_metrics.get('test_auc', -1):.4f}, ACC={best_test_metrics.get('test_acc', -1):.4f}")
            if window_test_loader is not None:
                best_window_metrics = trainer.evaluate_window_test()
                if best_window_metrics:
                    print(
                        f"[Best-Valid Epoch] Window Test AUC="
                        f"{best_window_metrics.get('window_test_auc', -1):.4f}, "
                        f"ACC={best_window_metrics.get('window_test_acc', -1):.4f}"
                        "   (pykt-comparable protocol)"
                    )
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
        if best_window_metrics:
            best_metrics["best_window_test_auc"] = best_window_metrics.get("window_test_auc", -1)
            best_metrics["best_window_test_acc"] = best_window_metrics.get("window_test_acc", -1)
        if last_test_metrics:
            best_metrics["last_test_auc"] = last_test_metrics.get("test_auc", -1)
            best_metrics["last_test_acc"] = last_test_metrics.get("test_acc", -1)
        save_run_config(os.path.join(ckpt_dir, "best_metrics.json"), best_metrics)

    return {
        "fold": fold_id,
        "run_name": run_name,
        "ckpt_dir": ckpt_dir,
        "emb_type": resolved_emb_type,
        "train_label_flip": train_label_flip_info,
        "best_metrics": best_metrics,
        "best_path": getattr(trainer, "best_path", None),
    }
