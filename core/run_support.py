import json
import os
import random

from rich import print

from core.cv_results import aggregate_fold_metrics, print_cv_summary, save_cv_summary


def set_seed(seed):
    """Set the global random seed for Python, NumPy, and PyTorch."""
    import numpy as np
    import torch

    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except Exception as exc:
        print("Set seed failed, details are ", exc)
    np.random.seed(seed)
    random.seed(seed)
    # Needed for deterministic cuBLAS GEMMs, and cheap: it only sizes a
    # workspace.
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    # CUDA_LAUNCH_BLOCKING serialises every kernel launch against the host. It
    # makes an async CUDA error surface at its real call site instead of at some
    # later sync point, which is worth a lot while debugging and costs a large
    # part of GPU throughput the rest of the time. It is not needed for
    # reproducibility -- the seeds and the cuDNN flags above cover that -- so it
    # is opt-in.
    if os.environ.get("KT_DEBUG") == "1":
        os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
        print("KT_DEBUG=1: CUDA_LAUNCH_BLOCKING enabled (slow; debugging only).")


def build_optimizer(train_cfg, model_cfg, model):
    import torch

    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    weight_decay = model_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0))
    base_lr = model_cfg["learning_rate"]
    parameters = (
        model.param_groups(base_lr)
        if callable(getattr(model, "param_groups", None))
        else model.parameters()
    )
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=base_lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
    return torch.optim.Adam(
        parameters,
        lr=base_lr,
        weight_decay=weight_decay,
    )


def build_hooks(wandb_cfg, run_name, dataset_name, model_name, fold_id, train_cfg, model_cfg, cv_run_name, ckpt_dir, emb_type):
    from core.hooks import BestMetricsHook, MetricsJsonlHook, SaveBestHook, WandbHook

    hooks = [
        BestMetricsHook(metric_key="valid_auc", mode="max"),
        SaveBestHook(save_dir=ckpt_dir, filename=f"{model_name}_{emb_type}_model.pt"),
        MetricsJsonlHook(
            os.path.join(ckpt_dir, "metrics.jsonl"),
            # So the rows survive being concatenated across runs, instead of
            # having to be attributed by parsing the directory path.
            identity={
                "dataset": dataset_name,
                "model": model_name,
                "fold": fold_id,
                "seed": train_cfg.get("seed"),
                "train_label_flip_ratio": train_cfg.get(
                    "train_label_flip_ratio", 0.0
                ),
            },
        ),
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


def print_run_overview(device, model, model_cfg, dataset_cfg, train_cfg,
                       model_info=None):
    from core.device_info import get_device_info

    info = get_device_info(device)
    print("Training on device:\n" + "\n".join(info.format_lines()))
    print(f"Model_Info:\n[green][bold]{model}[/bold][/green]\n")
    if model_info is not None:
        # The repr above says what the layers are; this says how big they are,
        # which is what decides whether two runs are comparable.
        from core.model_info import format_model_info

        print(format_model_info(model_info) + "\n")
    print("Model_Config:\n" + json.dumps(model_cfg, indent=2, ensure_ascii=True) + "\n")
    print("Dataset_Config:\n" + json.dumps(dataset_cfg, indent=2, ensure_ascii=True) + "\n")
    print("Train_Config:\n" + json.dumps(train_cfg, indent=2, ensure_ascii=True) + "\n")


def save_run_config(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def apply_overrides(train_cfg, model_cfg, overrides):
    train_keys = {"batch_size", "num_epochs", "dataset_mode", "patience"}
    model_keys = {
        "learning_rate",
        "emb_size",
        "dropout",
        "d_model",
        "d_ff",
        "num_attn_heads",
        "n_blocks",
        "num_blocks",
        "difficult_levels",
        "emb_type",
        "epsilon",
        "lambda_theta",
        "lambda_radius",
        "lambda_conf",
        "lambda_geo",
        "lambda_step",
        "lambda_rel",
        "lambda_kl",
        "lambda_prior",
        "kl_warmup_epochs",
        "clean_prior",
        "frequency_tau",
        "frequency_gamma",
        "use_frequency_scale",
        "reliability_hidden",
        "reliability_init",
        "use_reliability",
        "use_geometry_in_filter",
        "use_hierarchical_requirement",
        "use_bbp_radius_normalization",
        "use_radius_discrimination",
        "use_kab_radius_features",
        "use_radius_state_update",
        "use_coverage",
        "use_backbone",
        "use_student_radius",
        "use_question_radius",
        "update_student_radius",
        "radius_update_mode",
        "normalize_centers",
        "use_point_space",
        "use_concept_readout",
        "history_mode",
        "use_sequence_distance_attention",
        "use_time_distance_attention",
        "use_time_aware_kab",
        "use_time_forgetting",
        "use_timestamps",
        "use_dynamic_fusion",
        "use_scalar_item_difficulty",
        "max_concept_fusion_weight",
        "response_function",
    }
    for key in train_keys:
        if overrides.get(key) is not None:
            train_cfg[key] = overrides[key]
    for key in model_keys:
        if overrides.get(key) is not None:
            model_cfg[key] = overrides[key]
