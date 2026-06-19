import csv
import json
import os
import random
import statistics

from rich import print


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
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"


def build_optimizer(train_cfg, model_cfg, model):
    import torch

    optimizer_name = train_cfg.get("optimizer", "adam").lower()
    weight_decay = model_cfg.get("weight_decay", train_cfg.get("weight_decay", 0.0))
    if optimizer_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=model_cfg["learning_rate"],
            momentum=0.9,
            weight_decay=weight_decay,
        )
    return torch.optim.Adam(
        model.parameters(),
        lr=model_cfg["learning_rate"],
        weight_decay=weight_decay,
    )


def build_hooks(wandb_cfg, run_name, dataset_name, model_name, fold_id, train_cfg, model_cfg, cv_run_name, ckpt_dir, emb_type):
    from core.hooks import BestMetricsHook, MetricsJsonlHook, SaveBestHook, WandbHook

    hooks = [
        BestMetricsHook(metric_key="valid_auc", mode="max"),
        SaveBestHook(save_dir=ckpt_dir, filename=f"{model_name}_{emb_type}_model.pt"),
        MetricsJsonlHook(os.path.join(ckpt_dir, "metrics.jsonl")),
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
    from core.device_info import get_device_info

    info = get_device_info(device)
    print("Training on device:\n" + "\n".join(info.format_lines()))
    print(f"Model_Info:\n[green][bold]{model}[/bold][/green]\n")
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
        "C1",
        "C2",
        "C4",
        "epsilon",
        "lambda_theta",
        "lambda_radius",
        "lambda_conf",
        "lambda_center",
        "lambda_coverage",
        "lambda_gate",
        "max_coverage_weight",
        "coverage_gate_init",
        "use_bbp_radius_normalization",
        "use_radius_discrimination",
        "use_kab_radius_features",
        "use_radius_state_update",
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


def aggregate_fold_metrics(fold_results):
    numeric_keys = set()
    for result in fold_results:
        best_metrics = result.get("best_metrics") or {}
        for key, value in best_metrics.items():
            if isinstance(value, (int, float)):
                numeric_keys.add(key)

    summary = {}
    for key in sorted(numeric_keys):
        values = []
        for result in fold_results:
            best_metrics = result.get("best_metrics") or {}
            value = best_metrics.get(key)
            if isinstance(value, (int, float)):
                values.append(float(value))
        if not values:
            continue
        summary[key] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return summary


def save_cv_summary(cv_dir, cv_payload, fold_results):
    save_run_config(os.path.join(cv_dir, "cv_summary.json"), cv_payload)

    csv_path = os.path.join(cv_dir, "cv_summary.csv")
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "fold",
                "valid_auc",
                "valid_acc",
                "best_test_auc",
                "best_test_acc",
                "last_test_auc",
                "last_test_acc",
                "best_epoch",
                "ckpt_dir",
                "best_path",
                "run_name",
            ],
        )
        writer.writeheader()
        for result in fold_results:
            best_metrics = result.get("best_metrics") or {}
            writer.writerow(
                {
                    "fold": result.get("fold"),
                    "valid_auc": best_metrics.get("valid_auc"),
                    "valid_acc": best_metrics.get("valid_acc"),
                    "best_test_auc": best_metrics.get("best_test_auc"),
                    "best_test_acc": best_metrics.get("best_test_acc"),
                    "last_test_auc": best_metrics.get("last_test_auc"),
                    "last_test_acc": best_metrics.get("last_test_acc"),
                    "best_epoch": best_metrics.get("epoch"),
                    "ckpt_dir": result.get("ckpt_dir"),
                    "best_path": result.get("best_path"),
                    "run_name": result.get("run_name"),
                }
            )


def print_cv_summary(agg, cv_dir):
    _print_metric(agg, "valid_auc", "CV valid_auc", style="green")
    _print_metric(agg, "valid_acc", "CV valid_acc", style="green")
    _print_metric(agg, "best_test_auc", "CV best_test_auc", style="green")
    _print_metric(agg, "best_test_acc", "CV best_test_acc", style="green")
    _print_metric(agg, "last_test_auc", "CV last_test_auc ", style="cyan")
    _print_metric(agg, "last_test_acc", "CV last_test_acc ", style="cyan")
    print("")
    print(f"CV summary saved to: [bold]{cv_dir}[/bold]")


def _print_metric(agg, key, label, style):
    if key not in agg:
        return
    mean_v = agg[key]["mean"]
    std_v = agg[key]["std"]
    if style == "green":
        print(f"[green][bold]{label} mean={mean_v:.6f} std={std_v:.6f}[/bold][/green]")
    else:
        print(f"[{style}]{label} mean={mean_v:.6f} std={std_v:.6f}[/{style}]")
