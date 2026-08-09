"""Shared checkpoint and dataset helpers for offline KT analysis scripts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

import datasets.init_dataset  # noqa: F401 - register dataset builders
import models  # noqa: F401 - register models
from core.factory import build_dataset, build_model


ROOT = Path(__file__).resolve().parents[1]

MODEL_CONFIG_EXCLUDE = {
    "loss_c_all_lambda",
    "loss_q_all_lambda",
    "loss_c_next_lambda",
    "loss_q_next_lambda",
    "output_mode",
    "output_c_all_lambda",
    "output_c_next_lambda",
    "output_q_all_lambda",
    "output_q_next_lambda",
    "emb_type",
    "learning_rate",
    "use_timestamps",
    "dpath",
    "num_at",
    "num_it",
    "booster_strategy",
    "require_fold_embedding",
    "lambda_ball",
    "lambda_concept_next",
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
    "lambda_move",
    "lambda_item",
    "lambda_coverage",
    "lambda_item_difficulty",
    "lambda_coverage_gate",
    "lambda_response_gate",
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {value}")
    return device


def resolve_checkpoint(run_dir: Path, model_name: str, emb_type: str) -> Path:
    candidates = [
        run_dir / f"{model_name}_{emb_type}_model.pt",
        run_dir / f"{emb_type}_model.pt",
        run_dir / "model.pt",
    ]
    for path in candidates:
        if path.exists():
            return path

    pt_files = sorted(
        path
        for path in run_dir.glob("*.pt")
        if path.name != "last_epoch_model.pt"
    )
    if len(pt_files) == 1:
        return pt_files[0]
    if not pt_files:
        raise FileNotFoundError(f"No best-validation checkpoint found in {run_dir}")
    raise RuntimeError(
        f"Ambiguous checkpoints in {run_dir}: {[path.name for path in pt_files]}"
    )


def torch_load_weights(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def resolve_local_data_path(run_config: dict[str, Any]) -> Path:
    dataset_name = str(run_config["dataset_name"])
    configured = Path(str(run_config["dataset_config"].get("dpath", "")))
    if configured.exists():
        return configured.resolve()
    local = ROOT / "data" / dataset_name
    if local.exists():
        return local.resolve()
    raise FileNotFoundError(
        "Dataset directory does not exist in the saved config or local workspace: "
        f"configured={configured}, local={local}"
    )


def build_run_model(
    run_dir: Path,
    device: torch.device,
) -> tuple[dict[str, Any], torch.nn.Module, Path]:
    run_dir = run_dir.resolve()
    run_config = load_json(run_dir / "run_config.json")
    dataset_cfg = dict(run_config["dataset_config"])
    model_cfg = dict(run_config["model_config"])
    train_cfg = dict(run_config["train_config"])
    model_name = str(run_config["model_name"])
    emb_type = str(run_config.get("emb_type") or model_cfg.get("emb_type", "qid"))

    model_kwargs = {
        key: value
        for key, value in model_cfg.items()
        if key not in MODEL_CONFIG_EXCLUDE
    }
    model = build_model(
        model_name,
        num_c=int(dataset_cfg["num_c"]),
        num_q=int(dataset_cfg["num_q"]),
        emb_type=emb_type,
        seq_len=train_cfg.get("seq_len"),
        device=device,
        dpath=str(resolve_local_data_path(run_config)),
        num_at=model_cfg.get("num_at"),
        num_it=model_cfg.get("num_it"),
        **model_kwargs,
    ).to(device)

    checkpoint = resolve_checkpoint(run_dir, model_name, emb_type)
    state = torch_load_weights(checkpoint, device)
    model.load_state_dict(state, strict=True)
    model.eval()
    return run_config, model, checkpoint


def build_loaders(
    run_config: dict[str, Any],
    split: str,
    batch_size: int | None = None,
):
    dataset_name = str(run_config["dataset_name"])
    model_name = str(run_config["model_name"])
    fold = int(run_config["fold"])
    train_cfg = dict(run_config["train_config"])
    model_cfg = dict(run_config["model_config"])
    dataset_cfg = dict(run_config["dataset_config"])
    dataset_cfg["dpath"] = str(resolve_local_data_path(run_config))

    data_config = {dataset_name: dataset_cfg}
    use_timestamps = bool(
        train_cfg.get("use_timestamps", False)
        or model_cfg.get("use_timestamps", False)
    )
    dataset_mode = train_cfg.get("dataset_mode")
    resolved_batch_size = int(batch_size or train_cfg.get("batch_size", 64))

    if split in {"train", "valid"}:
        train_loader, valid_loader = build_dataset(
            "kt_default",
            dataset_name=dataset_name,
            data_config=data_config,
            fold=fold,
            batch_size=resolved_batch_size,
            model_name=model_name,
            dataset_mode=dataset_mode,
            use_timestamps=use_timestamps,
        )
        return [(split, train_loader if split == "train" else valid_loader)]
    if split == "test":
        test_loader = build_dataset(
            "kt_test",
            dataset_name=dataset_name,
            data_config=data_config,
            batch_size=resolved_batch_size,
            model_name=model_name,
            dataset_mode=dataset_mode,
            use_timestamps=use_timestamps,
        )
        return [("test", test_loader)]
    if split == "all":
        return (
            build_loaders(run_config, "train", resolved_batch_size)
            + build_loaders(run_config, "valid", resolved_batch_size)
            + build_loaders(run_config, "test", resolved_batch_size)
        )
    raise ValueError(f"Unknown split: {split}")


def concat_full(seqs: torch.Tensor | None, shifted: torch.Tensor | None):
    if seqs is None or shifted is None or seqs.numel() == 0:
        return None
    return torch.cat((seqs[:, :1], shifted), dim=1)


def safe_auc(target, score) -> float:
    target = np.asarray(target)
    score = np.asarray(score)
    mask = np.isfinite(target) & np.isfinite(score)
    target = target[mask]
    score = score[mask]
    if len(target) == 0 or len(np.unique(target)) < 2:
        return math.nan
    try:
        from sklearn.metrics import roc_auc_score

        return float(roc_auc_score(target, score))
    except Exception:
        return math.nan
