"""Export fold-wise, student-aligned predictions from trained KT checkpoints.

Usage:
    python scripts/export_predictions.py -d algebra2005 -m removed_model

Each exported CSV contains one row per valid test interaction:
    fold, student_id, interaction_id, y_true, y_pred

`interaction_id` is a deterministic per-student running index over valid test
interactions in dataloader order. It is created while the 2-D sequence/mask
structure is still available, so variable sequence lengths are handled safely.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import typer
from rich import print
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.trainers
import datasets.init_dataset
import models
from core.config import load_cfg
from core.dataset_names import normalize_dataset_name
from core.factory import build_dataset, build_model
from core.run_support import set_seed

app = typer.Typer(add_completion=False)

ALL_IN_ONE_MODELS = {
    "lpkt", "atdkt", "dimkt", "stablekt", "sparsekt", "robustkt",
    "dtransformer", "dkt_forget", "skvmn", "rekt", "lefokt_akt",
    "hqaf", "keenkt", "removed_model", "removed_model", "removed_model", "removed_model",
}

SUPPORTED_EXPORT_MODELS = {"removed_model", "akt", "ukt", "qikt"}
EXPORTER_VERSION = "2026-07-31-run-config-parity-v3"

# Keep model construction consistent with core/train_runner.py. These values are
# trainer/loss-side configuration and were not forwarded into model constructors
# during the original experiments.
TRAIN_RUNNER_OTHER_CONFIG_KEYS = {
    "loss_c_all_lambda", "loss_q_all_lambda", "loss_c_next_lambda",
    "loss_q_next_lambda", "output_mode", "output_c_all_lambda",
    "output_c_next_lambda", "output_q_all_lambda", "output_q_next_lambda",
    "emb_type", "learning_rate", "use_timestamps", "dpath", "num_at",
    "num_it", "booster_strategy", "require_fold_embedding",
    "lambda_item_difficulty", "lambda_rel", "lambda_kl", "lambda_prior",
    "kl_warmup_epochs", "clean_prior", "lambda_move", "lambda_item",
    "lambda_coverage_gate", "lambda_response_gate",
}


def find_cv_dir(dataset: str, model: str, train_root: Path) -> Optional[Path]:
    """Find the most recent CV directory for a dataset/model pair."""
    dataset_dir = train_root / dataset
    if dataset_dir.exists():
        matches = list(dataset_dir.glob(f"cv-{dataset}-{model}-*"))
        if matches:
            return sorted(matches)[-1]
    matches = list(train_root.glob(f"cv-{dataset}-{model}-*"))
    if matches:
        return sorted(matches)[-1]
    return None


def get_best_checkpoint(run_dir: Path, run_config: Optional[dict] = None) -> Optional[Path]:
    """Resolve the validation-best checkpoint without arbitrary filename sorting.

    Priority:
    1. An explicit checkpoint path stored in run_config.
    2. A checkpoint whose filename contains "best".
    3. The only non-last-epoch .pt file.

    If multiple ambiguous checkpoints remain, fail instead of silently choosing
    a potentially wrong epoch.
    """
    run_config = run_config or {}
    explicit_keys = (
        "best_checkpoint", "best_ckpt", "best_model_path", "checkpoint_path",
        "best_checkpoint_path", "best_model",
    )

    for key in explicit_keys:
        value = run_config.get(key)
        if not value:
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = run_dir / candidate
        if candidate.exists() and candidate.suffix == ".pt":
            return candidate

    checkpoints = sorted(
        p for p in run_dir.glob("*.pt") if "last_epoch" not in p.name.lower()
    )
    if not checkpoints:
        return None

    best_named = [p for p in checkpoints if "best" in p.name.lower()]
    if len(best_named) == 1:
        return best_named[0]
    if len(checkpoints) == 1:
        return checkpoints[0]

    names = ", ".join(p.name for p in checkpoints)
    raise RuntimeError(
        "Multiple checkpoint files found but no unique validation-best checkpoint "
        f"could be identified in {run_dir}: {names}. Store the best checkpoint path "
        "in run_config.json or keep a uniquely named *best*.pt file."
    )


def _as_2d(tensor: torch.Tensor, name: str, batch_size: int) -> torch.Tensor:
    """Normalize prediction/target tensors to [batch, time]."""
    if tensor is None:
        raise ValueError(f"{name} is None")
    if tensor.dim() == 3 and tensor.size(-1) == 1:
        tensor = tensor.squeeze(-1)
    if tensor.dim() == 1:
        if tensor.numel() == batch_size:
            tensor = tensor.reshape(batch_size, 1)
        else:
            raise ValueError(
                f"Cannot map 1-D {name} with {tensor.numel()} values to "
                f"batch_size={batch_size}"
            )
    if tensor.dim() != 2:
        raise ValueError(f"Expected {name} to be 2-D [batch,time], got {tuple(tensor.shape)}")
    return tensor


def collect_masked_batch(
    student_ids,
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor],
    interaction_counters: dict,
) -> list[tuple[int, int, int, float]]:
    """Collect valid predictions while preserving student/sequence structure.

    Returns rows of:
        (student_id, interaction_id, y_true, y_pred)

    `interaction_id` is a per-student running index and therefore continues
    correctly when a long student sequence is split into multiple chunks.
    """
    if student_ids is None:
        raise ValueError(
            "Test batch does not contain 'student_id' or 'uid'. Student-level "
            "cluster inference cannot be performed without a real student id."
        )

    if torch.is_tensor(student_ids):
        sid_values = student_ids.detach().cpu().reshape(-1).tolist()
    else:
        sid_values = np.asarray(student_ids).reshape(-1).tolist()

    batch_size = len(sid_values)
    pred = _as_2d(pred, "pred", batch_size).detach().cpu()
    target = _as_2d(target, "target", batch_size).detach().cpu()

    if pred.shape != target.shape:
        raise ValueError(
            f"Prediction/target shape mismatch: pred={tuple(pred.shape)}, "
            f"target={tuple(target.shape)}"
        )

    if mask is None:
        mask_cpu = torch.ones_like(target, dtype=torch.bool)
    else:
        mask_cpu = _as_2d(mask.bool(), "mask", batch_size).detach().cpu()
        if mask_cpu.shape != target.shape:
            raise ValueError(
                f"Mask/target shape mismatch: mask={tuple(mask_cpu.shape)}, "
                f"target={tuple(target.shape)}"
            )

    rows: list[tuple[int, int, int, float]] = []
    for b, sid_raw in enumerate(sid_values):
        sid = int(sid_raw)
        valid_positions = torch.nonzero(mask_cpu[b], as_tuple=False).reshape(-1).tolist()
        for t in valid_positions:
            y_value = int(round(float(target[b, t].item())))
            if y_value not in (0, 1):
                raise ValueError(f"Non-binary target {target[b, t].item()} for student {sid}")
            pred_value = float(pred[b, t].item())
            if not np.isfinite(pred_value):
                raise ValueError(f"Non-finite prediction {pred_value} for student {sid}")

            interaction_id = int(interaction_counters[sid])
            interaction_counters[sid] += 1
            rows.append((sid, interaction_id, y_value, pred_value))

    return rows


def _concat_full(seqs, shifted):
    if seqs is None or shifted is None or seqs.numel() == 0:
        return None
    return torch.cat([seqs[:, :1], shifted], dim=1)


def _nonempty(tensor):
    return tensor if tensor is not None and tensor.numel() > 0 else None


def _forward_removed_model_matrix(model, batch, device: str):
    """Mirror the existing a removed model evaluation path and keep [batch,time] output."""
    qseqs = _nonempty(batch.get("qseqs"))
    cseqs = _nonempty(batch.get("cseqs"))
    qshft = _nonempty(batch.get("shft_qseqs"))
    cshft = _nonempty(batch.get("shft_cseqs"))
    rseqs = _nonempty(batch.get("rseqs"))
    rshft = _nonempty(batch.get("shft_rseqs"))

    if qseqs is None or cseqs is None or rseqs is None or rshft is None:
        raise ValueError("a removed model requires question, concept, and response sequences.")

    q_full = _concat_full(qseqs.to(device), qshft.to(device) if qshft is not None else None)
    c_full = _concat_full(cseqs.to(device), cshft.to(device) if cshft is not None else None)
    r_full = _concat_full(rseqs.to(device), rshft.to(device))
    if q_full is None or c_full is None or r_full is None:
        raise ValueError("a removed model requires shifted question/concept/response sequences.")

    outputs = model(q_full.long(), c_full.long(), r_full.float())
    if not isinstance(outputs, dict):
        raise ValueError("a removed model must return a dict containing 'y' or 'logits'.")
    pred = outputs.get("y", outputs.get("logits"))
    if pred is None:
        raise ValueError("a removed model output does not contain 'y' or 'logits'.")
    return pred, rshft.to(device).float()


def _forward_akt_matrix(model, batch, device: str):
    """Reproduce AKTTrainer._forward_batch before masking."""
    qseqs = _nonempty(batch.get("qseqs"))
    cseqs = _nonempty(batch.get("cseqs"))
    rseqs = _nonempty(batch.get("rseqs"))
    qshft = _nonempty(batch.get("shft_qseqs"))
    cshft = _nonempty(batch.get("shft_cseqs"))
    rshft = _nonempty(batch.get("shft_rseqs"))

    if rseqs is None or rshft is None:
        raise ValueError("AKT requires response and shifted-response sequences.")

    q_full = _concat_full(
        qseqs.to(device) if qseqs is not None else None,
        qshft.to(device) if qshft is not None else None,
    )
    c_full = _concat_full(
        cseqs.to(device) if cseqs is not None else None,
        cshft.to(device) if cshft is not None else None,
    )
    r_full = _concat_full(rseqs.to(device), rshft.to(device))

    # Exactly as AKTTrainer: concepts are q_data and questions are pid_data.
    if c_full is not None:
        q_data = c_full
        pid_data = q_full
    else:
        q_data = q_full
        pid_data = None

    if q_data is None:
        raise ValueError("AKT requires concept or question sequences.")
    if r_full is None:
        raise ValueError("AKT requires shifted response sequences.")
    if pid_data is None and getattr(model, "n_pid", 0) > 0:
        raise ValueError("AKT requires question ids when n_pid > 0.")

    if pid_data is None:
        result = model(q_data.long(), r_full.long())
    else:
        result = model(q_data.long(), r_full.long(), pid_data.long())
    preds = result[0] if isinstance(result, tuple) else result

    # AKTTrainer unconditionally drops the full-sequence first position.
    preds = preds[:, 1:]
    target = rshft.to(device).float()
    if preds.shape != target.shape:
        raise ValueError(
            f"AKT prediction shape {tuple(preds.shape)} does not align with "
            f"shifted targets {tuple(target.shape)}."
        )
    return preds, target


def _forward_ukt_matrix(model, batch, device: str):
    """Reproduce UKTTrainer._forward_batch(batch, train=False) before masking."""
    qseqs = _nonempty(batch.get("qseqs"))
    cseqs = _nonempty(batch.get("cseqs"))
    rseqs = _nonempty(batch.get("rseqs"))
    qshft = _nonempty(batch.get("shft_qseqs"))
    cshft = _nonempty(batch.get("shft_cseqs"))
    rshft = _nonempty(batch.get("shft_rseqs"))
    masks = batch.get("masks")
    pidseqs = _nonempty(batch.get("pidseqs"))
    pidshft = _nonempty(batch.get("shft_pidseqs"))

    if rseqs is None or rshft is None:
        raise ValueError("UKT requires response and shifted-response sequences.")

    base_seqs = cseqs if cseqs is not None else qseqs
    base_shft = cshft if cshft is not None else qshft
    if base_seqs is None:
        raise ValueError("UKT requires question or concept sequences.")
    if base_shft is None:
        raise ValueError("UKT requires shifted question or concept sequences.")

    result = model(
        qseqs=qseqs.to(device).long() if qseqs is not None else None,
        rseqs=rseqs.to(device).long(),
        cseqs=base_seqs.to(device).long(),
        qshft=qshft.to(device).long() if qshft is not None else None,
        cshft=base_shft.to(device).long(),
        rshft=rshft.to(device).float(),
        pidseqs=pidseqs.to(device).long() if pidseqs is not None else None,
        pidshft=pidshft.to(device).long() if pidshft is not None else None,
        masks=masks.to(device) if masks is not None else None,
        train=False,
        shft_r_aug=None,
        r_aug=None,
    )
    preds = result
    target = rshft.to(device).float()
    preds_for_eval = preds[:, 1:] if preds.size(1) == target.size(1) + 1 else preds
    if preds_for_eval.shape != target.shape:
        raise ValueError(
            f"UKT prediction shape {tuple(preds.shape)} does not align with "
            f"shifted targets {tuple(target.shape)}."
        )
    return preds_for_eval, target


def _qikt_sigmoid_inverse(x: torch.Tensor, epsilon: float = 1e-8) -> torch.Tensor:
    # Keep the trainer's exact numerical expression for reproducibility.
    return torch.log(x / (1 - x + epsilon) + epsilon)


def _forward_qikt_matrix(model, batch, device: str):
    """Reproduce QIKTTrainer._forward_batch output fusion before masking."""
    qseqs = _nonempty(batch.get("qseqs"))
    cseqs = _nonempty(batch.get("cseqs"))
    rseqs = _nonempty(batch.get("rseqs"))
    qshft = _nonempty(batch.get("shft_qseqs"))
    cshft = _nonempty(batch.get("shft_cseqs"))
    rshft = _nonempty(batch.get("shft_rseqs"))
    sm = batch.get("smasks")
    masks = batch.get("masks")

    if qseqs is None or cseqs is None or rseqs is None or rshft is None:
        raise ValueError("QIKT requires question, concept, and response sequences.")
    if qshft is None or cshft is None:
        raise ValueError("QIKT requires shifted question and concept sequences.")
    if sm is None:
        raise ValueError("QIKT requires smasks.")

    qseqs = qseqs.to(device)
    cseqs = cseqs.to(device)
    rseqs = rseqs.to(device)
    qshft = qshft.to(device)
    cshft = cshft.to(device)
    rshft = rshft.to(device)
    sm = sm.to(device)
    masks = masks.to(device) if masks is not None else None

    q_full = _concat_full(qseqs, qshft)
    c_full = _concat_full(cseqs, cshft)
    r_full = _concat_full(rseqs, rshft)
    if q_full is None or c_full is None or r_full is None:
        raise ValueError("QIKT could not construct full sequences.")

    data = {
        "cq": q_full.long(),
        "cc": c_full.long(),
        "cr": r_full.long(),
        "q": qseqs,
        "c": cseqs,
        "r": rseqs,
        "qshft": qshft,
        "cshft": cshft,
        "rshft": rshft,
        "m": masks if masks is not None else torch.zeros_like(sm),
        "sm": sm,
    }
    outputs = model(data["cq"], data["cc"], data["cr"], data=data)

    # QIKTTrainer receives the full run-specific model_config as
    # ``other_config`` even though QIKTNet itself does not. Preserve those exact
    # fusion weights here without changing the checkpoint architecture.
    other_config = (
        getattr(model, "_export_eval_config", None)
        or getattr(model, "other_config", None)
        or {}
    )
    output_c_all_lambda = other_config.get("output_c_all_lambda", 1)
    output_c_next_lambda = other_config.get("output_c_next_lambda", 1)
    output_q_all_lambda = other_config.get("output_q_all_lambda", 1)

    required = {"y_question_all", "y_concept_all", "y_concept_next"}
    missing = required - set(outputs)
    if missing:
        raise ValueError(f"QIKT output is missing required keys: {sorted(missing)}")

    if getattr(model, "output_mode", None) == "an_irt":
        y = (
            _qikt_sigmoid_inverse(outputs["y_question_all"]) * output_q_all_lambda
            + _qikt_sigmoid_inverse(outputs["y_concept_all"]) * output_c_all_lambda
            + _qikt_sigmoid_inverse(outputs["y_concept_next"]) * output_c_next_lambda
        )
        y = torch.sigmoid(y)
    else:
        denominator = output_q_all_lambda + output_c_all_lambda + output_c_next_lambda
        if denominator == 0:
            raise ValueError("QIKT output fusion denominator is zero.")
        y = (
            outputs["y_question_all"] * output_q_all_lambda
            + outputs["y_concept_all"] * output_c_all_lambda
            + outputs["y_concept_next"] * output_c_next_lambda
        ) / denominator

    if y.shape != rshft.shape:
        raise ValueError(
            f"QIKT prediction shape {tuple(y.shape)} does not align with "
            f"shifted targets {tuple(rshft.shape)}."
        )
    return y, rshft


def _forward_prediction_matrix(model, batch, device: str, model_name: str):
    """Return trainer-aligned unmasked prediction and target matrices."""
    if model_name == "removed_model":
        return _forward_removed_model_matrix(model, batch, device)
    if model_name == "akt":
        return _forward_akt_matrix(model, batch, device)
    if model_name == "ukt":
        return _forward_ukt_matrix(model, batch, device)
    if model_name == "qikt":
        return _forward_qikt_matrix(model, batch, device)
    raise ValueError(
        f"Unsupported model '{model_name}'. This exporter is intentionally limited to "
        f"{sorted(SUPPORTED_EXPORT_MODELS)}."
    )


def run_inference(model, test_loader, device: str, model_name: str) -> list[tuple[int, int, int, float]]:
    """Run trainer-aligned test inference and preserve student/sequence identity.

    Any batch failure aborts the fold. Partial predictions are never returned.
    """
    if model_name not in SUPPORTED_EXPORT_MODELS:
        raise ValueError(
            f"Unsupported model '{model_name}'. Supported: {sorted(SUPPORTED_EXPORT_MODELS)}"
        )

    model.eval()
    model.to(device)
    rows: list[tuple[int, int, int, float]] = []
    interaction_counters = defaultdict(int)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            student_ids = batch.get("student_id", batch.get("uid"))
            mask_matrix = batch.get("smasks")
            try:
                pred_matrix, target_matrix = _forward_prediction_matrix(
                    model, batch, device, model_name
                )
                batch_rows = collect_masked_batch(
                    student_ids,
                    pred_matrix,
                    target_matrix,
                    mask_matrix,
                    interaction_counters,
                )
                rows.extend(batch_rows)
            except Exception as exc:
                raise RuntimeError(
                    f"Inference failed at batch {batch_idx} for model {model_name}: {exc}"
                ) from exc

    print(f"  Processed {len(test_loader)} batches, 0 errors")
    if not rows:
        raise RuntimeError("Inference completed without any valid predictions.")
    return rows


def build_fixed_test_loader(
    dataset: str,
    dataset_cfg: dict,
    batch_size: int,
    model_name: str,
    dataset_mode: Optional[str],
):
    """Build the single held-out test loader shared by all CV training runs.

    In this project, ``fold`` is used only to choose the train/validation split.
    Every validation-best checkpoint is evaluated on the same independent test
    file, so no fold identifier is passed to ``kt_test``.
    """
    print("  Using fixed held-out test set shared across CV runs.")
    return build_dataset(
        "kt_test",
        dataset_name=dataset,
        data_config=dataset_cfg,
        batch_size=batch_size,
        model_name=model_name,
        dataset_mode=dataset_mode,
    )

def _collect_json_key_values(obj, key: str, out: list[float]) -> None:
    if isinstance(obj, dict):
        for k, value in obj.items():
            if k == key and isinstance(value, (int, float)) and np.isfinite(value):
                out.append(float(value))
            else:
                _collect_json_key_values(value, key, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_json_key_values(value, key, out)


def _find_saved_best_test_auc(run_dir: Path) -> Optional[float]:
    """Find a unique saved best_test_auc in top-level JSON run artifacts."""
    values: list[float] = []
    for path in sorted(run_dir.glob("*.json")):
        try:
            with path.open(encoding="utf-8") as f:
                payload = json.load(f)
        except Exception:
            continue
        _collect_json_key_values(payload, "best_test_auc", values)

    if not values:
        return None
    reference = values[0]
    if all(np.isclose(v, reference, rtol=0.0, atol=1e-12) for v in values[1:]):
        return float(reference)
    return None


def _collect_string_values_for_key(obj, key: str, out: list[str]) -> None:
    """Recursively collect non-empty string values for a configuration key."""
    if isinstance(obj, dict):
        for k, value in obj.items():
            if k == key and isinstance(value, str) and value.strip():
                out.append(value.strip())
            else:
                _collect_string_values_for_key(value, key, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_string_values_for_key(value, key, out)


def _resolve_emb_type(
    model_name: str,
    run_config: dict,
    model_cfg: dict,
    checkpoint_path: Path,
) -> Optional[str]:
    """Resolve the embedding type used by the saved checkpoint.

    Priority is deliberately run-specific first, because statistical export must
    reconstruct the architecture that produced the checkpoint rather than rely
    on today's global config defaults. If run_config does not record emb_type,
    fall back to the model config and finally to conservative filename hints.
    """
    run_values: list[str] = []
    _collect_string_values_for_key(run_config, "emb_type", run_values)
    unique_run_values = list(dict.fromkeys(run_values))
    if len(unique_run_values) == 1:
        return unique_run_values[0]
    if len(unique_run_values) > 1:
        raise RuntimeError(
            "Ambiguous emb_type values in run_config: "
            + ", ".join(unique_run_values)
        )

    cfg_value = model_cfg.get("emb_type")
    if isinstance(cfg_value, str) and cfg_value.strip():
        return cfg_value.strip()

    name = checkpoint_path.name.lower()
    # These hints match the checkpoint naming convention used by this project.
    # Prefer the most specific token first.
    if "stoc_qid" in name:
        return "stoc_qid"
    if "iekt" in name:
        return "iekt"
    if "qid" in name:
        return "qid"
    return None



def _resolve_current_dataset_dpath(dataset: str, current_cfg: dict) -> str:
    """Resolve the locally available dataset path from the current data config."""
    dpath = current_cfg.get("dpath", f"data/{dataset}")
    if not os.path.isabs(dpath):
        dpath = os.path.normpath(os.path.join(ROOT, dpath))
    return os.path.normpath(dpath)


def _resolve_run_specific_configs(
    dataset: str,
    model: str,
    run_config: dict,
    kt_cfg: dict,
    data_config: dict,
) -> tuple[dict, dict, dict]:
    """Reconstruct the configs that were saved with the original training run.

    The run-specific train/model/dataset configs are authoritative. The only
    intentional substitution is ``dataset_config.dpath`` when the saved path is
    from another machine and does not exist locally; in that case the current
    data_config path is used while retaining the saved filenames, max_concepts,
    input_type, and other dataset metadata.
    """
    fallback_train = copy.deepcopy(kt_cfg.get("train_config", {}))
    fallback_model = copy.deepcopy(kt_cfg.get(model, {}))
    current_dataset = copy.deepcopy(data_config[dataset])

    train_cfg = copy.deepcopy(run_config.get("train_config") or fallback_train)
    model_cfg = copy.deepcopy(run_config.get("model_config") or fallback_model)
    dataset_cfg = copy.deepcopy(run_config.get("dataset_config") or current_dataset)

    saved_dpath = dataset_cfg.get("dpath")
    if not saved_dpath or not os.path.exists(saved_dpath):
        local_dpath = _resolve_current_dataset_dpath(dataset, current_dataset)
        if not os.path.exists(local_dpath):
            raise FileNotFoundError(
                "Neither the run-saved dataset path nor the current local dataset "
                f"path exists. saved={saved_dpath!r}, local={local_dpath!r}"
            )
        dataset_cfg["dpath"] = local_dpath
    else:
        dataset_cfg["dpath"] = os.path.normpath(saved_dpath)

    return train_cfg, model_cfg, dataset_cfg


def _build_model_like_training(
    model_name: str,
    model_cfg: dict,
    train_cfg: dict,
    dataset_cfg: dict,
    resolved_emb_type: str,
    device: str,
):
    """Mirror the supported-model portion of core.train_runner.train_one_fold."""
    model_kwargs = {
        k: v for k, v in model_cfg.items()
        if k not in TRAIN_RUNNER_OTHER_CONFIG_KEYS
    }

    # train_runner explicitly supplies num_pid for UKT.
    if model_name == "ukt":
        model_kwargs.setdefault("num_pid", dataset_cfg.get("num_q", 0))

    built_model = build_model(
        model_name,
        num_c=dataset_cfg["num_c"],
        num_q=dataset_cfg.get("num_q", 0),
        emb_type=resolved_emb_type,
        seq_len=train_cfg.get("seq_len"),
        device=device,
        dpath=dataset_cfg.get("dpath", ""),
        num_at=model_cfg.get("num_at"),
        num_it=model_cfg.get("num_it"),
        **model_kwargs,
    )

    # QIKTTrainer used model_cfg as trainer.other_config for prediction fusion.
    # Store a non-state attribute solely for the exporter adapter.
    if model_name == "qikt":
        built_model._export_eval_config = copy.deepcopy(model_cfg)

    return built_model

def _summarize_rows(rows: list[tuple[int, int, int, float]]) -> dict:
    if not rows:
        return {"n_samples": 0, "n_students": 0, "auc": None, "acc": None}
    y_true = np.asarray([r[2] for r in rows], dtype=np.int8)
    y_pred = np.asarray([r[3] for r in rows], dtype=np.float64)
    auc = float(roc_auc_score(y_true, y_pred)) if np.unique(y_true).size == 2 else None
    acc = float(np.mean((y_pred >= 0.5).astype(np.int8) == y_true))
    return {
        "n_samples": int(len(rows)),
        "n_students": int(len({r[0] for r in rows})),
        "auc": auc,
        "acc": acc,
    }


@app.command()
def main(
    dataset: str = typer.Option(..., "-d", help="Dataset name"),
    model: str = typer.Option(..., "-m", help="Model name"),
    cv_dir: Optional[str] = typer.Option(None, help="CV directory"),
    train_root: str = typer.Option("E:/project/knowledgeTracing/train_model", help="Train model root"),
    output_dir: Optional[str] = typer.Option(None, help="Output directory"),
    folds: str = typer.Option("0-4", help="Folds to export"),
    device: str = typer.Option("cuda:0", help="Device"),
    seed: int = typer.Option(3407, help="Seed"),
):
    """Export predictions from existing validation-best checkpoints."""
    dataset = normalize_dataset_name(dataset)
    model = model.lower()
    if model not in SUPPORTED_EXPORT_MODELS:
        print(
            f"[red]Unsupported model '{model}'. This statistical exporter supports "
            f"only: {', '.join(sorted(SUPPORTED_EXPORT_MODELS))}[/red]"
        )
        raise typer.Exit(2)
    train_path = Path(train_root)

    if cv_dir:
        cv_path = train_path / dataset / cv_dir
        if not cv_path.exists():
            cv_path = train_path / cv_dir
    else:
        cv_path = find_cv_dir(dataset, model, train_path)

    if not cv_path or not cv_path.exists():
        print(f"[red]CV directory not found for {dataset}/{model}[/red]")
        raise typer.Exit(1)

    print(f"[blue]Exporter version: {EXPORTER_VERSION}[/blue]")
    print(f"[blue]CV directory: {cv_path.name}[/blue]")
    out_path = Path(output_dir) if output_dir else cv_path / "predictions"
    out_path.mkdir(parents=True, exist_ok=True)

    if "-" in folds:
        start, end = map(int, folds.split("-"))
        fold_list = list(range(start, end + 1))
    else:
        fold_list = [int(f) for f in folds.split(",")]

    kt_cfg = load_cfg("configs/kt_config.json")
    data_config = load_cfg("configs/data_config.json")

    results = []
    set_seed(seed)

    for fold_id in fold_list:
        print(f"\n[yellow]Processing fold {fold_id}...[/yellow]")
        fold_dirs = list(cv_path.glob(f"{dataset}-{model}-fold{fold_id}-*"))
        if not fold_dirs:
            print(f"  [red]Fold {fold_id} directory not found[/red]")
            continue
        if len(fold_dirs) > 1:
            print(f"  [yellow]Multiple run directories found; using {sorted(fold_dirs)[-1].name}[/yellow]")
        run_dir = sorted(fold_dirs)[-1]

        run_config_path = run_dir / "run_config.json"
        run_config = {}
        if run_config_path.exists():
            with run_config_path.open() as f:
                run_config = json.load(f)

        try:
            ckpt_path = get_best_checkpoint(run_dir, run_config)
        except RuntimeError as exc:
            print(f"  [red]{exc}[/red]")
            continue
        if not ckpt_path:
            print("  [red]Checkpoint not found[/red]")
            continue
        print(f"  [green]Checkpoint: {ckpt_path.name}[/green]")

        try:
            train_cfg, model_cfg, dataset_cfg = _resolve_run_specific_configs(
                dataset=dataset,
                model=model,
                run_config=run_config,
                kt_cfg=kt_cfg,
                data_config=data_config,
            )

            # This is the critical parity point: train_runner passed the saved
            # train_config.dataset_mode to kt_test. For AKT/UKT/QIKT it is
            # "all_in_one" in the original runs, even though those model names are
            # not in this exporter's static ALL_IN_ONE_MODELS set.
            dataset_mode = train_cfg.get("dataset_mode")
            if dataset_mode is None:
                dataset_mode = "all_in_one" if model in ALL_IN_ONE_MODELS else None
            batch_size = int(train_cfg.get("batch_size", 64))

            print(f"  Run dataset_mode={dataset_mode}")
            print(f"  Run batch_size={batch_size}")
            print(f"  Dataset path={dataset_cfg.get('dpath')}")

            test_loader = build_fixed_test_loader(
                dataset=dataset,
                dataset_cfg=dataset_cfg,
                batch_size=batch_size,
                model_name=model,
                dataset_mode=dataset_mode,
            )

            resolved_emb_type = _resolve_emb_type(
                model_name=model,
                run_config=run_config,
                model_cfg=model_cfg,
                checkpoint_path=ckpt_path,
            )
            if resolved_emb_type is None:
                resolved_emb_type = "qid"
            print(f"  Resolved emb_type={resolved_emb_type}")

            built_model = _build_model_like_training(
                model_name=model,
                model_cfg=model_cfg,
                train_cfg=train_cfg,
                dataset_cfg=dataset_cfg,
                resolved_emb_type=resolved_emb_type,
                device=device,
            )

            state_dict = torch.load(ckpt_path, map_location=device)
            # Support checkpoints saved either as a raw state_dict or a wrapper dict.
            if isinstance(state_dict, dict) and "state_dict" in state_dict and isinstance(state_dict["state_dict"], dict):
                state_dict = state_dict["state_dict"]
            built_model.load_state_dict(state_dict)

            output_file = out_path / f"predictions_fold{fold_id}.csv"
            # Remove stale output before inference so a failed rerun cannot leave an
            # older invalid/partial file that the statistical script might consume.
            if output_file.exists():
                output_file.unlink()

            print(f"  Test loader batches: {len(test_loader)}")
            rows = run_inference(built_model, test_loader, device, model)
            summary = _summarize_rows(rows)
            if summary["n_samples"] <= 0 or summary["auc"] is None:
                raise RuntimeError("No valid binary test predictions were exported.")

            print(
                f"  Inference results: {summary['n_samples']} predictions from "
                f"{summary['n_students']} students"
            )
            print(f"  Exported test AUC={summary['auc']:.6f}, ACC={summary['acc']:.6f}")

            saved_auc = _find_saved_best_test_auc(run_dir)
            if saved_auc is not None:
                auc_diff = summary["auc"] - saved_auc
                print(f"  Saved best_test_auc={saved_auc:.6f}")
                print(f"  AUC difference={auc_diff:+.8f}")
                if abs(auc_diff) > 1e-6:
                    print(
                        "  [yellow]Warning: exported AUC does not exactly reproduce "
                        "the saved best_test_auc. Check checkpoint/config parity before "
                        "statistical testing.[/yellow]"
                    )
            else:
                print("  Saved best_test_auc: not uniquely found in run JSON artifacts")

            with output_file.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["fold", "student_id", "interaction_id", "y_true", "y_pred"])
                for sid, interaction_id, y_true, y_pred in rows:
                    writer.writerow([fold_id, sid, interaction_id, y_true, y_pred])

            print(f"  [green]Saved predictions to {output_file}[/green]")
            results.append({
                "fold": fold_id,
                "checkpoint": str(ckpt_path),
                "saved_best_test_auc": saved_auc,
                **summary,
            })

        except Exception as exc:
            print(f"  [red]Error: {exc}[/red]")
            import traceback
            traceback.print_exc()

    if results:
        summary = {
            "dataset": dataset,
            "model": model,
            "seed": seed,
            "n_folds": len(results),
            "total_samples": sum(r["n_samples"] for r in results),
            "folds": results,
        }
        with (out_path / "export_summary.json").open("w") as f:
            json.dump(summary, f, indent=2)
        print(f"\n[green]Done! Exported {len(results)} folds.[/green]")
        print(
            "[yellow]Before statistical testing, compare each exported fold AUC "
            "with the original experiment result. They should match (up to small "
            "floating-point differences).[/yellow]"
        )


if __name__ == "__main__":
    app()
