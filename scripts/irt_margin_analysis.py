"""IRT-stratified coverage-margin analysis for a removed model.

Formal analysis design
----------------------
For each of the five validation-best a removed model checkpoints, this script:

1. reconstructs the original run from ``run_config.json``;
2. evaluates the same fixed held-out test set used by training;
3. exports student-aligned factors at valid ``smasks`` positions;
4. verifies that exported ``y_pred`` reproduces the saved ``best_test_auc``;
5. splits ``base_logit`` into configurable equal-frequency IRT strata within that fold;
6. splits ``coverage_margin`` into Low/Mid/High tertiles *within each IRT stratum*;
7. computes empirical correctness for the resulting ``irt_bins x 3`` cells;
8. aggregates cell statistics across the five checkpoints without pooling the
   repeated test interactions.

The two primary analysis variables are exactly:

    irt_logit       = outputs["base_logit"]
    coverage_margin = outputs["coverage_margin"]

Usage
-----
Single dataset::

    python scripts/irt_margin_analysis.py -d algebra2005 --folds 0-4

All five paper datasets::

    python scripts/irt_margin_analysis.py batch
"""

from __future__ import annotations

import copy
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import typer
from rich import print
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

app = typer.Typer(add_completion=False, invoke_without_command=True)

SCRIPT_VERSION = "2026-07-31-irt-stratified-margin-v2"
MODEL_NAME = "removed_model"
FORMAL_FOLDS = [0, 1, 2, 3, 4]
DEFAULT_IRT_BINS = 5
MARGIN_LABELS = ["Low", "Mid", "High"]
AUC_PARITY_TOL = 1e-6

PAPER_DATASETS = [
    "algebra2005",
    "assist2009",
    "assist2017",
    "bridge2algebra2006",
    "nips_task34",
]

# Keep model construction consistent with core/train_runner.py. These values are
# trainer/loss-side configuration and were not forwarded into a removed model's model
# constructor during the original experiments.
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

# Raw audit fields. ``irt_logit`` is a renamed ``base_logit`` so the output file
# communicates the experimental role directly.
FACTOR_OUTPUT_MAP = {
    "y_pred": "y",
    "irt_logit": "base_logit",
    "coverage_margin": "coverage_margin",
    "theta": "theta",
    "difficulty": "difficulty",
    "shared_difficulty": "shared_difficulty",
    "effective_item_difficulty": "effective_item_difficulty",
    "discrimination": "discrimination",
    "student_radius": "student_radius",
    "question_radius": "question_radius",
    "center_dist": "center_dist",
    "coverage_logit": "coverage_logit",
    "coverage_gate": "coverage_gate",
    "gated_coverage_logit": "gated_coverage_logit",
}

RAW_COLUMNS = [
    "fold", "student_id", "interaction_id", "y_true", *FACTOR_OUTPUT_MAP.keys()
]


def make_irt_labels(irt_bins: int) -> list[str]:
    """Return ordered IRT stratum labels Q1..Qk for a validated bin count."""
    irt_bins = int(irt_bins)
    if irt_bins < 2:
        raise ValueError(f"irt_bins must be at least 2, got {irt_bins}")
    return [f"Q{i}" for i in range(1, irt_bins + 1)]


def _load_project_runtime():
    """Import project-specific modules lazily so pure analysis helpers are testable."""
    import core.trainers  # noqa: F401
    import datasets.init_dataset  # noqa: F401
    import models  # noqa: F401
    from core.config import load_cfg
    from core.dataset_names import normalize_dataset_name
    from core.factory import build_dataset, build_model
    from core.run_support import set_seed

    return {
        "load_cfg": load_cfg,
        "normalize_dataset_name": normalize_dataset_name,
        "build_dataset": build_dataset,
        "build_model": build_model,
        "set_seed": set_seed,
    }


# ---------------------------------------------------------------------------
# Pure analysis helpers
# ---------------------------------------------------------------------------


def strict_quantile_groups(
    values: pd.Series,
    q: int,
    labels: list[str],
    name: str,
) -> pd.Series:
    """Create exactly ``q`` equal-frequency groups or fail explicitly.

    We intentionally do not use ``duplicates='drop'`` because silently reducing
    the number of bins changes the pre-specified experiment.
    """
    series = pd.Series(values, copy=False)
    if len(labels) != q:
        raise ValueError(f"{name}: expected {q} labels, got {len(labels)}")
    if len(series) < q:
        raise ValueError(f"{name}: cannot form {q} quantile groups from {len(series)} rows")
    numeric = pd.to_numeric(series, errors="coerce")
    if numeric.isna().any() or not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError(f"{name}: quantile input contains NaN or Inf")

    try:
        grouped = pd.qcut(numeric, q=q, labels=labels, duplicates="raise")
    except ValueError as exc:
        raise ValueError(
            f"{name}: cannot form exactly {q} quantile groups. "
            "This usually means too many tied values at quantile boundaries."
        ) from exc

    result = pd.Series(grouped, index=series.index, name=f"{name}_group")
    if result.isna().any():
        raise ValueError(f"{name}: quantile grouping produced missing assignments")
    observed = set(result.astype(str).unique())
    expected = set(labels)
    if observed != expected:
        raise ValueError(
            f"{name}: expected groups {labels}, observed {sorted(observed)}"
        )
    return result


def _validate_analysis_dataframe(df: pd.DataFrame, fold_id: int) -> None:
    required = {"y_true", "y_pred", "irt_logit", "coverage_margin"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Fold {fold_id}: missing analysis columns {sorted(missing)}")
    if df.empty:
        raise ValueError(f"Fold {fold_id}: analysis dataframe is empty")
    y = pd.to_numeric(df["y_true"], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all():
        raise ValueError(f"Fold {fold_id}: y_true must be binary")
    for column in ("y_pred", "irt_logit", "coverage_margin"):
        values = pd.to_numeric(df[column], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Fold {fold_id}: {column} contains NaN or Inf")


def stratify_fold_dataframe(
    df: pd.DataFrame,
    fold_id: int,
    irt_bins: int = DEFAULT_IRT_BINS,
) -> pd.DataFrame:
    """Assign equal-frequency IRT strata and within-stratum margin tertiles."""
    _validate_analysis_dataframe(df, fold_id)
    irt_labels = make_irt_labels(irt_bins)
    work = df.copy()
    work["irt_stratum"] = strict_quantile_groups(
        work["irt_logit"], q=irt_bins, labels=irt_labels, name="irt_logit"
    )
    work["margin_group"] = pd.Series(index=work.index, dtype="object")

    for irt_label in irt_labels:
        idx = work.index[work["irt_stratum"].astype(str) == irt_label]
        if len(idx) < 3:
            raise ValueError(
                f"Fold {fold_id}/{irt_label}: only {len(idx)} rows; cannot form margin tertiles"
            )
        work.loc[idx, "margin_group"] = strict_quantile_groups(
            work.loc[idx, "coverage_margin"],
            q=3,
            labels=MARGIN_LABELS,
            name=f"coverage_margin[{irt_label}]",
        ).astype(str)

    if work["margin_group"].isna().any():
        raise ValueError(f"Fold {fold_id}: some rows were not assigned a margin tertile")
    work["irt_stratum"] = pd.Categorical(
        work["irt_stratum"].astype(str), categories=irt_labels, ordered=True
    )
    work["margin_group"] = pd.Categorical(
        work["margin_group"].astype(str), categories=MARGIN_LABELS, ordered=True
    )
    return work


def analyze_fold_dataframe(
    df: pd.DataFrame,
    fold_id: int,
    irt_bins: int = DEFAULT_IRT_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute one fold's ``irt_bins x 3`` cells and High-Low gaps."""
    irt_labels = make_irt_labels(irt_bins)
    work = stratify_fold_dataframe(df, fold_id, irt_bins=irt_bins)
    cell_rows: list[dict] = []
    gap_rows: list[dict] = []

    for irt_label in irt_labels:
        qdf = work[work["irt_stratum"] == irt_label]
        correctness_by_margin: dict[str, float] = {}
        for margin_label in MARGIN_LABELS:
            cell = qdf[qdf["margin_group"] == margin_label]
            if cell.empty:
                raise ValueError(
                    f"Fold {fold_id}/{irt_label}/{margin_label}: empty analysis cell"
                )
            correctness = float(cell["y_true"].mean())
            correctness_by_margin[margin_label] = correctness
            cell_rows.append({
                "fold": int(fold_id),
                "irt_stratum": irt_label,
                "margin_group": margin_label,
                "n_interactions": int(len(cell)),
                "n_students": int(cell["student_id"].nunique())
                if "student_id" in cell.columns else None,
                "correctness": correctness,
                "irt_logit_mean": float(cell["irt_logit"].mean()),
                "coverage_margin_mean": float(cell["coverage_margin"].mean()),
                "y_pred_mean": float(cell["y_pred"].mean()),
            })

        low = correctness_by_margin["Low"]
        mid = correctness_by_margin["Mid"]
        high = correctness_by_margin["High"]
        gap_rows.append({
            "fold": int(fold_id),
            "irt_stratum": irt_label,
            "low_correctness": low,
            "mid_correctness": mid,
            "high_correctness": high,
            "high_low_gap": high - low,
            "monotonic": bool(low < mid < high),
        })

    cells = pd.DataFrame(cell_rows)
    gaps = pd.DataFrame(gap_rows)
    expected_cells = irt_bins * len(MARGIN_LABELS)
    expected_gaps = irt_bins
    if len(cells) != expected_cells or len(gaps) != expected_gaps:
        raise AssertionError(
            f"Fold {fold_id}: expected {expected_cells} cells and {expected_gaps} gaps, "
            f"got {len(cells)} and {len(gaps)}"
        )
    return cells, gaps


def aggregate_fold_statistics(
    cells: pd.DataFrame,
    gaps: pd.DataFrame,
    expected_folds: Iterable[int],
    irt_bins: int = DEFAULT_IRT_BINS,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Aggregate checkpoint-specific statistics without pooling interactions."""
    irt_labels = make_irt_labels(irt_bins)
    expected = sorted({int(f) for f in expected_folds})
    observed_cells = sorted({int(f) for f in cells["fold"].unique()})
    observed_gaps = sorted({int(f) for f in gaps["fold"].unique()})
    if observed_cells != expected or observed_gaps != expected:
        raise ValueError(
            "Fold mismatch in aggregate statistics: "
            f"expected={expected}, cells={observed_cells}, gaps={observed_gaps}"
        )

    cell_rows: list[dict] = []
    for irt_label in irt_labels:
        for margin_label in MARGIN_LABELS:
            part = cells[
                (cells["irt_stratum"].astype(str) == irt_label)
                & (cells["margin_group"].astype(str) == margin_label)
            ].sort_values("fold")
            if len(part) != len(expected):
                raise ValueError(
                    f"{irt_label}/{margin_label}: expected {len(expected)} fold rows, got {len(part)}"
                )
            cell_rows.append({
                "irt_stratum": irt_label,
                "margin_group": margin_label,
                "correctness_mean": float(part["correctness"].mean()),
                "correctness_std": float(part["correctness"].std(ddof=1)),
                "n_interactions_mean": float(part["n_interactions"].mean()),
                "n_students_mean": float(part["n_students"].mean())
                if part["n_students"].notna().all() else None,
                "irt_logit_mean": float(part["irt_logit_mean"].mean()),
                "coverage_margin_mean": float(part["coverage_margin_mean"].mean()),
                "y_pred_mean": float(part["y_pred_mean"].mean()),
            })

    gap_rows: list[dict] = []
    for irt_label in irt_labels:
        part = gaps[gaps["irt_stratum"].astype(str) == irt_label].sort_values("fold")
        if len(part) != len(expected):
            raise ValueError(
                f"{irt_label}: expected {len(expected)} gap rows, got {len(part)}"
            )
        gap_rows.append({
            "irt_stratum": irt_label,
            "high_low_gap_mean": float(part["high_low_gap"].mean()),
            "high_low_gap_std": float(part["high_low_gap"].std(ddof=1)),
            "low_correctness_mean": float(part["low_correctness"].mean()),
            "mid_correctness_mean": float(part["mid_correctness"].mean()),
            "high_correctness_mean": float(part["high_correctness"].mean()),
            "monotonic_count": int(part["monotonic"].astype(bool).sum()),
            "n_folds": int(len(part)),
        })

    monotonic_count = int(gaps["monotonic"].astype(bool).sum())
    monotonic_total = int(len(gaps))
    metadata = {
        "n_folds": len(expected),
        "folds": expected,
        "monotonic_strata_count": monotonic_count,
        "monotonic_strata_total": monotonic_total,
        "monotonic_rate": float(monotonic_count / monotonic_total)
        if monotonic_total else None,
        "mean_high_low_gap": float(gaps["high_low_gap"].mean()),
    }
    return pd.DataFrame(cell_rows), pd.DataFrame(gap_rows), metadata


# ---------------------------------------------------------------------------
# Student-aligned factor collection
# ---------------------------------------------------------------------------


def _as_2d(tensor: torch.Tensor, name: str, batch_size: int) -> torch.Tensor:
    if tensor is None:
        raise ValueError(f"{name} is None")
    if tensor.dim() == 3 and tensor.size(-1) == 1:
        tensor = tensor.squeeze(-1)
    if tensor.dim() == 1:
        if tensor.numel() == batch_size:
            tensor = tensor.reshape(batch_size, 1)
        else:
            raise ValueError(
                f"Cannot map 1-D {name} with {tensor.numel()} values to batch_size={batch_size}"
            )
    if tensor.dim() != 2:
        raise ValueError(f"Expected {name} [batch,time], got {tuple(tensor.shape)}")
    return tensor


def collect_factor_batch(
    student_ids,
    target: torch.Tensor,
    mask: torch.Tensor,
    outputs: dict,
    interaction_counters: dict,
    fold_id: int,
) -> list[dict]:
    """Collect all required a removed model factors at exactly the same valid positions."""
    if student_ids is None:
        raise ValueError("Test batch has no student_id/uid")
    if torch.is_tensor(student_ids):
        sid_values = student_ids.detach().cpu().reshape(-1).tolist()
    else:
        sid_values = np.asarray(student_ids).reshape(-1).tolist()
    batch_size = len(sid_values)

    target_cpu = _as_2d(target, "target", batch_size).detach().cpu()
    mask_cpu = _as_2d(mask.bool(), "smasks", batch_size).detach().cpu()
    if target_cpu.shape != mask_cpu.shape:
        raise ValueError(
            f"target/smasks shape mismatch: {tuple(target_cpu.shape)} vs {tuple(mask_cpu.shape)}"
        )

    missing_keys = sorted(set(FACTOR_OUTPUT_MAP.values()) - set(outputs))
    if missing_keys:
        raise KeyError(f"a removed model output missing analysis fields: {missing_keys}")

    factor_matrices: dict[str, torch.Tensor] = {}
    for column, output_key in FACTOR_OUTPUT_MAP.items():
        matrix = _as_2d(outputs[output_key], output_key, batch_size).detach().cpu()
        if matrix.shape != target_cpu.shape:
            raise ValueError(
                f"{output_key}/target shape mismatch: {tuple(matrix.shape)} vs {tuple(target_cpu.shape)}"
            )
        factor_matrices[column] = matrix

    rows: list[dict] = []
    for b, sid_raw in enumerate(sid_values):
        sid = int(sid_raw)
        positions = torch.nonzero(mask_cpu[b], as_tuple=False).reshape(-1).tolist()
        for t in positions:
            y_float = float(target_cpu[b, t].item())
            y_true = int(round(y_float))
            if y_true not in (0, 1) or not np.isclose(y_float, y_true, atol=1e-6):
                raise ValueError(f"Non-binary y_true={y_float} for student={sid}, t={t}")

            row = {
                "fold": int(fold_id),
                "student_id": sid,
                "interaction_id": int(interaction_counters[sid]),
                "y_true": y_true,
            }
            interaction_counters[sid] += 1
            for column, matrix in factor_matrices.items():
                value = float(matrix[b, t].item())
                if not np.isfinite(value):
                    raise ValueError(
                        f"Non-finite {column} for student={sid}, interaction={row['interaction_id']}"
                    )
                row[column] = value
            if not (0.0 <= row["y_pred"] <= 1.0):
                raise ValueError(f"y_pred outside [0,1]: {row['y_pred']}")
            rows.append(row)
    return rows


def _concat_full(seqs, shifted):
    if seqs is None or shifted is None or seqs.numel() == 0:
        return None
    return torch.cat([seqs[:, :1], shifted], dim=1)


def _nonempty(tensor):
    return tensor if tensor is not None and tensor.numel() > 0 else None


def run_removed_model_factor_inference(
    model,
    test_loader,
    device: str,
    fold_id: int,
) -> pd.DataFrame:
    """Run a removed model on the fixed test set and preserve factor alignment."""
    model.eval()
    model.to(device)
    rows: list[dict] = []
    interaction_counters = defaultdict(int)

    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader, start=1):
            qseqs = _nonempty(batch.get("qseqs"))
            cseqs = _nonempty(batch.get("cseqs"))
            qshft = _nonempty(batch.get("shft_qseqs"))
            cshft = _nonempty(batch.get("shft_cseqs"))
            rseqs = _nonempty(batch.get("rseqs"))
            rshft = _nonempty(batch.get("shft_rseqs"))
            smasks = batch.get("smasks")
            student_ids = batch.get("student_id", batch.get("uid"))

            if any(x is None for x in (qseqs, cseqs, qshft, cshft, rseqs, rshft, smasks)):
                raise RuntimeError(f"Fold {fold_id}, batch {batch_idx}: missing required sequence/mask")

            q_full = _concat_full(qseqs.to(device), qshft.to(device))
            c_full = _concat_full(cseqs.to(device), cshft.to(device))
            r_full = _concat_full(rseqs.to(device), rshft.to(device))
            if q_full is None or c_full is None or r_full is None:
                raise RuntimeError(f"Fold {fold_id}, batch {batch_idx}: cannot construct full sequences")

            outputs = model(q_full.long(), c_full.long(), r_full.float())
            if not isinstance(outputs, dict):
                raise RuntimeError(f"Fold {fold_id}, batch {batch_idx}: a removed model output is not a dict")

            batch_rows = collect_factor_batch(
                student_ids=student_ids,
                target=rshft.to(device).float(),
                mask=smasks.to(device).bool(),
                outputs=outputs,
                interaction_counters=interaction_counters,
                fold_id=fold_id,
            )
            rows.extend(batch_rows)

    if not rows:
        raise RuntimeError(f"Fold {fold_id}: inference produced no valid test rows")
    print(f"  Processed {len(test_loader)} batches, 0 errors")
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


# ---------------------------------------------------------------------------
# Run-config-parity reconstruction
# ---------------------------------------------------------------------------


def find_cv_dir(dataset: str, train_root: Path) -> Optional[Path]:
    dataset_dir = train_root / dataset
    if dataset_dir.exists():
        matches = list(dataset_dir.glob(f"cv-{dataset}-{MODEL_NAME}-*"))
        if matches:
            return sorted(matches)[-1]
    matches = list(train_root.glob(f"cv-{dataset}-{MODEL_NAME}-*"))
    return sorted(matches)[-1] if matches else None


def get_best_checkpoint(run_dir: Path, run_config: Optional[dict] = None) -> Optional[Path]:
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
        f"Ambiguous validation-best checkpoint in {run_dir}: {names}"
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


def find_saved_best_test_auc(run_dir: Path) -> Optional[float]:
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
    if isinstance(obj, dict):
        for k, value in obj.items():
            if k == key and isinstance(value, str) and value.strip():
                out.append(value.strip())
            else:
                _collect_string_values_for_key(value, key, out)
    elif isinstance(obj, list):
        for value in obj:
            _collect_string_values_for_key(value, key, out)


def resolve_emb_type(run_config: dict, model_cfg: dict, checkpoint_path: Path) -> str:
    run_values: list[str] = []
    _collect_string_values_for_key(run_config, "emb_type", run_values)
    unique = list(dict.fromkeys(run_values))
    if len(unique) == 1:
        return unique[0]
    if len(unique) > 1:
        raise RuntimeError(f"Ambiguous emb_type in run_config: {unique}")
    value = model_cfg.get("emb_type")
    if isinstance(value, str) and value.strip():
        return value.strip()
    name = checkpoint_path.name.lower()
    if "stoc_qid" in name:
        return "stoc_qid"
    if "iekt" in name:
        return "iekt"
    if "qid" in name:
        return "qid"
    return "qid"


def _resolve_current_dataset_dpath(dataset: str, current_cfg: dict) -> str:
    dpath = current_cfg.get("dpath", f"data/{dataset}")
    if not os.path.isabs(dpath):
        dpath = os.path.normpath(os.path.join(ROOT, dpath))
    return os.path.normpath(dpath)


def resolve_run_specific_configs(
    dataset: str,
    run_config: dict,
    kt_cfg: dict,
    data_config: dict,
) -> tuple[dict, dict, dict]:
    fallback_train = copy.deepcopy(kt_cfg.get("train_config", {}))
    fallback_model = copy.deepcopy(kt_cfg.get(MODEL_NAME, {}))
    current_dataset = copy.deepcopy(data_config[dataset])

    train_cfg = copy.deepcopy(run_config.get("train_config") or fallback_train)
    model_cfg = copy.deepcopy(run_config.get("model_config") or fallback_model)
    dataset_cfg = copy.deepcopy(run_config.get("dataset_config") or current_dataset)

    saved_dpath = dataset_cfg.get("dpath")
    if not saved_dpath or not os.path.exists(saved_dpath):
        local_dpath = _resolve_current_dataset_dpath(dataset, current_dataset)
        if not os.path.exists(local_dpath):
            raise FileNotFoundError(
                "Neither saved nor current dataset path exists: "
                f"saved={saved_dpath!r}, local={local_dpath!r}"
            )
        dataset_cfg["dpath"] = local_dpath
    else:
        dataset_cfg["dpath"] = os.path.normpath(saved_dpath)
    return train_cfg, model_cfg, dataset_cfg


def build_removed_model_like_training(
    runtime: dict,
    model_cfg: dict,
    train_cfg: dict,
    dataset_cfg: dict,
    resolved_emb_type: str,
    device: str,
):
    model_kwargs = {
        k: v for k, v in model_cfg.items() if k not in TRAIN_RUNNER_OTHER_CONFIG_KEYS
    }
    return runtime["build_model"](
        MODEL_NAME,
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


def build_fixed_test_loader(
    runtime: dict,
    dataset: str,
    dataset_cfg: dict,
    batch_size: int,
    dataset_mode: Optional[str],
):
    print("  Using fixed held-out test set shared across CV runs.")
    return runtime["build_dataset"](
        "kt_test",
        dataset_name=dataset,
        data_config=dataset_cfg,
        batch_size=batch_size,
        model_name=MODEL_NAME,
        dataset_mode=dataset_mode,
    )


def _load_state_dict(path: Path, device: str):
    try:
        payload = torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location=device)
    if isinstance(payload, dict) and "state_dict" in payload and isinstance(payload["state_dict"], dict):
        payload = payload["state_dict"]
    return payload


def parse_folds(folds: str) -> list[int]:
    if "-" in folds:
        parts = folds.split("-")
        if len(parts) != 2:
            raise ValueError(f"Invalid fold range: {folds}")
        start, end = map(int, parts)
        values = list(range(start, end + 1))
    else:
        values = [int(x.strip()) for x in folds.split(",") if x.strip()]
    values = sorted(set(values))
    if values != FORMAL_FOLDS:
        raise ValueError(
            f"Formal IRT-margin analysis requires exactly folds {FORMAL_FOLDS}, got {values}"
        )
    return values


# ---------------------------------------------------------------------------
# Output and plotting
# ---------------------------------------------------------------------------


def _json_default(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if pd.isna(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def plot_margin_line(
    cell_summary: pd.DataFrame,
    output_file: Path,
    dataset: str,
    irt_bins: int = DEFAULT_IRT_BINS,
) -> None:
    irt_labels = make_irt_labels(irt_bins)
    fig_height = max(4.8, 0.34 * irt_bins + 3.2)
    fig, ax = plt.subplots(figsize=(7.2, fig_height))
    x = np.arange(len(MARGIN_LABELS))
    for irt_label in irt_labels:
        part = cell_summary[cell_summary["irt_stratum"] == irt_label].copy()
        part["_order"] = part["margin_group"].map({m: i for i, m in enumerate(MARGIN_LABELS)})
        part = part.sort_values("_order")
        ax.errorbar(
            x,
            part["correctness_mean"].to_numpy(),
            yerr=part["correctness_std"].to_numpy(),
            marker="o",
            capsize=3,
            label=irt_label,
        )
    ax.set_xticks(x, MARGIN_LABELS)
    ax.set_xlabel("Coverage Margin Group")
    ax.set_ylabel("Empirical Correctness")
    ax.set_title(f"IRT-stratified Coverage Margin Analysis — {dataset}")
    ax.legend(title="IRT Stratum")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_margin_heatmap(
    cell_summary: pd.DataFrame,
    output_file: Path,
    dataset: str,
    irt_bins: int = DEFAULT_IRT_BINS,
) -> None:
    irt_labels = make_irt_labels(irt_bins)
    matrix = np.zeros((len(irt_labels), len(MARGIN_LABELS)), dtype=float)
    for i, irt_label in enumerate(irt_labels):
        for j, margin_label in enumerate(MARGIN_LABELS):
            row = cell_summary[
                (cell_summary["irt_stratum"] == irt_label)
                & (cell_summary["margin_group"] == margin_label)
            ]
            if len(row) != 1:
                raise ValueError(f"Heatmap missing/duplicate cell {irt_label}/{margin_label}")
            matrix[i, j] = float(row.iloc[0]["correctness_mean"])

    fig, ax = plt.subplots(figsize=(6.4, max(5.0, 0.45 * irt_bins + 2.5)))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xticks(np.arange(len(MARGIN_LABELS)), MARGIN_LABELS)
    ax.set_yticks(np.arange(len(irt_labels)), irt_labels)
    ax.set_xlabel("Coverage Margin Group")
    ax.set_ylabel("IRT Stratum")
    ax.set_title(f"Empirical Correctness — {dataset}")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center")
    fig.colorbar(image, ax=ax, label="Empirical Correctness")
    fig.tight_layout()
    fig.savefig(output_file, format="pdf", bbox_inches="tight")
    plt.close(fig)


def write_analysis_outputs(
    dataset: str,
    output_dir: Path,
    fold_frames: dict[int, pd.DataFrame],
    fold_cells: pd.DataFrame,
    fold_gaps: pd.DataFrame,
    cell_summary: pd.DataFrame,
    gap_summary: pd.DataFrame,
    aggregate_meta: dict,
    fold_run_meta: list[dict],
    seed: int,
    irt_bins: int = DEFAULT_IRT_BINS,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for fold_id, df in sorted(fold_frames.items()):
        df.to_csv(output_dir / f"raw_fold{fold_id}.csv", index=False)
    fold_cells.to_csv(output_dir / "fold_cell_statistics.csv", index=False)
    cell_summary.to_csv(output_dir / "cell_statistics.csv", index=False)
    gap_summary.to_csv(output_dir / "high_low_gap.csv", index=False)

    plot_margin_line(
        cell_summary, output_dir / "margin_lineplot.pdf", dataset, irt_bins=irt_bins
    )
    plot_margin_heatmap(
        cell_summary, output_dir / "margin_heatmap.pdf", dataset, irt_bins=irt_bins
    )

    summary = {
        "script_version": SCRIPT_VERSION,
        "dataset": dataset,
        "model": MODEL_NAME,
        "seed": int(seed),
        "method": {
            "test_set": "same fixed held-out interactions in every CV-trained checkpoint",
            "irt_stratification": f"{irt_bins} equal-frequency bins per fold using a removed model base_logit",
            "irt_stratification_bins": int(irt_bins),
            "irt_logit_definition": "outputs['base_logit'] = discrimination * (theta - difficulty)",
            "margin_stratification": "3 equal-frequency bins within each IRT stratum per fold",
            "coverage_margin_definition": "student_radius - center_dist - question_radius",
            "cell_outcome": "empirical mean of y_true",
            "aggregation": "mean and sample std of cell correctness across five checkpoints; repeated test interactions are not pooled",
            "auc_parity_tolerance": AUC_PARITY_TOL,
        },
        "aggregate": aggregate_meta,
        "fold_runs": fold_run_meta,
        "cell_statistics": cell_summary.to_dict(orient="records"),
        "high_low_gap": gap_summary.to_dict(orient="records"),
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=_json_default)
    return summary_path


# ---------------------------------------------------------------------------
# Dataset orchestration
# ---------------------------------------------------------------------------


def run_dataset_analysis(
    dataset: str,
    train_root: Path,
    output_root: Path,
    folds: list[int],
    device: str,
    seed: int,
    cv_dir: Optional[str] = None,
    irt_bins: int = DEFAULT_IRT_BINS,
) -> dict:
    irt_labels = make_irt_labels(irt_bins)
    runtime = _load_project_runtime()
    dataset = runtime["normalize_dataset_name"](dataset)
    runtime["set_seed"](seed)

    if cv_dir:
        cv_path = train_root / dataset / cv_dir
        if not cv_path.exists():
            cv_path = train_root / cv_dir
    else:
        cv_path = find_cv_dir(dataset, train_root)
    if not cv_path or not cv_path.exists():
        raise FileNotFoundError(f"a removed model CV directory not found for {dataset}")

    print(f"[blue]IRT-margin script version: {SCRIPT_VERSION}[/blue]")
    print(f"[blue]Dataset: {dataset}[/blue]")
    print(f"[blue]CV directory: {cv_path}[/blue]")
    print(f"[blue]IRT strata: {irt_bins} ({irt_labels[0]}-{irt_labels[-1]})[/blue]")

    # Match the already verified exporter by using the current configs only as
    # fallbacks/path providers; run_config remains authoritative.
    kt_cfg = runtime["load_cfg"]("configs/kt_config.json")
    data_config = runtime["load_cfg"]("configs/data_config.json")

    dataset_output = output_root / dataset
    dataset_output.mkdir(parents=True, exist_ok=True)

    fold_frames: dict[int, pd.DataFrame] = {}
    cell_frames: list[pd.DataFrame] = []
    gap_frames: list[pd.DataFrame] = []
    fold_run_meta: list[dict] = []

    for fold_id in folds:
        print(f"\n[yellow]Processing fold {fold_id}...[/yellow]")
        fold_dirs = sorted(cv_path.glob(f"{dataset}-{MODEL_NAME}-fold{fold_id}-*"))
        if len(fold_dirs) != 1:
            if not fold_dirs:
                raise FileNotFoundError(f"Fold {fold_id} run directory not found in {cv_path}")
            raise RuntimeError(
                f"Fold {fold_id}: expected one run directory, found {[p.name for p in fold_dirs]}"
            )
        run_dir = fold_dirs[0]
        run_config_path = run_dir / "run_config.json"
        if not run_config_path.exists():
            raise FileNotFoundError(f"Missing {run_config_path}")
        with run_config_path.open(encoding="utf-8") as f:
            run_config = json.load(f)

        ckpt_path = get_best_checkpoint(run_dir, run_config)
        if ckpt_path is None:
            raise FileNotFoundError(f"Fold {fold_id}: validation-best checkpoint not found")
        print(f"  Checkpoint: {ckpt_path.name}")

        train_cfg, model_cfg, dataset_cfg = resolve_run_specific_configs(
            dataset=dataset,
            run_config=run_config,
            kt_cfg=kt_cfg,
            data_config=data_config,
        )
        dataset_mode = train_cfg.get("dataset_mode") or "all_in_one"
        batch_size = int(train_cfg.get("batch_size", 64))
        resolved_emb_type = resolve_emb_type(run_config, model_cfg, ckpt_path)
        print(f"  Run dataset_mode={dataset_mode}")
        print(f"  Run batch_size={batch_size}")
        print(f"  Resolved emb_type={resolved_emb_type}")

        test_loader = build_fixed_test_loader(
            runtime=runtime,
            dataset=dataset,
            dataset_cfg=dataset_cfg,
            batch_size=batch_size,
            dataset_mode=dataset_mode,
        )
        model = build_removed_model_like_training(
            runtime=runtime,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            dataset_cfg=dataset_cfg,
            resolved_emb_type=resolved_emb_type,
            device=device,
        ).to(device)
        model.load_state_dict(_load_state_dict(ckpt_path, device))

        prediction_head = getattr(model, "prediction_head", "irt")
        if str(prediction_head).lower() != "irt":
            raise RuntimeError(
                f"Fold {fold_id}: IRT-stratified analysis requires prediction_head='irt', got {prediction_head!r}"
            )

        df = run_removed_model_factor_inference(model, test_loader, device, fold_id)
        y_true = df["y_true"].to_numpy(dtype=int)
        y_pred = df["y_pred"].to_numpy(dtype=float)
        if np.unique(y_true).size != 2:
            raise RuntimeError(f"Fold {fold_id}: test targets contain fewer than two classes")
        exported_auc = float(roc_auc_score(y_true, y_pred))
        saved_auc = find_saved_best_test_auc(run_dir)
        if saved_auc is None:
            raise RuntimeError(
                f"Fold {fold_id}: could not find a unique saved best_test_auc; parity cannot be verified"
            )
        auc_difference = exported_auc - saved_auc
        print(f"  Exported test AUC={exported_auc:.8f}")
        print(f"  Saved best_test_auc={saved_auc:.8f}")
        print(f"  AUC difference={auc_difference:+.8f}")
        if abs(auc_difference) > AUC_PARITY_TOL:
            raise RuntimeError(
                f"Fold {fold_id}: AUC parity failed: |{auc_difference:+.8f}| > {AUC_PARITY_TOL}"
            )

        cells, gaps = analyze_fold_dataframe(df, fold_id, irt_bins=irt_bins)
        fold_frames[fold_id] = df
        cell_frames.append(cells)
        gap_frames.append(gaps)
        fold_run_meta.append({
            "fold": int(fold_id),
            "run_dir": str(run_dir),
            "checkpoint": str(ckpt_path),
            "n_interactions": int(len(df)),
            "n_students": int(df["student_id"].nunique()),
            "exported_auc": exported_auc,
            "saved_best_test_auc": saved_auc,
            "auc_difference": auc_difference,
        })

    fold_cells = pd.concat(cell_frames, ignore_index=True)
    fold_gaps = pd.concat(gap_frames, ignore_index=True)
    cell_summary, gap_summary, aggregate_meta = aggregate_fold_statistics(
        fold_cells, fold_gaps, expected_folds=folds, irt_bins=irt_bins
    )

    summary_path = write_analysis_outputs(
        dataset=dataset,
        output_dir=dataset_output,
        fold_frames=fold_frames,
        fold_cells=fold_cells,
        fold_gaps=fold_gaps,
        cell_summary=cell_summary,
        gap_summary=gap_summary,
        aggregate_meta=aggregate_meta,
        fold_run_meta=fold_run_meta,
        seed=seed,
        irt_bins=irt_bins,
    )

    print("\n[green]IRT-stratified margin analysis complete.[/green]")
    print(f"  Monotonic strata: {aggregate_meta['monotonic_strata_count']}/{aggregate_meta['monotonic_strata_total']}")
    print(f"  Mean High-Low gap: {aggregate_meta['mean_high_low_gap']:+.6f}")
    print(f"  Summary: {summary_path}")
    return {
        "dataset": dataset,
        "output_dir": str(dataset_output),
        "summary_path": str(summary_path),
        "irt_bins": int(irt_bins),
        **aggregate_meta,
    }


def _resolve_output_root(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    dataset: Optional[str] = typer.Option(None, "--dataset", "-d", help="Dataset name"),
    train_root: str = typer.Option(
        "E:/project/knowledgeTracing/train_model", "--train-root", help="Root directory for trained models"
    ),
    output_root: str = typer.Option(
        "output/irt_margin_analysis", "--output-root", help="Root directory for analysis outputs"
    ),
    folds: str = typer.Option("0-4", "--folds", help="Formal analysis folds; must be 0-4"),
    device: str = typer.Option("cuda:0", "--device", help="PyTorch device"),
    seed: int = typer.Option(3407, "--seed", help="Random seed"),
    irt_bins: int = typer.Option(DEFAULT_IRT_BINS, "--irt-bins", min=2, help="Number of equal-frequency IRT strata per fold"),
    cv_dir: Optional[str] = typer.Option(None, "--cv-dir", help="Explicit a removed model CV directory name"),
):
    """Run one dataset when no subcommand is supplied."""
    if ctx.invoked_subcommand is not None:
        return
    if not dataset:
        print("[red]Provide -d/--dataset for a single dataset, or use the 'batch' command.[/red]")
        raise typer.Exit(2)
    try:
        fold_list = parse_folds(folds)
        run_dataset_analysis(
            dataset=dataset,
            train_root=Path(train_root),
            output_root=_resolve_output_root(output_root),
            folds=fold_list,
            device=device,
            seed=seed,
            cv_dir=cv_dir,
            irt_bins=irt_bins,
        )
    except Exception as exc:
        print(f"[red]Analysis failed: {exc}[/red]")
        raise typer.Exit(1) from exc


@app.command("batch")
def batch_command(
    datasets: str = typer.Option(
        ",".join(PAPER_DATASETS), "--datasets", help="Comma-separated dataset names"
    ),
    train_root: str = typer.Option(
        "E:/project/knowledgeTracing/train_model", "--train-root", help="Root directory for trained models"
    ),
    output_root: str = typer.Option(
        "output/irt_margin_analysis", "--output-root", help="Root directory for analysis outputs"
    ),
    folds: str = typer.Option("0-4", "--folds", help="Formal analysis folds; must be 0-4"),
    device: str = typer.Option("cuda:0", "--device", help="PyTorch device"),
    seed: int = typer.Option(3407, "--seed", help="Random seed"),
    irt_bins: int = typer.Option(DEFAULT_IRT_BINS, "--irt-bins", min=2, help="Number of equal-frequency IRT strata per fold"),
):
    """Run the formal analysis across the five paper datasets."""
    fold_list = parse_folds(folds)
    dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
    if not dataset_list:
        raise typer.BadParameter("No datasets supplied")

    root = _resolve_output_root(output_root)
    results = []
    for dataset in dataset_list:
        print("\n" + "=" * 90)
        print(f"[bold]IRT-stratified Margin Analysis: {dataset}[/bold]")
        print("=" * 90)
        try:
            result = run_dataset_analysis(
                dataset=dataset,
                train_root=Path(train_root),
                output_root=root,
                folds=fold_list,
                device=device,
                seed=seed,
                irt_bins=irt_bins,
            )
            results.append(result)
        except Exception as exc:
            print(f"[red]Batch aborted on {dataset}: {exc}[/red]")
            # Formal batch analysis must be complete; never silently summarize a
            # subset of datasets.
            raise typer.Exit(1) from exc

    batch_summary = {
        "script_version": SCRIPT_VERSION,
        "seed": seed,
        "irt_bins": int(irt_bins),
        "datasets": results,
    }
    root.mkdir(parents=True, exist_ok=True)
    batch_path = root / "batch_summary.json"
    with batch_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=2, ensure_ascii=False, default=_json_default)
    print(f"\n[green]All datasets completed. Batch summary: {batch_path}[/green]")


if __name__ == "__main__":
    app()
