"""Exact student-cluster significance tests for KT models on one fixed test set.

Experimental design assumed by this script
------------------------------------------
Training uses five train/validation CV runs.  The validation-best checkpoint
from every run is evaluated on the *same* held-out test set.  Prediction files
remain separate by CV run (historically named ``fold``):

    predictions_fold0.csv ... predictions_fold4.csv

The paper-aligned primary statistic is

    mean_f [ AUC_A(f) - AUC_B(f) ]

rather than an AUC computed after pooling the five repeated predictions.

Dependence handling
-------------------
The held-out student is the resampling cluster.  Because the same test students
appear in all five CV runs, one student's permutation swap decision or bootstrap
multiplicity is shared across *all* runs.

Inference
---------
1. Two-sided paired student-cluster permutation test with Monte Carlo +1
   correction.
2. Paired student-cluster percentile bootstrap for a 95% CI.
3. Holm-Bonferroni correction across datasets in batch mode.

Both resampling procedures are exact for the statistic above.  The permutation
implementation uses an algebraically equivalent student influence decomposition;
the bootstrap implementation uses exact student-pair Mann-Whitney contributions,
which avoids recomputing and resorting hundreds of thousands of predictions in
every replicate.
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich import print
from sklearn.metrics import roc_auc_score

app = typer.Typer(add_completion=False)

SCRIPT_VERSION = "2026-07-31-fixed-test-cluster-v4"
FOLD_RE = re.compile(r"predictions_fold(\d+)\.csv$")
EXPECTED_FOLDS = (0, 1, 2, 3, 4)


def _extract_fold_id(path: Path) -> int:
    match = FOLD_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot infer fold id from file name: {path.name}")
    return int(match.group(1))


def _prediction_files(prediction_dir: Path) -> dict[int, Path]:
    files: dict[int, Path] = {}
    for path in sorted(prediction_dir.glob("predictions_fold*.csv")):
        fold_id = _extract_fold_id(path)
        if fold_id in files:
            raise ValueError(
                f"Duplicate prediction file for fold {fold_id} in {prediction_dir}"
            )
        files[fold_id] = path
    if not files:
        raise FileNotFoundError(
            f"No predictions_fold*.csv files found in {prediction_dir}"
        )
    return files


def _require_expected_folds(files: dict[int, Path], prediction_dir: Path) -> None:
    expected = set(EXPECTED_FOLDS)
    actual = set(files)
    if actual != expected:
        raise ValueError(
            "This paper protocol requires exactly five CV-run prediction files "
            f"{list(EXPECTED_FOLDS)}. Found {sorted(actual)} in {prediction_dir}."
        )


def _load_prediction_file(
    pred_file: Path,
    expected_fold: int,
) -> dict[tuple[int, int], tuple[int, float]]:
    """Load one run into ``(student_id, interaction_id) -> (label, score)``."""
    records: dict[tuple[int, int], tuple[int, float]] = {}

    with pred_file.open(newline="") as f:
        reader = csv.DictReader(f)
        required = {"student_id", "interaction_id", "y_true", "y_pred"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(
                f"{pred_file} is missing required columns: {sorted(missing)}"
            )

        for row_no, row in enumerate(reader, start=2):
            sid = int(float(row["student_id"]))
            iid = int(float(row["interaction_id"]))
            y = int(float(row["y_true"]))
            pred = float(row["y_pred"])

            if "fold" in row and row["fold"] not in (None, ""):
                row_fold = int(float(row["fold"]))
                if row_fold != expected_fold:
                    raise ValueError(
                        f"Fold mismatch in {pred_file}:{row_no}: row fold={row_fold}, "
                        f"file fold={expected_fold}"
                    )

            if y not in (0, 1):
                raise ValueError(
                    f"Non-binary y_true={y} in {pred_file}:{row_no}"
                )
            if not np.isfinite(pred):
                raise ValueError(
                    f"Non-finite y_pred={pred} in {pred_file}:{row_no}"
                )

            key = (sid, iid)
            if key in records:
                raise ValueError(
                    f"Duplicate key {key} in {pred_file}:{row_no}"
                )
            records[key] = (y, pred)

    if not records:
        raise ValueError(f"Prediction file is empty: {pred_file}")
    return records


def load_aligned_folds(
    pred_dir_a: Path,
    pred_dir_b: Path,
) -> dict[int, dict[str, np.ndarray]]:
    """Strictly align Model A and Model B within each CV run."""
    files_a = _prediction_files(pred_dir_a)
    files_b = _prediction_files(pred_dir_b)
    _require_expected_folds(files_a, pred_dir_a)
    _require_expected_folds(files_b, pred_dir_b)

    if set(files_a) != set(files_b):
        raise ValueError(
            "CV-run sets differ between models: "
            f"A={sorted(files_a)}, B={sorted(files_b)}"
        )

    aligned: dict[int, dict[str, np.ndarray]] = {}

    for fold_id in sorted(files_a):
        print(f"  Loading and aligning CV run {fold_id}...")
        rec_a = _load_prediction_file(files_a[fold_id], fold_id)
        rec_b = _load_prediction_file(files_b[fold_id], fold_id)

        keys_a = set(rec_a)
        keys_b = set(rec_b)
        if keys_a != keys_b:
            only_a = sorted(keys_a - keys_b)[:5]
            only_b = sorted(keys_b - keys_a)[:5]
            raise ValueError(
                f"Prediction keys differ in CV run {fold_id}: "
                f"A-only={len(keys_a - keys_b)} examples={only_a}; "
                f"B-only={len(keys_b - keys_a)} examples={only_b}"
            )

        keys = sorted(keys_a)
        student_ids = np.fromiter((k[0] for k in keys), dtype=np.int64)
        interaction_ids = np.fromiter((k[1] for k in keys), dtype=np.int64)
        y_true = np.empty(len(keys), dtype=np.int8)
        y_pred_a = np.empty(len(keys), dtype=np.float64)
        y_pred_b = np.empty(len(keys), dtype=np.float64)

        for i, key in enumerate(keys):
            y_a, p_a = rec_a[key]
            y_b, p_b = rec_b[key]
            if y_a != y_b:
                raise ValueError(
                    f"y_true mismatch in CV run {fold_id} for key {key}: "
                    f"A={y_a}, B={y_b}"
                )
            y_true[i] = y_a
            y_pred_a[i] = p_a
            y_pred_b[i] = p_b

        if np.unique(y_true).size < 2:
            raise ValueError(
                f"CV run {fold_id} contains only one class; AUC is undefined"
            )

        aligned[fold_id] = {
            "student_id": student_ids,
            "interaction_id": interaction_ids,
            "y_true": y_true,
            "y_pred_a": y_pred_a,
            "y_pred_b": y_pred_b,
        }

        print(
            f"    {len(y_true)} interactions, "
            f"{len(np.unique(student_ids))} students"
        )

    return aligned


def _safe_auc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if np.unique(y_true).size < 2:
        raise ValueError("AUC is undefined because y_true contains only one class")
    return float(roc_auc_score(y_true, y_pred))


def compute_observed_statistics(
    folds: dict[int, dict[str, np.ndarray]],
) -> dict:
    """Compute the primary statistic: mean of run-wise AUC differences."""
    per_fold = []
    aucs_a: list[float] = []
    aucs_b: list[float] = []
    deltas: list[float] = []

    for fold_id in sorted(folds):
        data = folds[fold_id]
        auc_a = _safe_auc(data["y_true"], data["y_pred_a"])
        auc_b = _safe_auc(data["y_true"], data["y_pred_b"])
        delta = auc_a - auc_b
        aucs_a.append(auc_a)
        aucs_b.append(auc_b)
        deltas.append(delta)
        per_fold.append(
            {
                "fold": int(fold_id),
                "auc_a": auc_a,
                "auc_b": auc_b,
                "delta_auc": delta,
                "n_interactions": int(len(data["y_true"])),
                "n_students": int(len(np.unique(data["student_id"]))),
            }
        )

    return {
        "mean_auc_a": float(np.mean(aucs_a)),
        "std_auc_a": float(np.std(aucs_a, ddof=0)),
        "mean_auc_b": float(np.mean(aucs_b)),
        "std_auc_b": float(np.std(aucs_b, ddof=0)),
        "mean_delta_auc": float(np.mean(deltas)),
        "std_delta_auc": float(np.std(deltas, ddof=0)),
        "folds": per_fold,
    }


def _prepare_shared_test_clusters(
    folds: dict[int, dict[str, np.ndarray]],
) -> tuple[
    dict[int, dict[str, np.ndarray]],
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
]:
    """Validate one fixed held-out test layout shared by all five CV runs."""
    if not folds:
        raise ValueError("No CV-run predictions were provided")
    if tuple(sorted(folds)) != EXPECTED_FOLDS:
        raise ValueError(
            "Expected CV runs 0-4 for the five-run protocol; "
            f"received {sorted(folds)}"
        )

    fold_ids = sorted(folds)
    reference = folds[fold_ids[0]]
    ref_student = reference["student_id"]
    ref_interaction = reference["interaction_id"]
    ref_y = reference["y_true"]

    for fold_id in fold_ids[1:]:
        data = folds[fold_id]
        same_layout = (
            np.array_equal(data["student_id"], ref_student)
            and np.array_equal(data["interaction_id"], ref_interaction)
            and np.array_equal(data["y_true"], ref_y)
        )
        if not same_layout:
            raise ValueError(
                "CV runs do not contain the same fixed held-out test set. "
                f"Run {fold_ids[0]} and run {fold_id} differ in student_id, "
                "interaction_id, or y_true."
            )

    student_values, student_codes = np.unique(ref_student, return_inverse=True)
    cluster_indices = [
        np.flatnonzero(student_codes == i) for i in range(len(student_values))
    ]

    prepared = {
        fold_id: {
            **folds[fold_id],
            "student_values": student_values,
            "student_codes": student_codes,
        }
        for fold_id in fold_ids
    }
    return prepared, student_values, student_codes, cluster_indices


# ---------------------------------------------------------------------------
# Exact fast permutation machinery
# ---------------------------------------------------------------------------


def _positive_query_wins(
    positive_scores: np.ndarray,
    negative_reference: np.ndarray,
) -> np.ndarray:
    """Pair wins for each positive score against all reference negatives."""
    reference = np.sort(np.asarray(negative_reference, dtype=np.float64))
    query = np.asarray(positive_scores, dtype=np.float64)
    left = np.searchsorted(reference, query, side="left")
    right = np.searchsorted(reference, query, side="right")
    return left.astype(np.float64) + 0.5 * (right - left)


def _all_positive_wins_against_negative_query(
    negative_scores: np.ndarray,
    positive_reference: np.ndarray,
) -> np.ndarray:
    """Pair wins contributed by all positives against each queried negative."""
    reference = np.sort(np.asarray(positive_reference, dtype=np.float64))
    query = np.asarray(negative_scores, dtype=np.float64)
    left = np.searchsorted(reference, query, side="left")
    right = np.searchsorted(reference, query, side="right")
    return (len(reference) - right).astype(np.float64) + 0.5 * (right - left)


def _fold_permutation_student_weights(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    student_codes: np.ndarray,
    n_students: int,
) -> np.ndarray:
    """Return exact linear coefficients for student-wise paired swaps.

    For a swap-sign vector r_s in {+1,-1}, the run-wise AUC difference after
    swapping all observations of student s when r_s=-1 is exactly

        delta(r) = sum_s r_s * weight_s.

    This follows from the Mann-Whitney representation of AUC and the fact that
    both models are swapped together within each student cluster.
    """
    y = np.asarray(y_true)
    a = np.asarray(y_pred_a, dtype=np.float64)
    b = np.asarray(y_pred_b, dtype=np.float64)
    codes = np.asarray(student_codes, dtype=np.int64)

    pos = y == 1
    neg = y == 0
    n_pos = int(np.sum(pos))
    n_neg = int(np.sum(neg))
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC requires both positive and negative labels")

    pos_codes = codes[pos]
    neg_codes = codes[neg]
    ap, bp = a[pos], b[pos]
    an, bn = a[neg], b[neg]

    # Student as the positive member of a positive-negative pair.
    row_aa = np.bincount(
        pos_codes,
        weights=_positive_query_wins(ap, an),
        minlength=n_students,
    )
    row_bb = np.bincount(
        pos_codes,
        weights=_positive_query_wins(bp, bn),
        minlength=n_students,
    )
    row_ab = np.bincount(
        pos_codes,
        weights=_positive_query_wins(ap, bn),
        minlength=n_students,
    )
    row_ba = np.bincount(
        pos_codes,
        weights=_positive_query_wins(bp, an),
        minlength=n_students,
    )
    positive_role = 0.5 * (row_aa - row_bb + row_ab - row_ba)

    # Student as the negative member of a positive-negative pair.
    col_aa = np.bincount(
        neg_codes,
        weights=_all_positive_wins_against_negative_query(an, ap),
        minlength=n_students,
    )
    col_bb = np.bincount(
        neg_codes,
        weights=_all_positive_wins_against_negative_query(bn, bp),
        minlength=n_students,
    )
    col_ab = np.bincount(
        neg_codes,
        weights=_all_positive_wins_against_negative_query(bn, ap),
        minlength=n_students,
    )
    col_ba = np.bincount(
        neg_codes,
        weights=_all_positive_wins_against_negative_query(an, bp),
        minlength=n_students,
    )
    negative_role = 0.5 * (col_aa - col_bb - col_ab + col_ba)

    weights = (positive_role + negative_role) / float(n_pos * n_neg)

    # Internal exactness check: all +1 signs reproduce the ordinary AUC delta.
    ordinary_delta = _safe_auc(y, a) - _safe_auc(y, b)
    if not np.isclose(np.sum(weights), ordinary_delta, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "Permutation weight decomposition failed its exactness check: "
            f"weights={np.sum(weights):.12g}, direct={ordinary_delta:.12g}"
        )
    return weights


def _mean_shared_permutation_weights(
    prepared: dict[int, dict[str, np.ndarray]],
    student_codes: np.ndarray,
    n_students: int,
) -> np.ndarray:
    weights = np.zeros(n_students, dtype=np.float64)
    for fold_id in sorted(prepared):
        d = prepared[fold_id]
        weights += _fold_permutation_student_weights(
            d["y_true"],
            d["y_pred_a"],
            d["y_pred_b"],
            student_codes,
            n_students,
        )
    return weights / float(len(prepared))


def _monte_carlo_two_sided_pvalue(
    observed: float,
    simulated: np.ndarray,
) -> float:
    simulated = np.asarray(simulated, dtype=np.float64)
    extreme = int(np.sum(np.abs(simulated) >= abs(float(observed))))
    return float((extreme + 1) / (len(simulated) + 1))


def student_level_paired_cluster_permutation_test(
    folds: dict[int, dict[str, np.ndarray]],
    n_permutations: int = 10000,
    seed: int = 3407,
    batch_size: int = 2048,
) -> dict:
    """Exact paired student-cluster permutation test for the fixed test set."""
    if n_permutations <= 0:
        raise ValueError("n_permutations must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    prepared, student_values, student_codes, _ = _prepare_shared_test_clusters(folds)
    observed = compute_observed_statistics(folds)
    observed_delta = observed["mean_delta_auc"]
    n_students = len(student_values)

    mean_weights = _mean_shared_permutation_weights(
        prepared, student_codes, n_students
    )
    if not np.isclose(
        float(np.sum(mean_weights)), observed_delta, rtol=1e-10, atol=1e-12
    ):
        raise RuntimeError(
            "Shared permutation weights do not reproduce the observed mean "
            f"delta: {np.sum(mean_weights):.12g} vs {observed_delta:.12g}"
        )

    print(f"  Observed mean AUC (Model A): {observed['mean_auc_a']:.6f}")
    print(f"  Observed mean AUC (Model B): {observed['mean_auc_b']:.6f}")
    print(f"  Observed 5-run mean Delta AUC: {observed_delta:+.6f}")
    print(
        f"  Running {n_permutations} shared-student permutations over "
        f"{n_students} held-out students..."
    )

    permutation_deltas = np.empty(n_permutations, dtype=np.float64)
    offset = 0
    while offset < n_permutations:
        size = min(batch_size, n_permutations - offset)
        # One Bernoulli decision per student, reused across all five CV runs.
        swap = rng.random((size, n_students)) < 0.5
        signs = np.where(swap, -1.0, 1.0)
        permutation_deltas[offset : offset + size] = signs @ mean_weights
        offset += size

    p_value = _monte_carlo_two_sided_pvalue(
        observed_delta, permutation_deltas
    )

    return {
        "p_value": p_value,
        "observed_delta_auc": observed_delta,
        "observed_auc_a": observed["mean_auc_a"],
        "observed_auc_b": observed["mean_auc_b"],
        "mean_auc_a": observed["mean_auc_a"],
        "std_auc_a": observed["std_auc_a"],
        "mean_auc_b": observed["mean_auc_b"],
        "std_auc_b": observed["std_auc_b"],
        "fold_statistics": observed["folds"],
        "permutation_deltas": permutation_deltas,
        "n_students": int(n_students),
        "n_cv_runs": int(len(prepared)),
        "n_permutations": int(n_permutations),
        "seed": int(seed),
        "p_value_correction": "(extreme + 1) / (B + 1)",
    }


# ---------------------------------------------------------------------------
# Exact fast cluster bootstrap machinery
# ---------------------------------------------------------------------------


def _student_pair_u_matrix(
    y_true: np.ndarray,
    scores: np.ndarray,
    student_codes: np.ndarray,
    n_students: int,
) -> np.ndarray:
    """Exact student-pair Mann-Whitney numerator matrix for one score vector.

    Entry H[s,t] is the total AUC pair credit (1 for a win, 0.5 for a tie)
    contributed by positive observations from student s against negative
    observations from student t.
    """
    y = np.asarray(y_true)
    scores = np.asarray(scores, dtype=np.float64)
    codes = np.asarray(student_codes, dtype=np.int64)
    pos = y == 1
    neg = y == 0
    pos_scores = scores[pos]
    neg_scores = scores[neg]
    pos_codes = codes[pos]
    neg_codes = codes[neg]

    h = np.zeros((n_students, n_students), dtype=np.float64)

    # Choose the orientation with fewer queried observations.
    if len(pos_scores) <= len(neg_scores):
        for neg_student in range(n_students):
            ref = neg_scores[neg_codes == neg_student]
            if ref.size == 0:
                continue
            wins = _positive_query_wins(pos_scores, ref)
            h[:, neg_student] = np.bincount(
                pos_codes, weights=wins, minlength=n_students
            )
    else:
        for pos_student in range(n_students):
            ref = pos_scores[pos_codes == pos_student]
            if ref.size == 0:
                continue
            wins = _all_positive_wins_against_negative_query(
                neg_scores, ref
            )
            h[pos_student, :] = np.bincount(
                neg_codes, weights=wins, minlength=n_students
            )

    return h


def _mean_student_pair_delta_matrix(
    prepared: dict[int, dict[str, np.ndarray]],
    student_codes: np.ndarray,
    n_students: int,
) -> np.ndarray:
    """Mean across runs of exact Model-A minus Model-B pair-credit matrices."""
    delta = np.zeros((n_students, n_students), dtype=np.float64)
    for fold_id in sorted(prepared):
        print(f"    Precomputing bootstrap pair contributions for CV run {fold_id}...")
        d = prepared[fold_id]
        h_a = _student_pair_u_matrix(
            d["y_true"], d["y_pred_a"], student_codes, n_students
        )
        h_b = _student_pair_u_matrix(
            d["y_true"], d["y_pred_b"], student_codes, n_students
        )
        delta += h_a - h_b
    return delta / float(len(prepared))


def _bootstrap_delta_from_counts(
    counts: np.ndarray,
    mean_pair_delta: np.ndarray,
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
) -> float:
    """Exact mean-run AUC delta for one bootstrap student multiplicity vector."""
    m = np.asarray(counts, dtype=np.float64)
    p = float(m @ positive_counts)
    n = float(m @ negative_counts)
    if p <= 0 or n <= 0:
        return float("nan")
    numerator = float(m @ mean_pair_delta @ m)
    return numerator / (p * n)


def student_level_paired_cluster_bootstrap(
    folds: dict[int, dict[str, np.ndarray]],
    n_bootstrap: int = 10000,
    seed: int = 3407,
    ci_level: float = 0.95,
    batch_size: int = 256,
) -> dict:
    """Exact paired student-cluster bootstrap with shared resamples across runs."""
    if n_bootstrap <= 0:
        raise ValueError("n_bootstrap must be positive")
    if not (0.0 < ci_level < 1.0):
        raise ValueError("ci_level must be between 0 and 1")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    rng = np.random.default_rng(seed)
    prepared, student_values, student_codes, _ = _prepare_shared_test_clusters(folds)
    n_students = len(student_values)
    reference = prepared[sorted(prepared)[0]]
    y = reference["y_true"]

    positive_counts = np.bincount(
        student_codes[y == 1], minlength=n_students
    ).astype(np.float64)
    negative_counts = np.bincount(
        student_codes[y == 0], minlength=n_students
    ).astype(np.float64)

    print(
        "  Precomputing exact student-pair contributions for bootstrap "
        f"({n_students} x {n_students})..."
    )
    mean_pair_delta = _mean_student_pair_delta_matrix(
        prepared, student_codes, n_students
    )

    # Exactness check for the unresampled test set (all multiplicities = 1).
    observed = compute_observed_statistics(folds)
    direct = observed["mean_delta_auc"]
    matrix_direct = _bootstrap_delta_from_counts(
        np.ones(n_students, dtype=np.float64),
        mean_pair_delta,
        positive_counts,
        negative_counts,
    )
    if not np.isclose(matrix_direct, direct, rtol=1e-10, atol=1e-12):
        raise RuntimeError(
            "Bootstrap pair matrix failed its exactness check: "
            f"matrix={matrix_direct:.12g}, direct={direct:.12g}"
        )

    print(
        f"  Running {n_bootstrap} shared-student bootstrap samples over "
        f"{n_students} held-out students..."
    )

    bootstrap_deltas = np.empty(n_bootstrap, dtype=np.float64)
    filled = 0
    attempts = 0
    probabilities = np.full(n_students, 1.0 / n_students, dtype=np.float64)

    while filled < n_bootstrap:
        size = min(batch_size, n_bootstrap - filled)
        counts = rng.multinomial(
            n_students, probabilities, size=size
        ).astype(np.float64, copy=False)
        attempts += size

        p = counts @ positive_counts
        n = counts @ negative_counts
        valid = (p > 0) & (n > 0)
        if not np.any(valid):
            continue

        valid_counts = counts[valid]
        # m^T D m for every bootstrap multiplicity vector m.
        numerators = np.sum((valid_counts @ mean_pair_delta) * valid_counts, axis=1)
        deltas = numerators / (p[valid] * n[valid])

        take = min(len(deltas), n_bootstrap - filled)
        bootstrap_deltas[filled : filled + take] = deltas[:take]
        filled += take

    alpha = 1.0 - ci_level
    ci_lower = float(np.percentile(bootstrap_deltas, 100 * alpha / 2))
    ci_upper = float(np.percentile(bootstrap_deltas, 100 * (1 - alpha / 2)))

    return {
        "delta_auc_mean": float(np.mean(bootstrap_deltas)),
        "delta_auc_std": float(np.std(bootstrap_deltas, ddof=0)),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_level": float(ci_level),
        "bootstrap_deltas": bootstrap_deltas,
        "n_students": int(n_students),
        "n_cv_runs": int(len(prepared)),
        "n_bootstrap": int(n_bootstrap),
        "n_attempts": int(attempts),
        "seed": int(seed),
        "resampling_unit": "held-out student; multiplicities shared across all CV runs",
    }


def run_statistical_test(
    pred_dir_a: Path,
    pred_dir_b: Path,
    model_a_name: str = "a removed model",
    model_b_name: str = "Baseline",
    n_permutations: int = 10000,
    n_bootstrap: int = 10000,
    seed: int = 3407,
) -> dict:
    """Run alignment, observed statistics, permutation test, and bootstrap."""
    print(f"[cyan]Statistical-test version: {SCRIPT_VERSION}[/cyan]")
    print(
        f"\n[yellow]Loading and aligning {model_a_name} vs {model_b_name}...[/yellow]"
    )
    folds = load_aligned_folds(pred_dir_a, pred_dir_b)
    _prepare_shared_test_clusters(folds)  # fail before any expensive resampling

    print("\n[blue]Observed AUCs on the common fixed held-out test set:[/blue]")
    observed = compute_observed_statistics(folds)
    for row in observed["folds"]:
        print(
            f"  CV run {row['fold']}: A={row['auc_a']:.6f}, "
            f"B={row['auc_b']:.6f}, Delta={row['delta_auc']:+.6f}"
        )
    print(
        f"  mean ± std A: {observed['mean_auc_a']:.6f} ± "
        f"{observed['std_auc_a']:.6f}"
    )
    print(
        f"  mean ± std B: {observed['mean_auc_b']:.6f} ± "
        f"{observed['std_auc_b']:.6f}"
    )
    print(f"  primary mean Delta AUC: {observed['mean_delta_auc']:+.6f}")

    print("\n[blue]Running paired student-cluster permutation test...[/blue]")
    perm_result = student_level_paired_cluster_permutation_test(
        folds, n_permutations=n_permutations, seed=seed
    )

    print("\n[blue]Running paired student-cluster bootstrap...[/blue]")
    bootstrap_result = student_level_paired_cluster_bootstrap(
        folds, n_bootstrap=n_bootstrap, seed=seed
    )

    return {
        "method": {
            "script_version": SCRIPT_VERSION,
            "primary_statistic": "mean across 5 CV runs of (AUC_A - AUC_B)",
            "test_set": "same fixed held-out interactions in every CV run",
            "cluster": "student",
            "permutation": "one student swap decision shared across all 5 CV runs",
            "bootstrap": "one student multiplicity sample shared across both models and all 5 CV runs",
            "permutation_p_value": "two-sided Monte Carlo with +1 correction",
            "bootstrap_ci": "percentile 95% CI",
            "seed": int(seed),
        },
        "model_a": model_a_name,
        "model_b": model_b_name,
        "observed": observed,
        "permutation_test": perm_result,
        "bootstrap": bootstrap_result,
    }


def holm_bonferroni_correction(p_values: list[float]) -> list[float]:
    """Return Holm-adjusted p-values in original input order."""
    n = len(p_values)
    if n == 0:
        return []

    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    adjusted = [0.0] * n
    cumulative_max = 0.0

    for rank, (orig_idx, p) in enumerate(indexed):
        value = min(1.0, float(p) * (n - rank))
        value = max(value, cumulative_max)
        adjusted[orig_idx] = value
        cumulative_max = value

    return adjusted


def get_significance_marker(p_value: float) -> str:
    if p_value < 0.001:
        return "***"
    if p_value < 0.01:
        return "**"
    if p_value < 0.05:
        return "*"
    return ""


def _format_p(p: float) -> str:
    if p < 0.001:
        return "<0.001"
    return f"{p:.4f}"


def format_result_table(
    results: list[dict],
    apply_holm: bool = True,
) -> str:
    """Format a compact paper-oriented Markdown summary table."""
    p_values = [r["permutation_test"]["p_value"] for r in results]
    adjusted = holm_bonferroni_correction(p_values) if apply_holm else p_values

    lines = [
        "| Dataset | Baseline | a removed model AUC (mean±std) | Baseline AUC (mean±std) | Δ AUC | 95% CI | Raw p | Adj. p | Sig. |",
        "|---|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]

    for r, adj_p in zip(results, adjusted):
        perm = r["permutation_test"]
        boot = r["bootstrap"]
        raw_p = float(perm["p_value"])
        r["adjusted_p_value"] = float(adj_p)
        lines.append(
            f"| {r.get('dataset', '-')} | {r.get('baseline', r.get('model_b', '-'))} | "
            f"{perm['mean_auc_a']:.4f}±{perm['std_auc_a']:.4f} | "
            f"{perm['mean_auc_b']:.4f}±{perm['std_auc_b']:.4f} | "
            f"{perm['observed_delta_auc']:+.4f} | "
            f"[{boot['ci_lower']:+.4f}, {boot['ci_upper']:+.4f}] | "
            f"{_format_p(raw_p)} | {_format_p(float(adj_p))} | "
            f"{get_significance_marker(float(adj_p))} |"
        )

    return "\n".join(lines)


def _compact_result(result: dict) -> dict:
    """Drop large resampling distributions before JSON serialization."""
    compact = dict(result)
    compact["permutation_test"] = {
        k: v
        for k, v in result["permutation_test"].items()
        if k != "permutation_deltas"
    }
    compact["bootstrap"] = {
        k: v
        for k, v in result["bootstrap"].items()
        if k != "bootstrap_deltas"
    }
    return compact


def _find_latest_complete_prediction_dir(
    train_root: Path,
    dataset: str,
    model: str,
) -> Path:
    """Pick the newest CV directory that contains all five exported runs."""
    cv_dirs = sorted(
        (train_root / dataset).glob(f"cv-{dataset}-{model}-*"), reverse=True
    )
    diagnostics: list[str] = []
    for cv_dir in cv_dirs:
        pred_dir = cv_dir / "predictions"
        if not pred_dir.exists():
            diagnostics.append(f"{cv_dir.name}: no predictions directory")
            continue
        files = _prediction_files(pred_dir)
        if set(files) == set(EXPECTED_FOLDS):
            return pred_dir
        diagnostics.append(f"{cv_dir.name}: folds={sorted(files)}")

    detail = "; ".join(diagnostics[:5]) if diagnostics else "no CV directories"
    raise FileNotFoundError(
        f"Could not find a complete 5-run prediction directory for "
        f"{dataset}/{model}. Checked: {detail}"
    )


@app.command()
def main(
    dataset: str = typer.Option(..., "--dataset", "-d", help="Dataset name"),
    model_a: str = typer.Option("removed_model", "--model-a", help="Model A name"),
    model_b: str = typer.Option(..., "--model-b", help="Model B name"),
    pred_dir_a: Optional[str] = typer.Option(
        None, "--pred-a", help="Prediction directory for model A"
    ),
    pred_dir_b: Optional[str] = typer.Option(
        None, "--pred-b", help="Prediction directory for model B"
    ),
    train_model_root: str = typer.Option(
        "E:/project/knowledgeTracing/train_model",
        "--train-root",
        help="Root directory for trained models",
    ),
    n_permutations: int = typer.Option(
        10000, "--n-perm", help="Number of permutations"
    ),
    n_bootstrap: int = typer.Option(
        10000, "--n-boot", help="Number of bootstrap samples"
    ),
    seed: int = typer.Option(3407, "--seed", help="Random seed"),
    output_dir: Optional[str] = typer.Option(
        None, "--output-dir", help="Output directory for results"
    ),
):
    """Run the fixed-test statistical comparison for one dataset."""
    train_root = Path(train_model_root)
    dir_a = (
        Path(pred_dir_a)
        if pred_dir_a
        else _find_latest_complete_prediction_dir(train_root, dataset, model_a)
    )
    dir_b = (
        Path(pred_dir_b)
        if pred_dir_b
        else _find_latest_complete_prediction_dir(train_root, dataset, model_b)
    )

    print(f"[blue]Model A ({model_a}) predictions: {dir_a}[/blue]")
    print(f"[blue]Model B ({model_b}) predictions: {dir_b}[/blue]")

    result = run_statistical_test(
        dir_a,
        dir_b,
        model_a_name=model_a,
        model_b_name=model_b,
        n_permutations=n_permutations,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    result["dataset"] = dataset
    result["baseline"] = model_b

    perm = result["permutation_test"]
    boot = result["bootstrap"]
    print("\n" + "=" * 74)
    print("  Statistical Test Results")
    print("=" * 74)
    print(
        f"Model A AUC: {perm['mean_auc_a']:.6f} ± {perm['std_auc_a']:.6f}"
    )
    print(
        f"Model B AUC: {perm['mean_auc_b']:.6f} ± {perm['std_auc_b']:.6f}"
    )
    print(f"Mean Delta AUC: {perm['observed_delta_auc']:+.6f}")
    print(f"Permutation p:  {perm['p_value']:.6g}")
    print(
        f"95% bootstrap CI: [{boot['ci_lower']:+.6f}, "
        f"{boot['ci_upper']:+.6f}]"
    )

    out_path = Path(output_dir) if output_dir else Path("output/statistical_tests")
    out_path.mkdir(parents=True, exist_ok=True)
    result_file = out_path / f"statistical_test_{dataset}_{model_a}_vs_{model_b}.json"
    with result_file.open("w") as f:
        json.dump(_compact_result(result), f, indent=2)
    print(f"\n[green]Results saved to: {result_file}[/green]")


@app.command("batch")
def batch_test(
    datasets: str = typer.Option(
        "algebra2005,assist2009,assist2017,bridge2algebra2006,nips_task34",
        "--datasets",
        help="Comma-separated dataset names",
    ),
    model_a: str = typer.Option("removed_model", "--model-a", help="Model A name"),
    train_model_root: str = typer.Option(
        "E:/project/knowledgeTracing/train_model",
        "--train-root",
        help="Root directory for trained models",
    ),
    n_permutations: int = typer.Option(
        10000, "--n-perm", help="Number of permutations"
    ),
    n_bootstrap: int = typer.Option(
        10000, "--n-boot", help="Number of bootstrap samples"
    ),
    seed: int = typer.Option(3407, "--seed", help="Random seed"),
    apply_holm: bool = typer.Option(
        True, "--holm/--no-holm", help="Apply Holm-Bonferroni correction"
    ),
):
    """Run all five pre-specified comparisons, then apply Holm correction."""
    dataset_list = [d.strip() for d in datasets.split(",") if d.strip()]
    dataset_baseline_map = {
        "algebra2005": "ukt",
        "assist2009": "ukt",
        "assist2017": "ukt",
        "bridge2algebra2006": "akt",
        "nips_task34": "qikt",
    }

    train_root = Path(train_model_root)
    results: list[dict] = []
    failures: list[str] = []

    for dataset in dataset_list:
        baseline = dataset_baseline_map.get(dataset)
        if baseline is None:
            failures.append(f"{dataset}: no baseline mapping")
            continue

        print(f"\n{'=' * 74}")
        print(f"  Testing {dataset}: {model_a} vs {baseline}")
        print(f"{'=' * 74}")

        try:
            dir_a = _find_latest_complete_prediction_dir(
                train_root, dataset, model_a
            )
            dir_b = _find_latest_complete_prediction_dir(
                train_root, dataset, baseline
            )
            print(f"  Model A predictions: {dir_a}")
            print(f"  Model B predictions: {dir_b}")

            result = run_statistical_test(
                dir_a,
                dir_b,
                model_a_name=model_a,
                model_b_name=baseline,
                n_permutations=n_permutations,
                n_bootstrap=n_bootstrap,
                seed=seed,
            )
            result["dataset"] = dataset
            result["baseline"] = baseline
            results.append(result)
        except Exception as exc:
            failures.append(f"{dataset}: {exc}")
            print(f"[red]Error testing {dataset}: {exc}[/red]")

    # Holm's family is pre-specified as the requested dataset set.  Do not
    # silently correct a smaller subset if one comparison failed.
    if failures:
        print("\n[red]Batch aborted before Holm correction because some comparisons failed:[/red]")
        for failure in failures:
            print(f"  - {failure}")
        raise typer.Exit(1)

    if not results:
        print("[red]No dataset comparison completed successfully.[/red]")
        raise typer.Exit(1)

    table = format_result_table(results, apply_holm=apply_holm)
    print("\n" + "=" * 120)
    print("  Summary of Statistical Tests")
    print("=" * 120)
    print(table)

    output_dir = Path("output/statistical_tests")
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_file = output_dir / "statistical_test_summary.json"
    md_file = output_dir / "statistical_test_summary.md"

    compact_results = [_compact_result(r) for r in results]
    with summary_file.open("w") as f:
        json.dump(compact_results, f, indent=2)
    with md_file.open("w") as f:
        f.write("# Statistical Test Results\n\n")
        f.write(f"Script version: `{SCRIPT_VERSION}`\n\n")
        f.write("Primary statistic: mean over five CV-trained checkpoints of ")
        f.write("`AUC(a removed model) - AUC(baseline)` on the same held-out test set.\n\n")
        f.write(f"Number of permutations: {n_permutations}\n\n")
        f.write(f"Number of bootstrap samples: {n_bootstrap}\n\n")
        f.write(f"Seed: {seed}\n\n")
        f.write(table)
        f.write("\n\nSignificance markers are based on Holm-adjusted p-values.\n")

    print(f"\n[green]Summary saved to:[/green]\n  - {summary_file}\n  - {md_file}")


if __name__ == "__main__":
    app()
