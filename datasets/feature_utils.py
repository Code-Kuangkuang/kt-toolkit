import os
from collections import defaultdict

import numpy as np
import pandas as pd


def parse_first_int(value):
    token = str(value).split("_", 1)[0]
    if token == "":
        return -1
    return int(float(token))


def parse_int_list(value):
    if value is None or value == "":
        return []
    return [parse_first_int(x) for x in str(value).split(",") if x != ""]


def compute_question_frequency_counts(
    dpath,
    train_valid_file,
    folds,
    num_q,
):
    """Count question occurrences using only the requested training folds."""
    path = os.path.join(dpath, train_valid_file)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Question-frequency source not found: {path}"
        )
    num_q = int(num_q)
    if num_q <= 0:
        raise ValueError(f"Question counts require num_q > 0, got {num_q}.")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    required = {"fold", "questions"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Frequency source missing columns: {sorted(missing)}"
        )

    fold_set = {int(fold) for fold in folds}
    if not fold_set:
        raise ValueError("Question-frequency folds must not be empty.")
    selected = df[df["fold"].astype(int).isin(fold_set)]
    if selected.empty:
        raise ValueError(
            f"No frequency rows found for folds {sorted(fold_set)}."
        )

    counts = np.zeros(num_q, dtype=np.int64)
    for raw_questions in selected["questions"]:
        for question in parse_int_list(raw_questions):
            if question == -1:
                continue
            if question < 0 or question >= num_q:
                raise ValueError(
                    "Question ID out of range in frequency source; "
                    f"expected -1 or [0, {num_q - 1}], got {question}."
                )
            counts[question] += 1

    nonzero = counts[counts > 0]
    summary = {
        "source_file": os.path.normpath(path),
        "folds": sorted(fold_set),
        "total_interactions": int(counts.sum()),
        "nonzero_questions": int(nonzero.size),
        "max_count": int(nonzero.max()) if nonzero.size else 0,
        "mean_nonzero_count": float(nonzero.mean()) if nonzero.size else 0.0,
    }
    return counts, summary


def log2_gap(value):
    import math

    return round(math.log(value + 1, 2))


def compute_dkt_forget_gaps(row, input_type):
    skills = (
        str(row["concepts"]).split(",")
        if "concepts" in input_type
        else str(row["questions"]).split(",")
    )
    timestamps = parse_int_list(row["timestamps"]) if "timestamps" in row.index else []
    repeated_gap, sequence_gap, past_counts = [], [], []
    last_skill_time = {}
    counts = {}
    prev_time = None

    for raw_skill, timestamp in zip(skills, timestamps):
        skill = parse_first_int(raw_skill)
        if skill not in last_skill_time or skill == -1:
            cur_repeated_gap = 0
        else:
            cur_repeated_gap = log2_gap((timestamp - last_skill_time[skill]) / 1000 / 60) + 1
        last_skill_time[skill] = timestamp
        repeated_gap.append(cur_repeated_gap)

        if prev_time is None or timestamp == -1:
            cur_sequence_gap = 0
        else:
            cur_sequence_gap = log2_gap((timestamp - prev_time) / 1000 / 60) + 1
        prev_time = timestamp
        sequence_gap.append(cur_sequence_gap)

        counts.setdefault(skill, 0)
        past_counts.append(log2_gap(counts[skill]))
        counts[skill] += 1

    return repeated_gap, sequence_gap, past_counts


def compute_dkt_forget_stats(dpath, filenames, input_type):
    max_rgap, max_sgap, max_pcount = 0, 0, 0
    checked_paths = []
    found_timestamp_file = False
    for filename in filenames:
        if not filename:
            continue
        path = os.path.join(dpath, filename)
        checked_paths.append(path)
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, dtype=str, keep_default_na=False)
        if "timestamps" not in df.columns:
            continue
        found_timestamp_file = True
        for _, row in df.iterrows():
            rgap, sgap, pcount = compute_dkt_forget_gaps(row, input_type)
            if rgap:
                max_rgap = max(max_rgap, max(rgap))
            if sgap:
                max_sgap = max(max_sgap, max(sgap))
            if pcount:
                max_pcount = max(max_pcount, max(pcount))
    if not found_timestamp_file:
        raise ValueError(
            "DKT-forget requires at least one existing sequence file with a "
            f"'timestamps' column for gap statistics. Checked: {checked_paths}"
        )
    return {
        "num_rgap": max_rgap + 1,
        "num_sgap": max_sgap + 1,
        "num_pcount": max_pcount + 1,
    }


def compute_history_correctness(concepts, responses):
    history = []
    right, total = 0, 0
    for response in responses:
        if response == 1:
            right += 1
        total += 1
        history.append(right / total if total else 0.0)
    return history


def compute_dimkt_difficulty_maps(dpath, train_valid_file, diff_level, folds=None):
    path = os.path.join(dpath, train_valid_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"DIMKT difficulty source not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if folds is not None:
        if "fold" not in df.columns:
            raise ValueError(
                f"DIMKT difficulty source missing required 'fold' column: {path}"
            )
        fold_set = {int(fold) for fold in folds}
        if not fold_set:
            raise ValueError("DIMKT difficulty folds must not be empty.")
        df = df[df["fold"].astype(int).isin(fold_set)]
        if df.empty:
            raise ValueError(
                f"No DIMKT difficulty rows found for folds {sorted(fold_set)}."
            )
    skill_totals = defaultdict(lambda: [0, 0])
    question_totals = defaultdict(lambda: [0, 0])

    for _, row in df.iterrows():
        concepts = parse_int_list(row["concepts"]) if "concepts" in row.index else []
        questions = parse_int_list(row["questions"]) if "questions" in row.index else []
        responses = parse_int_list(row["responses"])
        for concept, response in zip(concepts, responses):
            if concept == -1 or response == -1:
                continue
            skill_totals[concept][0] += response
            skill_totals[concept][1] += 1
        for question, response in zip(questions, responses):
            if question == -1 or response == -1:
                continue
            question_totals[question][0] += response
            question_totals[question][1] += 1

    def _to_level(stats):
        result = {}
        for key, (correct, total) in stats.items():
            if total < 30 or correct == 0:
                result[key] = 1
            else:
                result[key] = int((correct / total) * diff_level) + 1
        return result

    return {
        "skills": _to_level(skill_totals),
        "questions": _to_level(question_totals),
    }


def compute_hqaf_feature_maps(dpath, train_valid_file, diff_level=50, num_time_bins=20, folds=None):
    path = os.path.join(dpath, train_valid_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"HQAF feature source not found: {path}")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if folds is not None and "fold" in df.columns:
        fold_set = {int(f) for f in folds}
        df = df[df["fold"].astype(int).isin(fold_set)]

    skill_totals = defaultdict(lambda: [0, 0])
    question_totals = defaultdict(lambda: [0, 0])
    question_times = defaultdict(list)
    all_times = []

    for _, row in df.iterrows():
        concepts = parse_int_list(row["concepts"]) if "concepts" in row.index else []
        questions = parse_int_list(row["questions"]) if "questions" in row.index else []
        responses = parse_int_list(row["responses"])
        use_times = parse_float_list(row["usetimes"]) if "usetimes" in row.index else []

        for concept, response in zip(concepts, responses):
            if concept == -1 or response == -1:
                continue
            skill_totals[concept][0] += response
            skill_totals[concept][1] += 1
        for question, response in zip(questions, responses):
            if question == -1 or response == -1:
                continue
            question_totals[question][0] += response
            question_totals[question][1] += 1
        for question, use_time in zip(questions, use_times):
            if question == -1 or use_time < 0:
                continue
            question_times[question].append(use_time)
            all_times.append(use_time)

    time_bin_edges = _build_log_quantile_edges(all_times, num_time_bins)

    def _to_level(stats):
        result = {}
        for key, (correct, total) in stats.items():
            if total < 30 or correct == 0:
                result[key] = 1
            else:
                result[key] = int((correct / total) * diff_level) + 1
        return result

    question_avg_time = {}
    for question, values in question_times.items():
        if values:
            question_avg_time[question] = _to_time_bin(float(np.mean(values)), time_bin_edges, num_time_bins)

    return {
        "skills": _to_level(skill_totals),
        "questions": _to_level(question_totals),
        "question_avg_time": question_avg_time,
        "time_bin_edges": time_bin_edges,
        "num_time_bins": int(num_time_bins),
        "has_usetimes": "usetimes" in df.columns,
        "has_type": "type" in df.columns,
    }


def parse_float_list(value):
    if value is None or value == "":
        return []
    result = []
    for x in str(value).split(","):
        if x == "":
            continue
        result.append(float(x))
    return result


def _build_log_quantile_edges(values, num_time_bins):
    positive = np.array([float(v) for v in values if float(v) > 0], dtype=float)
    if positive.size == 0:
        return []
    if positive.max() <= num_time_bins and np.allclose(positive, np.round(positive)):
        return []
    log_values = np.log1p(positive)
    quantiles = np.linspace(0, 1, int(num_time_bins) + 1)[1:-1]
    if quantiles.size == 0:
        return []
    edges = np.quantile(log_values, quantiles)
    return [float(edge) for edge in np.unique(edges)]


def _to_time_bin(value, edges, num_time_bins):
    if value < 0:
        return -1
    if not edges and value <= num_time_bins:
        return int(max(0, min(num_time_bins, round(value))))
    if value <= 0:
        return 0
    idx = int(np.searchsorted(np.asarray(edges, dtype=float), np.log1p(float(value)), side="right"))
    return max(0, min(int(num_time_bins) - 1, idx))
