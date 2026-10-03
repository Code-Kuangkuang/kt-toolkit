import os
import random
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


def compute_dkt_forget_stats(dpath, filenames, input_type, folds=None):
    """Size the gap embedding tables DKT-Forget indexes into.

    A gap is "how long since this learner last met this concept", log2-bucketed,
    so the returned counts are table heights rather than statistics about the
    data. They are what made this function read the test file: not to see any
    label -- it reads `timestamps` only -- but to know how many rows to allocate.

    `folds` restricts the source to the current fold's training rows, which is
    what AGENTS.md requires of any derived feature. That leaves a table that can
    be too short, because valid or test may hold a longer gap than training ever
    saw, so one extra row is reserved as an out-of-vocabulary bucket meaning
    "longer than anything in training". `clamp_dkt_forget_gaps` maps oversized
    values onto it.

    Passing `folds=None` restores the older behaviour of taking a maximum over
    every split, which is what pyKT does.
    """
    max_rgap, max_sgap, max_pcount = 0, 0, 0
    checked_paths = []
    found_timestamp_file = False
    fold_set = {int(fold) for fold in folds} if folds is not None else None
    if fold_set is not None and not fold_set:
        raise ValueError("DKT-forget gap folds must not be empty.")

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
        if fold_set is not None:
            if "fold" not in df.columns:
                # The test file carries no fold column; under a train-folds
                # scope it contributes nothing and is skipped rather than
                # silently contributing everything.
                continue
            df = df[df["fold"].astype(int).isin(fold_set)]
            if df.empty:
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
        scope = "training folds" if fold_set is not None else "any split"
        raise ValueError(
            "DKT-forget requires at least one existing sequence file with a "
            f"'timestamps' column for gap statistics, within {scope}. "
            f"Checked: {checked_paths}"
        )

    # +1 turns a maximum into a count; the second +1 under a fold scope is the
    # out-of-vocabulary row.
    oov = 1 if fold_set is not None else 0
    return {
        "num_rgap": max_rgap + 1 + oov,
        "num_sgap": max_sgap + 1 + oov,
        "num_pcount": max_pcount + 1 + oov,
    }


def clamp_dkt_forget_gaps(values, cap):
    """Fold anything at or past `cap` onto the last row of the table.

    Under a train-folds scope that last row is the reserved out-of-vocabulary
    bucket, so an unseen long gap becomes "longer than training ever saw"
    instead of an index error. With no cap the values pass through.
    """
    if not cap:
        return values
    limit = int(cap) - 1
    return [min(int(v), limit) for v in values]


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


def compute_item_difficulty_logodds(dpath, train_valid_file, num_q, folds=None,
                                    alpha=10.0, grouping=None, group_seed=3407,
                                    cold_start=False, concept_alias=None):
    """Per-question empirical difficulty as a standardised log-odds vector.

    Answers a question the learned Rasch scalar raises: SimpleKT's `qid_scalar`
    variant learns one number per question, and on assist2009 that number is
    0.91-correlated (within concept, Spearman) with the question's training-fold
    correct rate.  If the gradient is only recovering a count, the count can be
    supplied directly and the parameter frozen -- which is what this table is
    for.  See research/results_item_parameterisation.md.

    Shape is `[num_q + 1]`, matching `Embedding(num_pid + 1, 1)`; the last row is
    the padding slot and stays at 0.

    Three choices worth stating, because each one moves the numbers:

    * **Shrinkage.** A raw correct rate is undefined at `total == 0` and wild at
      `total == 1`; assist2009 has 2,023 questions seen exactly once in a fold's
      training rows.  The rate is shrunk toward the global rate with strength
      `alpha`, i.e. `(correct + alpha*p0) / (total + alpha)`, so a question's
      estimate moves away from the prior only as evidence accumulates.

    * **Centring on `logit(p0)`, not on the mean.** This makes an unseen question
      come out at exactly 0, which is what the *learned* table does for the same
      questions -- `SimpleKT.reset()` starts every row at 0 and an unseen row
      gets no gradient.  Centring on the mean instead would hand unseen items a
      nonzero difficulty, which is a different model, not a frozen version of
      this one.

    * **Scale is normalised away.** The vector is divided by its own standard
      deviation over seen questions, because the term it feeds is
      `difficult_param * q_embed_diff[c]` and `q_embed_diff` is learnable: the
      model can set the overall magnitude itself.  What is frozen is the
      *relative ordering and spacing* of questions, which is the hypothesis
      under test.

    Counts every response that is not `-1`, matching
    `compute_dimkt_difficulty_maps` above, so the two difficulty features in this
    repository are derived from the same rows.

    With `grouping="concept"` the shrinkage target stops being the global rate
    and becomes the item's own concept group, minus the item itself. That is the
    variant under test: on a dataset where most items are seen too few times to
    measure -- algebra2005 averages 3.3 responses per item and only 4.1% reach
    ten -- the global rate is the same number for everyone and carries no item
    information, while the group's rate at least says which neighbourhood the
    item sits in. `grouping="random"` keeps the group sizes and reshuffles the
    members, and is the control that says whether the concept partition itself
    is doing the work.
    """
    tables = compute_difficulty_logodds_tables(
        dpath, train_valid_file, folds=folds, num_q=num_q, alpha=alpha,
        grouping=grouping, group_seed=group_seed, cold_start=cold_start,
        concept_alias=concept_alias,
    )
    return tables["items"]


def compute_item_difficulty_ingredients(dpath, train_valid_file, num_q,
                                        folds=None, grouping=None,
                                        group_seed=3407, cold_start=False,
                                        concept_alias=None):
    """The parts `compute_item_difficulty_logodds` folds together, kept separate.

    That function applies the shrinkage itself, at a fixed `alpha`, and returns
    one number per item. An arm that wants to *learn* how hard to shrink has to
    do the fold inside the model, where a gradient can reach it, so it needs the
    three inputs rather than the result:

        rate    [num_q+1]  the item's own training-fold correct rate, 0 unseen
        target  [num_q+1]  what it shrinks towards -- the global rate under
                           `grouping=None`, the item's leave-one-out group rate
                           otherwise
        count   [num_q+1]  how many training-fold responses the item has, which
                           is what the shrinkage weight is a function of

    plus the scalar `base_rate` the log-odds are centred on. `alpha` is absent
    on purpose: it is exactly the quantity the caller is no longer fixing.

    Same rows, same folds and same grouping as the frozen table, so the two arms
    differ in one thing only.
    """
    tables = compute_difficulty_logodds_tables(
        dpath, train_valid_file, folds=folds, num_q=num_q,
        grouping=grouping, group_seed=group_seed, with_ingredients=True,
        cold_start=cold_start, concept_alias=concept_alias,
    )
    return {
        "rate": tables["item_rate"],
        "target": tables["item_target"],
        "count": tables["item_count"],
        "base_rate": tables["base_rate"],
        "cold_start": bool(cold_start),
    }


def _question_groups_from_qmatrix(dpath, num_q):
    """First concept of every question, read from `qmatrix.npz` rather than rows.

    The counting pass can only resolve a question's concept if the question
    appears in a training-fold row, so a question that does not becomes a
    singleton and shrinks towards the global rate. On assist2009 that is 625 of
    17,738 items -- and they are exactly the items grouping is supposed to
    rescue, since an item with no observations has nothing else to go on. The
    `_grouped` and `_grouprand` arms are therefore identical on the subset where
    the hypothesis has the most to say.

    Reading the membership from the Q-matrix instead fixes that, and it leaks
    nothing: `cseqs` already carries the concept ids of test questions into
    every batch, so question-to-concept is information the model is handed at
    inference time regardless. What stays fitted to the training folds is the
    group's *rate*; only the membership comes from metadata.

    Returns `[num_q + 1]` of concept ids, `-1` where a question has no concept.
    """
    path = os.path.join(dpath, "qmatrix.npz")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Cold-start grouping reads question membership from {path}, which "
            f"is missing. Build it with `python scripts/build_qmatrix.py --dataset-name <dataset>`."
        )
    qmatrix = np.load(path)["matrix"][:num_q] > 0
    groups = np.full(num_q + 1, -1, dtype=np.int64)
    rows, cols = np.nonzero(qmatrix)
    if not len(rows):
        return groups
    # np.nonzero yields rows ascending, so the first hit per row is its first
    # concept -- the same choice the counting pass makes with `cids[0]`.
    first = np.searchsorted(rows, np.arange(num_q))
    has_concept = first < len(rows)
    valid = has_concept & (rows[np.minimum(first, len(rows) - 1)] == np.arange(num_q))
    groups[np.arange(num_q)[valid]] = cols[first[valid]]
    return groups


def _standardised_logodds(correct, total, p0, alpha, target=None,
                          zero_unseen=True):
    """The shared core of the difficulty tables. See the caller for the choices.

    `target` is what each row is shrunk *towards*. Scalar `p0` by default, which
    is the global rate; pass a `[len(correct)]` array to shrink each item
    towards its own group's rate instead. Centring stays on `p0` either way, so
    an unseen row still comes out at exactly 0 and still matches what the
    learned table does for the same row.
    """
    seen = total > 0
    if target is None:
        target = p0
    target = np.clip(np.asarray(target, dtype=np.float64), 1e-6, 1 - 1e-6)
    smoothed = (correct + alpha * target) / np.maximum(total + alpha, 1e-12)
    smoothed = np.clip(smoothed, 1e-6, 1 - 1e-6)
    logodds = np.log(smoothed / (1.0 - smoothed)) - np.log(p0 / (1.0 - p0))
    scale = float(logodds[seen].std())
    if scale > 0:
        logodds = logodds / scale
    if zero_unseen:
        logodds[~seen] = 0.0
    logodds[-1] = 0.0  # padding slot, never a real item
    return logodds.astype(np.float32)


def _leave_one_out_group_rate(correct, total, groups, p0):
    """Each item's group rate with that item's own responses removed.

    Self-exclusion is not decoration. Shrinking an item towards a mean that
    already contains the item pulls it towards itself, which looks like
    borrowing but is not: a question seen once would be shrunk towards a target
    it just contributed to, and the estimate would inherit its own noise instead
    of the group's signal. Measured the same way in
    `research/prereq_difficulty_reliability.py`, where including the item was
    what separated a real effect from an inflated one.

    Groups with no other member fall back to `p0`, which is what an ungrouped
    item would have got.
    """
    n = len(correct)
    g_correct = np.zeros(n, dtype=np.float64)
    g_total = np.zeros(n, dtype=np.float64)
    np.add.at(g_correct, groups, correct)
    np.add.at(g_total, groups, total)
    others_c = g_correct[groups] - correct
    others_t = g_total[groups] - total
    rate = np.where(others_t > 0, others_c / np.maximum(others_t, 1e-12), p0)
    return rate


def parse_concept_lists(value):
    """`"25,26_31_27,-1"` -> `[[25], [26, 31, 27], []]`.

    `parse_int_list` keeps only the first concept of a multi-concept position,
    which is what DIMKT's binned difficulty has always used.  A table meant to
    stand next to `pool_concept_embeddings` cannot do that: the model averages
    over every concept of the item, so the counted difficulty has to as well, or
    the two disagree on what "this item's concept" means for exactly the 15-30%
    of positions the pooling was introduced for.
    """
    if value is None or value == "":
        return []
    out = []
    for token in str(value).split(","):
        if token == "":
            out.append([])
            continue
        ids = []
        for part in token.split("_"):
            part = part.strip()
            if part == "":
                continue
            cid = int(float(part))
            if cid >= 0:
                ids.append(cid)
        out.append(ids)
    return out


def _dense_groups(q_group, rng=None):
    """Concept ids -> dense group indices in `[0, len(q_group))`.

    An item with no resolved concept becomes its own singleton, so it shrinks
    towards the global rate exactly as it did before grouping existed.

    With `rng`, members are reshuffled between groups while the multiset of
    group *sizes* is preserved. That is the control: it holds group size,
    number of groups and the leave-one-out arithmetic fixed and varies only
    *which items sit together*, so a gain that survives it is a gain from the
    concept partition rather than from averaging over a bag of that size.
    """
    dense = np.arange(len(q_group), dtype=np.int64)
    buckets = defaultdict(list)
    for item, cid in enumerate(q_group):
        if cid >= 0:
            buckets[int(cid)].append(item)
    members = [m for v in buckets.values() for m in v]
    sizes = [len(v) for v in buckets.values()]
    if rng is not None:
        members = list(members)
        rng.shuffle(members)
    cursor = 0
    for gid, size in enumerate(sizes):
        for item in members[cursor:cursor + size]:
            dense[item] = gid
        cursor += size
    return dense


def compute_difficulty_logodds_tables(dpath, train_valid_file, folds=None,
                                      num_q=None, num_c=None, alpha=10.0,
                                      grouping=None, group_seed=3407,
                                      with_ingredients=False, cold_start=False,
                                      concept_alias=None):
    """Question- and concept-level empirical difficulty, from one pass of the file.

    Returns `{"items": [num_q+1], "concepts": [num_c+1], "base_rate": p0}`, with
    the requested tables only.  Both are standardised log-odds; see
    `compute_item_difficulty_logodds` for why they are shrunk, centred on
    `logit(p0)` and scaled to unit spread.

    One pass because `nullkt` needs both and these files reach 128 MB.
    """
    path = os.path.join(dpath, train_valid_file)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Difficulty source not found: {path}")
    if num_q is None and num_c is None:
        raise ValueError("Ask for at least one of num_q or num_c.")

    alpha = float(alpha)
    if alpha < 0:
        raise ValueError(f"alpha must be non-negative, got {alpha}.")

    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if folds is not None:
        if "fold" not in df.columns:
            raise ValueError(
                f"Difficulty source missing required 'fold' column: {path}"
            )
        fold_set = {int(fold) for fold in folds}
        if not fold_set:
            raise ValueError("Difficulty folds must not be empty.")
        df = df[df["fold"].astype(int).isin(fold_set)]
        if df.empty:
            raise ValueError(
                f"No difficulty rows found for folds {sorted(fold_set)}."
            )
    if num_q is not None and "questions" not in df.columns:
        raise ValueError(
            f"Item difficulty needs question ids; {path} has no 'questions' column."
        )
    if num_c is not None and "concepts" not in df.columns:
        raise ValueError(
            f"Concept difficulty needs concepts; {path} has no 'concepts' column."
        )

    q_correct = q_total = c_correct = c_total = None
    if num_q is not None:
        num_q = int(num_q)
        q_correct = np.zeros(num_q + 1, dtype=np.float64)
        q_total = np.zeros(num_q + 1, dtype=np.float64)
    if num_c is not None:
        num_c = int(num_c)
        c_correct = np.zeros(num_c + 1, dtype=np.float64)
        c_total = np.zeros(num_c + 1, dtype=np.float64)

    # Which group each question belongs to, for `grouping="concept"`. Taken from
    # the first concept of the position where the question is first seen: a
    # question's concept set is fixed in these files, so this is deterministic,
    # and it gives one group per item, which is what shrinking towards a group
    # requires. Items whose group never resolves stay in their own singleton
    # group and therefore fall back to the global rate.
    want_groups = grouping is not None and num_q is not None
    q_group = np.full(num_q + 1, -1, dtype=np.int64) if want_groups else None
    if want_groups and "concepts" not in df.columns:
        raise ValueError(
            f"grouping={grouping!r} needs concepts; {path} has no 'concepts' column."
        )

    questions_col = df["questions"] if num_q is not None else [""] * len(df)
    concepts_col = (df["concepts"] if (num_c is not None or want_groups)
                    else [""] * len(df))
    for questions, concepts, responses in zip(questions_col, concepts_col,
                                              df["responses"]):
        response_list = parse_int_list(responses)
        if num_q is not None:
            q_list = parse_int_list(questions)
            for question, response in zip(q_list, response_list):
                if question < 0 or question >= num_q or response == -1:
                    continue
                q_correct[question] += response
                q_total[question] += 1
            if want_groups:
                for question, cids in zip(q_list, parse_concept_lists(concepts)):
                    if 0 <= question < num_q and cids and q_group[question] < 0:
                        q_group[question] = cids[0]
        if num_c is not None:
            for concept_ids, response in zip(parse_concept_lists(concepts),
                                             response_list):
                if response == -1:
                    continue
                for concept in concept_ids:
                    if concept >= num_c:
                        continue
                    c_correct[concept] += response
                    c_total[concept] += 1

    if want_groups and cold_start:
        # Fill in only the questions the pass could not resolve, i.e. the ones
        # no training row touches. Filling gaps rather than replacing the whole
        # assignment keeps `_cold` a one-variable change: the Q-matrix loses
        # concept order, so its "first concept" is the lowest index while the
        # CSV's is the first listed, and on a multi-concept question those
        # disagree. Overwriting seen items would move ~20% of assist2009's
        # questions into different groups and a `_cold` vs `_grouped` gap could
        # no longer be attributed to the cold items.
        missing = q_group < 0
        if missing.any():
            from_qmatrix = _question_groups_from_qmatrix(dpath, int(num_q))
            q_group = np.where(missing, from_qmatrix, q_group)

    if want_groups and concept_alias is not None:
        # Collapse concept ids that name the same skill. Two items testing
        # "Number Line" under two of its three ids otherwise shrink towards two
        # different targets and neither borrows from the other.
        #
        # After the cold-start fill, not before: ids recovered from the Q-matrix
        # have to be collapsed too, or a cold item lands in the uncollapsed half
        # of a split skill and the two changes interact.
        alias = np.asarray(concept_alias, dtype=np.int64)
        inside = (q_group >= 0) & (q_group < len(alias))
        # -1 stays -1, so an item with no concept at all is still a singleton.
        q_group = np.where(inside, alias[np.clip(q_group, 0, len(alias) - 1)], q_group)

    # The base rate comes from question rows when we have them, so that a
    # multi-concept position is counted once rather than once per concept.
    if num_q is not None:
        seen = q_total > 0
        if not seen.any():
            raise ValueError(f"No question responses in {path} for folds {folds}.")
        p0 = float(q_correct[seen].sum() / q_total[seen].sum())
    else:
        seen = c_total > 0
        if not seen.any():
            raise ValueError(f"No concept responses in {path} for folds {folds}.")
        p0 = float(c_correct[seen].sum() / c_total[seen].sum())

    out = {"base_rate": p0}
    if num_q is not None:
        target = None
        if want_groups:
            if grouping == "concept":
                dense = _dense_groups(q_group)
            elif grouping == "random":
                dense = _dense_groups(q_group, random.Random(int(group_seed)))
            else:
                raise ValueError(
                    f"grouping must be None, 'concept' or 'random', got {grouping!r}."
                )
            target = _leave_one_out_group_rate(q_correct, q_total, dense, p0)
            out["group_sizes"] = np.bincount(dense, minlength=len(dense))
            out["group_unresolved"] = int((q_group < 0).sum())
        out["items"] = _standardised_logodds(
            q_correct, q_total, p0, alpha, target=target,
            # Under `cold_start` an unseen item keeps its group's rate instead
            # of being reset to 0. The zeroing exists so `qid_frozen` stays a
            # faithful frozen copy of the learned Rasch table, which gives an
            # item with no gradient exactly 0 -- but that rationale is about the
            # global-target arm. For a grouped arm it deletes the hypothesis:
            # borrowing from the neighbourhood is most of the point, and an item
            # with zero observations is where there is most to borrow.
            zero_unseen=not cold_start,
        )
        if with_ingredients:
            # The unshrunk parts, for the arm that learns the shrinkage weight
            # instead of fixing it at `alpha`. `_standardised_logodds` folds
            # these together irreversibly, so an arm that needs to differentiate
            # through the fold has to be handed them separately.
            seen_q = q_total > 0
            out["item_rate"] = np.divide(
                q_correct, q_total, out=np.zeros_like(q_correct), where=seen_q
            )
            out["item_count"] = q_total
            out["item_target"] = (
                np.full_like(q_total, p0) if target is None
                else np.asarray(target, dtype=np.float64)
            )
    if num_c is not None:
        out["concepts"] = _standardised_logodds(c_correct, c_total, p0, alpha)
    return out
