import os
import hashlib
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset

from .feature_utils import compute_dkt_forget_gaps, compute_history_correctness, _to_time_bin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_DIR = PROJECT_ROOT / ".cache" / "kt_dataset"

# One-by-one mode expands a multi-KC question into consecutive rows that all
# carry the SAME response, flagged by the `is_repeat` column.  Scoring those
# rows asks the model to predict a label its own input history already contains,
# so they are dropped from supervision.  On assist2009 they are 15.6% of the
# scored positions, which is why concept-level AUC used to read far above the
# question-level number.  Set KT_SCORE_REPEATED_KC=1 to reproduce old runs.
SCORE_REPEATED_KC = os.environ.get("KT_SCORE_REPEATED_KC", "0") == "1"


def _dataset_cache_path(file_path, tag):
    path = Path(file_path).resolve()
    stat = path.stat()
    cache_root = Path(os.environ.get("KT_DATASET_CACHE_DIR", DEFAULT_CACHE_DIR))
    digest_src = f"{path}|{stat.st_mtime_ns}|{stat.st_size}|{tag}"
    digest = hashlib.sha1(digest_src.encode("utf-8")).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    return cache_root / f"{path.stem}_{digest}.pkl"


def _time_maps_cache_tag(time_idx_maps):
    if not time_idx_maps:
        return "raw_time"
    parts = []
    for name in ("at2idx", "it2idx"):
        mapping = time_idx_maps.get(name, {})
        body = "|".join(f"{k}:{v}" for k, v in sorted(mapping.items()))
        parts.append(hashlib.sha1(body.encode("utf-8")).hexdigest()[:10])
    return "mapped_time_" + "_".join(parts)


def _feature_cache_tag(
    include_dkt_forget=False,
    difficulty_maps=None,
    include_history=False,
    include_hqaf_attrs=False,
    hqaf_feature_maps=None,
):
    parts = [
        "df1" if include_dkt_forget else "df0",
        "hist1" if include_history else "hist0",
    ]
    if difficulty_maps:
        bodies = []
        for name in ("skills", "questions"):
            mapping = difficulty_maps.get(name, {})
            body = "|".join(f"{k}:{v}" for k, v in sorted(mapping.items()))
            bodies.append(hashlib.sha1(body.encode("utf-8")).hexdigest()[:10])
        parts.append("diff_" + "_".join(bodies))
    else:
        parts.append("diff0")
    parts.append("hqaf1" if include_hqaf_attrs else "hqaf0")
    if hqaf_feature_maps:
        qtime = hqaf_feature_maps.get("question_avg_time", {})
        qtime_body = "|".join(f"{k}:{v}" for k, v in sorted(qtime.items()))
        edge_body = "|".join(str(v) for v in hqaf_feature_maps.get("time_bin_edges", []))
        parts.append(hashlib.sha1((qtime_body + "|" + edge_body).encode("utf-8")).hexdigest()[:10])
    # Part of the cache key: the supervision mask changes with this flag, so a
    # pickle written under the old behaviour must not be reused under the new.
    parts.append("screp1" if SCORE_REPEATED_KC else "screp0")
    return "_".join(parts)


def _difficulty_sequence(values, mapping):
    result = []
    for value in values:
        if value == -1:
            result.append(-1)
        else:
            result.append(int(mapping.get(value, 1)))
    return result


def _question_avg_time_sequence(values, mapping):
    result = []
    for value in values:
        if value == -1:
            result.append(-1)
        else:
            result.append(int(mapping.get(value, 1)))
    return result


def _time_bin_sequence(row, seq_len, hqaf_feature_maps):
    values = _parse_int_sequence(row["usetimes"]) if "usetimes" in row.index and row["usetimes"] else []
    edges = (hqaf_feature_maps or {}).get("time_bin_edges", [])
    num_time_bins = int((hqaf_feature_maps or {}).get("num_time_bins", 20))
    if not values:
        return [0] * seq_len
    result = [_to_time_bin(float(v), edges, num_time_bins) for v in values]
    return _fit_sequence(result, seq_len, 0)


def _type_sequence(row, seq_len):
    if "type" not in row.index or not row["type"]:
        return [0] * seq_len
    return _fit_sequence(_parse_int_sequence(row["type"]), seq_len, 0)


def _fit_sequence(values, length, pad_val):
    if len(values) == length:
        return values
    return (values + [pad_val] * length)[:length]


def _build_itseqs(timestamps, seq_len, time_idx_maps=None):
    if timestamps:
        shft_timestamps = timestamps[:1] + timestamps[:-1]
        values = [
            max(min((t - s) // 1000 // 60, 43200), -1)
            for t, s in zip(timestamps, shft_timestamps)
        ]
    else:
        values = [1] * seq_len
    if len(values) != seq_len:
        values = (values + [1] * seq_len)[:seq_len]

    if not time_idx_maps:
        return values

    it2idx = time_idx_maps.get("it2idx", {})
    fallback = it2idx.get("-1", 0)
    return [it2idx.get(str(int(v)), fallback) for v in values]


def _build_utseqs(row, seq_len, time_idx_maps=None):
    if "usetimes" in row.index and row["usetimes"]:
        raw_values = [int(float(t)) for t in str(row["usetimes"]).split(",") if t != ""]
    else:
        raw_values = []

    if not time_idx_maps:
        values = raw_values if raw_values else [0] * seq_len
        if len(values) != seq_len:
            values = (values + [0] * seq_len)[:seq_len]
        return values

    values = [v // 1000 for v in raw_values] if raw_values else [-1] * seq_len
    if len(values) != seq_len:
        values = (values + [-1] * seq_len)[:seq_len]

    at2idx = time_idx_maps.get("at2idx", {})
    fallback = at2idx.get("-1", 0)
    return [at2idx.get(str(int(v)), fallback) for v in values]


def _filter_rows_by_folds(df, folds, sequence_path):
    """Filter by fold when fold column exists; allow fold-less test files."""
    fold_set = set(folds)
    if "fold" in df.columns:
        df["fold"] = df["fold"].astype(int)
        filtered = df[df["fold"].isin(fold_set)]
        if filtered.empty:
            available_folds = sorted(df["fold"].unique().tolist())
            raise ValueError(
                f"No rows found in {sequence_path} for folds {sorted(fold_set)}. "
                f"Available folds: {available_folds}."
            )
        return filtered
    if fold_set == {-1}:
        return df
    raise KeyError(
        f"Column 'fold' is required in {sequence_path} for training/validation splits."
    )


def _resolve_selectmasks(row, responses):
    """Supervision mask for one sequence, as (mask, n_repeat_positions_dropped).

    Uses the provided selectmasks, or derives them from the responses when the
    column is missing or the wrong length.  Positions flagged by `is_repeat` are
    then set to -1 so they are not scored: they are the extra KC rows of a
    question whose response is already visible earlier in the same sequence.
    Question-level files carry no `is_repeat` column and are left untouched.
    """
    raw_masks = None
    if "selectmasks" in row.index and row["selectmasks"]:
        parsed = [int(x) for x in row["selectmasks"].split(",")]
        if len(parsed) == len(responses):
            raw_masks = parsed
    if raw_masks is None:
        raw_masks = [1 if r != -1 else -1 for r in responses]

    if SCORE_REPEATED_KC or "is_repeat" not in row.index or not row["is_repeat"]:
        return raw_masks, 0
    repeats = _parse_int_sequence(row["is_repeat"])
    if len(repeats) != len(raw_masks):
        return raw_masks, 0
    dropped = sum(1 for m, rep in zip(raw_masks, repeats) if rep == 1 and m != -1)
    return [-1 if rep == 1 else m for m, rep in zip(raw_masks, repeats)], dropped


def _report_repeat_filter(sequence_path, dropped, kept):
    """Say out loud how much supervision the repeat filter removed.  Silence
    here would make a protocol change look like a model change."""
    if not dropped:
        return
    total = dropped + kept
    print(
        f"Repeated-KC positions excluded from scoring: {dropped}/{total} "
        f"({dropped / total:.1%}) in {Path(sequence_path).name}"
    )


def _parse_int_sequence(value):
    if value is None or value == "":
        return []
    seq = []
    for x in str(value).split(","):
        if x == "":
            continue
        seq.append(int(float(x)))
    return seq


def _pad_1d_sequences(sequences, pad_val):
    if len(sequences) == 0:
        return sequences
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_val] * (max_len - len(seq)) for seq in sequences]


def _pad_2d_sequences(sequences, pad_row):
    if len(sequences) == 0:
        return sequences
    max_len = max(len(seq) for seq in sequences)
    return [seq + [pad_row[:] for _ in range(max_len - len(seq))] for seq in sequences]


def _sanitize_response_sequence(seq_tensor):
    """Map unknown response -1 to 0 for model inputs."""
    return torch.where(seq_tensor < 0, torch.zeros_like(seq_tensor), seq_tensor)


class KTDataset(Dataset):
    def __init__(
        self,
        file_path,
        input_type,
        folds,
        pad_val=-1,
        use_timestamps=False,
        time_idx_maps=None,
        include_dkt_forget=False,
        difficulty_maps=None,
        include_history=False,
        include_hqaf_attrs=False,
        hqaf_feature_maps=None,
    ):
        super().__init__()
        self.dataset_mode = "one_by_one"
        self.input_type = input_type
        self.pad_val = pad_val
        self.use_timestamps = use_timestamps
        self.time_idx_maps = time_idx_maps
        self.include_dkt_forget = include_dkt_forget
        self.difficulty_maps = difficulty_maps
        self.include_history = include_history
        self.include_hqaf_attrs = include_hqaf_attrs
        self.hqaf_feature_maps = hqaf_feature_maps or {}
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(f) for f in folds])
        cache_tag = (
            ("ts1" if self.use_timestamps else "ts0")
            + "_"
            + _time_maps_cache_tag(time_idx_maps)
            + "_"
            + _feature_cache_tag(
                include_dkt_forget,
                difficulty_maps,
                include_history,
                include_hqaf_attrs,
                self.hqaf_feature_maps,
            )
        )
        processed = _dataset_cache_path(file_path, f"{folds_str}_{cache_tag}_ut1_kt")

        if not os.path.exists(processed):
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)
        else:
            self.dori = pd.read_pickle(processed)

        if self.use_timestamps and ("tseqs" not in self.dori or "utseqs" not in self.dori):
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        dcur = {}
        mseqs = self.dori["masks"][index]
        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            # Handle uid specially - just pass through the scalar value
            if key == "uid":
                dcur["uid"] = self.dori["uid"][index]
                continue
            # Skip empty tensors/lists to avoid IndexError
            val = self.dori[key]
            if isinstance(val, list) and len(val) == 0:
                continue
            if hasattr(val, 'numel') and val.numel() == 0:
                continue
            cur = self.dori[key][index]
            if key == "rseqs":
                cur = _sanitize_response_sequence(cur)
            seqs = cur[:-1] * mseqs
            shft_seqs = cur[1:] * mseqs
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        return dcur

    def _load_data(self, sequence_path, folds):
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "smasks": [], "itseqs": [], "uid": []}
        repeat_dropped = 0
        scored_kept = 0
        if self.include_dkt_forget:
            dori["rgaps"], dori["sgaps"], dori["pcounts"] = [], [], []
        if self.difficulty_maps:
            dori["sdseqs"], dori["qdseqs"] = [], []
        if self.include_history:
            dori["historycorrs"] = []
        if self.include_hqaf_attrs:
            dori["quseqs"], dori["utTseqs"], dori["pTseqs"] = [], [], []
        # Load timestamps if available and requested
        if self.use_timestamps:
            dori["tseqs"] = []
            dori["utseqs"] = []
        df = pd.read_csv(sequence_path, dtype=str, keep_default_na=False)
        df = _filter_rows_by_folds(df, folds, sequence_path)

        for _, row in df.iterrows():
            # Extract uid (student ID)
            uid = int(row.get("uid", row.get("student_id", -1)))
            dori["uid"].append(uid)
            concepts = []
            questions = []
            if "concepts" in self.input_type:
                concepts = _parse_int_sequence(row["concepts"])
                dori["cseqs"].append(concepts)
            if "questions" in self.input_type:
                questions = _parse_int_sequence(row["questions"])
                dori["qseqs"].append(questions)
            responses = _parse_int_sequence(row["responses"])
            dori["rseqs"].append(responses)
            smask, n_repeat_dropped = _resolve_selectmasks(row, responses)
            dori["smasks"].append(smask)
            repeat_dropped += n_repeat_dropped
            scored_kept += sum(1 for m in smask if m != -1)
            if self.include_dkt_forget:
                if "timestamps" not in row.index or not row["timestamps"]:
                    raise ValueError(f"DKT-forget requires timestamps in {sequence_path}.")
                rgap, sgap, pcount = compute_dkt_forget_gaps(row, self.input_type)
                dori["rgaps"].append(_fit_sequence(rgap, len(responses), 0))
                dori["sgaps"].append(_fit_sequence(sgap, len(responses), 0))
                dori["pcounts"].append(_fit_sequence(pcount, len(responses), 0))
            if self.difficulty_maps:
                skill_map = self.difficulty_maps.get("skills", {})
                question_map = self.difficulty_maps.get("questions", {})
                dori["sdseqs"].append(_fit_sequence(_difficulty_sequence(concepts, skill_map), len(responses), -1))
                dori["qdseqs"].append(_fit_sequence(_difficulty_sequence(questions, question_map), len(responses), -1))
            if self.include_history:
                dori["historycorrs"].append(_fit_sequence(compute_history_correctness(concepts, responses), len(responses), 0.0))
            if self.include_hqaf_attrs:
                qtime_map = self.hqaf_feature_maps.get("question_avg_time", {})
                dori["quseqs"].append(_fit_sequence(_question_avg_time_sequence(questions, qtime_map), len(responses), -1))
                dori["utTseqs"].append(_time_bin_sequence(row, len(responses), self.hqaf_feature_maps))
                dori["pTseqs"].append(_type_sequence(row, len(responses)))
            if self.use_timestamps:
                timestamps = []
                if "timestamps" in row.index and row["timestamps"]:
                    timestamps = _parse_int_sequence(row["timestamps"])
                raw_timestamps = timestamps
                if len(timestamps) != len(responses):
                    timestamps = (timestamps + [0] * len(responses))[:len(responses)]
                dori["tseqs"].append(timestamps)
                dori["itseqs"].append(_build_itseqs(raw_timestamps, len(responses), self.time_idx_maps))
                dori["utseqs"].append(_build_utseqs(row, len(responses), self.time_idx_maps))

        _report_repeat_filter(sequence_path, repeat_dropped, scored_kept)

        dori["cseqs"] = _pad_1d_sequences(dori["cseqs"], self.pad_val)
        dori["qseqs"] = _pad_1d_sequences(dori["qseqs"], self.pad_val)
        dori["rseqs"] = _pad_1d_sequences(dori["rseqs"], self.pad_val)
        dori["smasks"] = _pad_1d_sequences(dori["smasks"], self.pad_val)
        if self.include_dkt_forget:
            dori["rgaps"] = _pad_1d_sequences(dori["rgaps"], self.pad_val)
            dori["sgaps"] = _pad_1d_sequences(dori["sgaps"], self.pad_val)
            dori["pcounts"] = _pad_1d_sequences(dori["pcounts"], self.pad_val)
        if self.difficulty_maps:
            dori["sdseqs"] = _pad_1d_sequences(dori["sdseqs"], self.pad_val)
            dori["qdseqs"] = _pad_1d_sequences(dori["qdseqs"], self.pad_val)
        if self.include_history:
            dori["historycorrs"] = _pad_1d_sequences(dori["historycorrs"], 0.0)
        if self.include_hqaf_attrs:
            dori["quseqs"] = _pad_1d_sequences(dori["quseqs"], self.pad_val)
            dori["utTseqs"] = _pad_1d_sequences(dori["utTseqs"], self.pad_val)
            dori["pTseqs"] = _pad_1d_sequences(dori["pTseqs"], self.pad_val)
        if self.use_timestamps:
            dori["tseqs"] = _pad_1d_sequences(dori.get("tseqs", []), self.pad_val)
            dori["itseqs"] = _pad_1d_sequences(dori.get("itseqs", []), self.pad_val)
            dori["utseqs"] = _pad_1d_sequences(dori.get("utseqs", []), self.pad_val)

        if len(dori["cseqs"]) > 0:
            seq_for_mask = torch.tensor(dori["cseqs"], dtype=torch.long)
        elif len(dori["qseqs"]) > 0:
            seq_for_mask = torch.tensor(dori["qseqs"], dtype=torch.long)
        else:
            raise ValueError("No concepts/questions found in sequence file.")

        if len(dori["cseqs"]) > 0:
            dori["cseqs"] = torch.tensor(dori["cseqs"], dtype=torch.long)
        else:
            dori["cseqs"] = torch.tensor([])
        if len(dori["qseqs"]) > 0:
            dori["qseqs"] = torch.tensor(dori["qseqs"], dtype=torch.long)
        else:
            dori["qseqs"] = torch.tensor([])
        dori["rseqs"] = torch.tensor(dori["rseqs"], dtype=torch.float)
        dori["smasks"] = torch.tensor(dori["smasks"], dtype=torch.long)
        dori["uid"] = torch.tensor(dori["uid"], dtype=torch.long)

        # Convert timestamps to tensor if loaded
        if self.use_timestamps and "tseqs" in dori and len(dori["tseqs"]) > 0:
            dori["tseqs"] = torch.tensor(dori["tseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["tseqs"] = torch.tensor([])

        # Convert itseqs to tensor if loaded
        if self.use_timestamps and "itseqs" in dori and len(dori["itseqs"]) > 0:
            dori["itseqs"] = torch.tensor(dori["itseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["itseqs"] = torch.tensor([], dtype=torch.long)

        if self.use_timestamps and "utseqs" in dori and len(dori["utseqs"]) > 0:
            dori["utseqs"] = torch.tensor(dori["utseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["utseqs"] = torch.tensor([], dtype=torch.long)

        for key in ("rgaps", "sgaps", "pcounts", "sdseqs", "qdseqs", "quseqs", "utTseqs", "pTseqs"):
            if key in dori:
                dori[key] = torch.tensor(dori[key], dtype=torch.long)
        if "historycorrs" in dori:
            dori["historycorrs"] = torch.tensor(dori["historycorrs"], dtype=torch.float)

        dori["masks"] = (seq_for_mask[:, :-1] != self.pad_val) & (
            seq_for_mask[:, 1:] != self.pad_val
        )
        dori["smasks"] = dori["smasks"][:, 1:] != self.pad_val
        return dori


class KTQueDataset(Dataset):
    """Question-level dataset for ALL-in-One mode.

    Supports 2D concept sequences [batch, seq_len, max_concepts].
    Each concept position can contain multiple concepts separated by "_".
    """
    def __init__(
        self,
        file_path,
        input_type,
        folds,
        concept_num,
        max_concepts=4,
        pad_val=-1,
        concept_mode: str = "multi",
        use_timestamps=False,
        time_idx_maps=None,
        include_dkt_forget=False,
        difficulty_maps=None,
        include_history=False,
        include_hqaf_attrs=False,
        hqaf_feature_maps=None,
    ):
        super().__init__()
        self.dataset_mode = "all_in_one"
        self.input_type = input_type
        self.concept_num = concept_num
        self.max_concepts = max_concepts
        self.pad_val = pad_val
        self.concept_mode = concept_mode
        self.use_timestamps = use_timestamps
        self.time_idx_maps = time_idx_maps
        self.include_dkt_forget = include_dkt_forget
        self.difficulty_maps = difficulty_maps
        self.include_history = include_history
        self.include_hqaf_attrs = include_hqaf_attrs
        self.hqaf_feature_maps = hqaf_feature_maps or {}
        if concept_mode == "first" and int(max_concepts or 1) > 1:
            # Silence here is how a comparison table ends up mixing models that
            # saw every KC of a question with models that saw only the first.
            print(
                f"Concept truncation: this model reads KC 1 of up to "
                f"{int(max_concepts)} per question; the rest are discarded. "
                f"Only comparable with other concept_mode='first' runs."
            )
        folds = sorted(list(folds))
        folds_str = "_" + "_".join([str(f) for f in folds])
        cache_tag = (
            ("ts1" if self.use_timestamps else "ts0")
            + "_"
            + _time_maps_cache_tag(time_idx_maps)
            + "_"
            + _feature_cache_tag(
                include_dkt_forget,
                difficulty_maps,
                include_history,
                include_hqaf_attrs,
                self.hqaf_feature_maps,
            )
        )
        processed = _dataset_cache_path(
            file_path,
            f"{folds_str}_{cache_tag}_{concept_mode}_mc{max_concepts}_ut1_qlevel",
        )

        if not os.path.exists(processed):
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)
        else:
            self.dori = pd.read_pickle(processed)

        if self.use_timestamps and ("tseqs" not in self.dori or "utseqs" not in self.dori):
            self.dori = self._load_data(file_path, folds)
            pd.to_pickle(self.dori, processed)

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        dcur = {}
        mseqs = self.dori["masks"][index]

        def _apply_time_mask(x, m):
            mask = m
            while mask.dim() < x.dim():
                mask = mask.unsqueeze(-1)
            return x * mask

        for key in self.dori:
            if key in ["masks", "smasks"]:
                continue
            # Handle uid specially - just pass through the scalar value
            if key == "uid":
                dcur["uid"] = self.dori["uid"][index]
                continue
            # Skip empty tensors/lists to avoid IndexError
            val = self.dori[key]
            if isinstance(val, list) and len(val) == 0:
                continue
            if hasattr(val, 'numel') and val.numel() == 0:
                continue

            cur = self.dori[key][index]
            if key == "rseqs":
                cur = _sanitize_response_sequence(cur)
            if key == "cseqs" and cur.dim() >= 2 and self.concept_mode == "first":
                # Keeps only KC 1 and discards the rest.  Announced once at
                # construction -- a table that mixes this with concept_mode
                # 'multi' is comparing models shown different questions.
                cur = cur[:, 0]

            seqs = _apply_time_mask(cur[:-1], mseqs)
            shft_seqs = _apply_time_mask(cur[1:], mseqs)
            dcur[key] = seqs
            dcur["shft_" + key] = shft_seqs
        dcur["masks"] = mseqs
        dcur["smasks"] = self.dori["smasks"][index]
        return dcur

    def _load_data(self, sequence_path, folds):
        """Load data with 2D concept sequences.

        Concepts are stored as [seq_len, max_concepts] 2D array.
        Format: "1_2_3_-1" means concepts [1,2,3,-1] at one position.
        """
        dori = {"qseqs": [], "cseqs": [], "rseqs": [], "smasks": [], "itseqs": [], "uid": []}
        repeat_dropped = 0
        scored_kept = 0
        if self.include_dkt_forget:
            dori["rgaps"], dori["sgaps"], dori["pcounts"] = [], [], []
        if self.difficulty_maps:
            dori["sdseqs"], dori["qdseqs"] = [], []
        if self.include_history:
            dori["historycorrs"] = []
        if self.include_hqaf_attrs:
            dori["quseqs"], dori["utTseqs"], dori["pTseqs"] = [], [], []
        # Load timestamps if available and requested
        if self.use_timestamps:
            dori["tseqs"] = []
            dori["utseqs"] = []
        df = pd.read_csv(sequence_path, dtype=str, keep_default_na=False)
        df = _filter_rows_by_folds(df, folds, sequence_path)

        for _, row in df.iterrows():
            # Extract uid (student ID)
            uid = int(row.get("uid", row.get("student_id", -1)))
            dori["uid"].append(uid)
            first_concepts = []
            questions = []
            if "concepts" in self.input_type:
                row_skills = []
                raw_skills = row["concepts"].split(",")
                for concept in raw_skills:
                    if concept == "-1":
                        skills = [-1] * self.max_concepts
                    else:
                        skills = [int(float(_)) for _ in concept.split("_") if _ != ""]
                        if len(skills) > self.max_concepts:
                            raise ValueError(
                                f"Question has {len(skills)} concepts in {sequence_path}, "
                                f"exceeding max_concepts={self.max_concepts}."
                            )
                        skills = skills + [-1] * (self.max_concepts - len(skills))
                    row_skills.append(skills)
                    first_concepts.append(skills[0] if skills else -1)
                dori["cseqs"].append(row_skills)  # 2D: [seq_len, max_concepts]
            if "questions" in self.input_type:
                questions = _parse_int_sequence(row["questions"])
                dori["qseqs"].append(questions)
            responses = _parse_int_sequence(row["responses"])
            dori["rseqs"].append(responses)
            smask, n_repeat_dropped = _resolve_selectmasks(row, responses)
            dori["smasks"].append(smask)
            repeat_dropped += n_repeat_dropped
            scored_kept += sum(1 for m in smask if m != -1)
            if self.include_dkt_forget:
                rgap, sgap, pcount = compute_dkt_forget_gaps(row, self.input_type)
                dori["rgaps"].append(_fit_sequence(rgap, len(responses), 0))
                dori["sgaps"].append(_fit_sequence(sgap, len(responses), 0))
                dori["pcounts"].append(_fit_sequence(pcount, len(responses), 0))
            if self.difficulty_maps:
                skill_map = self.difficulty_maps.get("skills", {})
                question_map = self.difficulty_maps.get("questions", {})
                dori["sdseqs"].append(_fit_sequence(_difficulty_sequence(first_concepts, skill_map), len(responses), -1))
                dori["qdseqs"].append(_fit_sequence(_difficulty_sequence(questions, question_map), len(responses), -1))
            if self.include_history:
                dori["historycorrs"].append(_fit_sequence(compute_history_correctness(first_concepts, responses), len(responses), 0.0))
            if self.include_hqaf_attrs:
                qtime_map = self.hqaf_feature_maps.get("question_avg_time", {})
                dori["quseqs"].append(_fit_sequence(_question_avg_time_sequence(questions, qtime_map), len(responses), -1))
                dori["utTseqs"].append(_time_bin_sequence(row, len(responses), self.hqaf_feature_maps))
                dori["pTseqs"].append(_type_sequence(row, len(responses)))
            if self.use_timestamps:
                timestamps = []
                if "timestamps" in row.index and row["timestamps"]:
                    timestamps = _parse_int_sequence(row["timestamps"])
                raw_timestamps = timestamps
                if len(timestamps) != len(responses):
                    timestamps = (timestamps + [0] * len(responses))[:len(responses)]
                dori["tseqs"].append(timestamps)
                dori["itseqs"].append(_build_itseqs(raw_timestamps, len(responses), self.time_idx_maps))
                dori["utseqs"].append(_build_utseqs(row, len(responses), self.time_idx_maps))

        _report_repeat_filter(sequence_path, repeat_dropped, scored_kept)

        dori["cseqs"] = _pad_2d_sequences(dori["cseqs"], [-1] * self.max_concepts)
        dori["qseqs"] = _pad_1d_sequences(dori["qseqs"], self.pad_val)
        dori["rseqs"] = _pad_1d_sequences(dori["rseqs"], self.pad_val)
        dori["smasks"] = _pad_1d_sequences(dori["smasks"], self.pad_val)
        if self.include_dkt_forget:
            dori["rgaps"] = _pad_1d_sequences(dori["rgaps"], self.pad_val)
            dori["sgaps"] = _pad_1d_sequences(dori["sgaps"], self.pad_val)
            dori["pcounts"] = _pad_1d_sequences(dori["pcounts"], self.pad_val)
        if self.difficulty_maps:
            dori["sdseqs"] = _pad_1d_sequences(dori["sdseqs"], self.pad_val)
            dori["qdseqs"] = _pad_1d_sequences(dori["qdseqs"], self.pad_val)
        if self.include_history:
            dori["historycorrs"] = _pad_1d_sequences(dori["historycorrs"], 0.0)
        if self.include_hqaf_attrs:
            dori["quseqs"] = _pad_1d_sequences(dori["quseqs"], self.pad_val)
            dori["utTseqs"] = _pad_1d_sequences(dori["utTseqs"], self.pad_val)
            dori["pTseqs"] = _pad_1d_sequences(dori["pTseqs"], self.pad_val)
        if self.use_timestamps:
            dori["tseqs"] = _pad_1d_sequences(dori.get("tseqs", []), self.pad_val)
            dori["itseqs"] = _pad_1d_sequences(dori.get("itseqs", []), self.pad_val)
            dori["utseqs"] = _pad_1d_sequences(dori.get("utseqs", []), self.pad_val)

        # Convert to tensors
        if len(dori["cseqs"]) > 0:
            # cseqs is 2D: [num_samples, seq_len, max_concepts]
            dori["cseqs"] = torch.tensor(dori["cseqs"], dtype=torch.long)
            seq_for_mask = (dori["cseqs"] >= 0).any(dim=-1)
        elif len(dori["qseqs"]) > 0:
            seq_for_mask = torch.tensor(dori["qseqs"], dtype=torch.long)
        else:
            raise ValueError("No concepts/questions found in sequence file.")

        if len(dori["qseqs"]) > 0:
            dori["qseqs"] = torch.tensor(dori["qseqs"], dtype=torch.long)
        else:
            dori["qseqs"] = torch.tensor([])

        dori["rseqs"] = torch.tensor(dori["rseqs"], dtype=torch.float)
        dori["smasks"] = torch.tensor(dori["smasks"], dtype=torch.long)
        dori["uid"] = torch.tensor(dori["uid"], dtype=torch.long)

        # Convert timestamps to tensor if loaded
        if self.use_timestamps and "tseqs" in dori and len(dori["tseqs"]) > 0:
            dori["tseqs"] = torch.tensor(dori["tseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["tseqs"] = torch.tensor([])

        # Convert itseqs to tensor if loaded (for KTQueDataset)
        if self.use_timestamps and "itseqs" in dori and len(dori["itseqs"]) > 0:
            dori["itseqs"] = torch.tensor(dori["itseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["itseqs"] = torch.tensor([], dtype=torch.long)

        if self.use_timestamps and "utseqs" in dori and len(dori["utseqs"]) > 0:
            dori["utseqs"] = torch.tensor(dori["utseqs"], dtype=torch.long)
        elif self.use_timestamps:
            dori["utseqs"] = torch.tensor([], dtype=torch.long)

        for key in ("rgaps", "sgaps", "pcounts", "sdseqs", "qdseqs", "quseqs", "utTseqs", "pTseqs"):
            if key in dori:
                dori[key] = torch.tensor(dori[key], dtype=torch.long)
        if "historycorrs" in dori:
            dori["historycorrs"] = torch.tensor(dori["historycorrs"], dtype=torch.float)

        if seq_for_mask.dtype == torch.bool:
            dori["masks"] = seq_for_mask[:, :-1] & seq_for_mask[:, 1:]
        else:
            dori["masks"] = (seq_for_mask[:, :-1] != self.pad_val) & (
                seq_for_mask[:, 1:] != self.pad_val
            )
        dori["smasks"] = dori["smasks"][:, 1:] != self.pad_val
        return dori
