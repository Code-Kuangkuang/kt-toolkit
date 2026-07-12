"""Build a reproducible 5,000-student subset of the processed Junyi2015 data.

The builder samples users from the unsliced question-level files, then filters the
already aligned raw and max-length sequence representations used by KT-Toolkit.
It never writes into the source directory and it publishes the output only after
all representations contain every selected user.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import shutil
import statistics
import sys
import tempfile
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


ALGORITHM_VERSION = 1
DATASET_NAME = "junyi_sub5k"
DISPLAY_NAME = "Junyi-sub-5k (ours)"
METADATA_FILES = ("keyid2idx.json", "junyi_Exercise_table.csv")


@dataclass(frozen=True)
class ViewSpec:
    name: str
    train_file: str
    test_file: str
    one_row_per_user: bool
    collect_statistics: bool = False


VIEW_SPECS = (
    ViewSpec(
        "question_raw",
        "train_valid_quelevel.csv",
        "test_quelevel.csv",
        one_row_per_user=True,
        collect_statistics=True,
    ),
    ViewSpec(
        "question_sequences",
        "train_valid_sequences_quelevel.csv",
        "test_sequences_quelevel.csv",
        one_row_per_user=False,
    ),
    ViewSpec(
        "concept_raw",
        "train_valid.csv",
        "test.csv",
        one_row_per_user=True,
    ),
    ViewSpec(
        "concept_sequences",
        "train_valid_sequences.csv",
        "test_sequences.csv",
        one_row_per_user=False,
    ),
)

SELECTION_FILES = (
    "train_valid_quelevel.csv",
    "test_quelevel.csv",
)
HASHED_SOURCE_FILES = SELECTION_FILES + METADATA_FILES
SEQUENCE_COLUMNS = ("questions", "concepts", "responses", "timestamps", "usetimes")


def _set_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


_set_csv_field_limit()


def _uid_sort_key(uid: str) -> tuple[int, int | str, str]:
    try:
        return (0, int(uid), uid)
    except ValueError:
        return (1, uid, uid)


def _json_uid(uid: str) -> int | str:
    try:
        return int(uid)
    except ValueError:
        return uid


def _split_values(value: str) -> list[str]:
    if value is None or value == "":
        return []
    return value.split(",")


def _valid_interaction_count(row: dict[str, str], source_name: str) -> int:
    missing = [column for column in SEQUENCE_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"{source_name} is missing columns: {missing}")

    lengths = {column: len(_split_values(row[column])) for column in SEQUENCE_COLUMNS}
    if len(set(lengths.values())) != 1:
        raise ValueError(
            f"unaligned sequence for uid={row.get('uid')!r} in {source_name}: {lengths}"
        )
    return sum(value != "-1" for value in _split_values(row["responses"]))


def _required_source_names() -> tuple[str, ...]:
    names: list[str] = []
    for spec in VIEW_SPECS:
        names.extend((spec.train_file, spec.test_file))
    names.extend(METADATA_FILES)
    return tuple(dict.fromkeys(names))


def _validate_source_files(source_dir: Path) -> None:
    missing = [name for name in _required_source_names() if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(
            f"Junyi source directory {source_dir} is missing required files: {missing}"
        )


def _file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fin:
        while chunk := fin.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _source_snapshot(source_dir: Path) -> dict[str, dict[str, Any]]:
    snapshot: dict[str, dict[str, Any]] = {}
    for name in _required_source_names():
        stat = (source_dir / name).stat()
        snapshot[name] = {
            "size_bytes": stat.st_size,
            "modified_time_ns": stat.st_mtime_ns,
        }
    return snapshot


def _read_user_profiles(source_dir: Path) -> dict[str, int]:
    profiles: dict[str, int] = {}
    for name in SELECTION_FILES:
        path = source_dir / name
        print(f"[profiles] scanning {path}", flush=True)
        with path.open("r", encoding="utf-8-sig", newline="") as fin:
            reader = csv.DictReader(fin)
            if reader.fieldnames is None or "uid" not in reader.fieldnames:
                raise ValueError(f"{path} does not contain a uid column")
            for row_number, row in enumerate(reader, start=2):
                uid = (row.get("uid") or "").strip()
                if not uid:
                    raise ValueError(f"empty uid in {path} at row {row_number}")
                if uid in profiles:
                    raise ValueError(f"duplicate uid={uid!r} across Junyi raw files")
                profiles[uid] = _valid_interaction_count(row, name)
        print(f"[profiles] collected {len(profiles):,} users so far", flush=True)
    return profiles


def _sample_users(
    profiles: dict[str, int],
    *,
    students: int,
    strata: int,
    min_seq_len: int,
    seed: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    eligible = [
        (uid, length) for uid, length in profiles.items() if length >= min_seq_len
    ]
    eligible.sort(key=lambda item: (item[1], _uid_sort_key(item[0])))
    if len(eligible) < students:
        raise ValueError(
            f"only {len(eligible):,} users meet min_seq_len={min_seq_len}; "
            f"cannot sample {students:,}"
        )
    if students % strata != 0:
        raise ValueError("students must be divisible by strata for equal allocation")

    groups: list[list[tuple[str, int]]] = [[] for _ in range(strata)]
    for rank, item in enumerate(eligible):
        group_index = min(strata - 1, rank * strata // len(eligible))
        groups[group_index].append(item)

    quota = students // strata
    rng = random.Random(seed)
    selected: dict[str, dict[str, Any]] = {}
    strata_manifest: list[dict[str, Any]] = []
    for stratum, group in enumerate(groups):
        if len(group) < quota:
            raise ValueError(
                f"stratum {stratum} has {len(group)} eligible users, below quota {quota}"
            )
        sampled = rng.sample(group, quota)
        for uid, sequence_length in sampled:
            selected[uid] = {
                "sequence_length": sequence_length,
                "stratum": stratum,
            }
        group_lengths = [length for _, length in group]
        sampled_lengths = [length for _, length in sampled]
        strata_manifest.append(
            {
                "stratum": stratum,
                "eligible_students": len(group),
                "eligible_length_min": min(group_lengths),
                "eligible_length_max": max(group_lengths),
                "selected_students": len(sampled),
                "selected_length_min": min(sampled_lengths),
                "selected_length_max": max(sampled_lengths),
            }
        )

    return selected, {
        "source_students": len(profiles),
        "eligible_students": len(eligible),
        "eligible_length_min": eligible[0][1],
        "eligible_length_max": eligible[-1][1],
        "strata": strata_manifest,
    }


def _assign_splits_and_folds(
    selected: dict[str, dict[str, Any]],
    *,
    test_ratio: float,
    folds: int,
    seed: int,
) -> dict[str, Any]:
    selected_uids = sorted(selected, key=_uid_sort_key)
    test_students = int(round(len(selected_uids) * test_ratio))
    train_students = len(selected_uids) - test_students
    if test_students <= 0 or train_students <= 0:
        raise ValueError("test_ratio must leave at least one user in both splits")
    if train_students % folds != 0:
        raise ValueError(
            f"train_valid users ({train_students}) must be divisible by folds ({folds})"
        )

    split_order = selected_uids.copy()
    random.Random(seed + 1).shuffle(split_order)
    test_uids = set(split_order[:test_students])
    train_uids = [uid for uid in selected_uids if uid not in test_uids]

    fold_order = train_uids.copy()
    random.Random(seed + 2).shuffle(fold_order)
    fold_counts = {fold: 0 for fold in range(folds)}
    for index, uid in enumerate(fold_order):
        fold = index % folds
        selected[uid].update({"split": "train_valid", "fold": fold})
        fold_counts[fold] += 1
    for uid in test_uids:
        selected[uid].update({"split": "test", "fold": -1})

    return {
        "train_valid_students": train_students,
        "test_students": test_students,
        "fold_students": {str(fold): count for fold, count in fold_counts.items()},
    }


def _read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.reader(fin)
        try:
            return next(reader)
        except StopIteration as exc:
            raise ValueError(f"empty CSV file: {path}") from exc


def _union_headers(paths: Iterable[Path]) -> list[str]:
    fields: list[str] = []
    for path in paths:
        for field in _read_header(path):
            if field not in fields:
                fields.append(field)
    for required in ("fold", "uid"):
        if required not in fields:
            raise ValueError(f"source view is missing required column {required!r}")
    return fields


class StatisticsCollector:
    def __init__(self, question_to_area: dict[str, str]):
        self.question_to_area = question_to_area
        self._groups: dict[str, dict[str, Any]] = defaultdict(self._new_group)
        self.unmapped_question_ids: set[str] = set()

    @staticmethod
    def _new_group() -> dict[str, Any]:
        return {
            "students": set(),
            "questions": set(),
            "concepts": set(),
            "areas": set(),
            "interactions": 0,
            "correct": 0,
            "sequence_lengths": [],
        }

    def add(self, row: dict[str, str], split: str, expected_length: int) -> None:
        lengths = {column: len(_split_values(row[column])) for column in SEQUENCE_COLUMNS}
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"unaligned selected raw sequence for uid={row.get('uid')!r}: {lengths}"
            )
        questions = _split_values(row["questions"])
        concepts = _split_values(row["concepts"])
        responses = _split_values(row["responses"])
        valid_positions = [index for index, value in enumerate(responses) if value != "-1"]
        if len(valid_positions) != expected_length:
            raise ValueError(
                f"uid={row.get('uid')!r} changed length: expected {expected_length}, "
                f"found {len(valid_positions)}"
            )

        for group_name in ("overall", split):
            group = self._groups[group_name]
            group["students"].add(row["uid"])
            group["interactions"] += len(valid_positions)
            group["sequence_lengths"].append(len(valid_positions))
            for index in valid_positions:
                question = questions[index]
                concept = concepts[index]
                response = responses[index]
                if question != "-1":
                    group["questions"].add(question)
                    area = self.question_to_area.get(question)
                    if area:
                        group["areas"].add(area)
                    else:
                        self.unmapped_question_ids.add(question)
                if concept != "-1":
                    for concept_id in concept.split("_"):
                        if concept_id and concept_id != "-1":
                            group["concepts"].add(concept_id)
                if response == "1":
                    group["correct"] += 1

    @staticmethod
    def _percentile(values: list[int], fraction: float) -> float | None:
        if not values:
            return None
        ordered = sorted(values)
        position = (len(ordered) - 1) * fraction
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return float(ordered[lower])
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    def finalize(self) -> dict[str, dict[str, Any]]:
        result: dict[str, dict[str, Any]] = {}
        for name in ("overall", "train_valid", "test"):
            group = self._groups[name]
            lengths = group["sequence_lengths"]
            interactions = group["interactions"]
            result[name] = {
                "students": len(group["students"]),
                "questions": len(group["questions"]),
                "interactions": interactions,
                "knowledge_areas": len(group["areas"]),
                "knowledge_concepts": len(group["concepts"]),
                "correct_rate": (
                    round(group["correct"] / interactions, 8) if interactions else None
                ),
                "sequence_length": {
                    "min": min(lengths) if lengths else None,
                    "p25": self._percentile(lengths, 0.25),
                    "median": statistics.median(lengths) if lengths else None,
                    "mean": round(statistics.fmean(lengths), 6) if lengths else None,
                    "p75": self._percentile(lengths, 0.75),
                    "max": max(lengths) if lengths else None,
                },
            }
        return result


def _load_metadata(
    source_dir: Path,
) -> tuple[dict[int | str, str], dict[str, str], dict[str, Any]]:
    mapping_path = source_dir / "keyid2idx.json"
    with mapping_path.open("r", encoding="utf-8") as fin:
        mapping = json.load(fin)
    for key in ("questions", "concepts", "uid"):
        if key not in mapping or not isinstance(mapping[key], dict):
            raise ValueError(f"{mapping_path} is missing mapping {key!r}")

    uid_inverse: dict[int | str, str] = {
        _json_uid(str(mapped_uid)): str(source_uid)
        for source_uid, mapped_uid in mapping["uid"].items()
    }

    exercise_path = source_dir / "junyi_Exercise_table.csv"
    name_to_areas: dict[str, set[str]] = defaultdict(set)
    with exercise_path.open("r", encoding="utf-8-sig", newline="") as fin:
        reader = csv.DictReader(fin)
        if reader.fieldnames is None or not {"name", "area"}.issubset(reader.fieldnames):
            raise ValueError(f"{exercise_path} must contain name and area columns")
        for row in reader:
            name = (row.get("name") or "").strip()
            area = (row.get("area") or "").strip()
            if name and area:
                name_to_areas[name].add(area)

    question_to_area: dict[str, str] = {}
    ambiguous_names: list[str] = []
    missing_names: list[str] = []
    for encoded_name, question_id in mapping["questions"].items():
        exercise_name = encoded_name.replace("####", "_")
        areas = name_to_areas.get(exercise_name, set())
        if len(areas) == 1:
            question_to_area[str(question_id)] = next(iter(areas))
        elif len(areas) > 1:
            ambiguous_names.append(exercise_name)
        else:
            missing_names.append(exercise_name)

    metadata_summary = {
        "mapped_questions": len(mapping["questions"]),
        "mapped_concepts": len(mapping["concepts"]),
        "mapped_users": len(mapping["uid"]),
        "question_area_matches": len(question_to_area),
        "question_area_missing": len(missing_names),
        "question_area_ambiguous": len(ambiguous_names),
        "missing_question_names": missing_names[:20],
        "ambiguous_question_names": ambiguous_names[:20],
    }
    return uid_inverse, question_to_area, metadata_summary


def _filter_view(
    source_dir: Path,
    staging_dir: Path,
    spec: ViewSpec,
    selected: dict[str, dict[str, Any]],
    statistics_collector: StatisticsCollector,
) -> dict[str, Any]:
    source_paths = (source_dir / spec.train_file, source_dir / spec.test_file)
    fieldnames = _union_headers(source_paths)
    target_paths = {
        "train_valid": staging_dir / spec.train_file,
        "test": staging_dir / spec.test_file,
    }
    seen_counts: dict[str, int] = defaultdict(int)
    source_rows = 0
    output_rows = {"train_valid": 0, "test": 0}

    print(
        f"[{spec.name}] filtering {spec.train_file} + {spec.test_file}", flush=True
    )
    with target_paths["train_valid"].open(
        "w", encoding="utf-8", newline=""
    ) as train_out, target_paths["test"].open(
        "w", encoding="utf-8", newline=""
    ) as test_out:
        writers = {
            "train_valid": csv.DictWriter(train_out, fieldnames=fieldnames),
            "test": csv.DictWriter(test_out, fieldnames=fieldnames),
        }
        for writer in writers.values():
            writer.writeheader()

        for source_path in source_paths:
            with source_path.open("r", encoding="utf-8-sig", newline="") as fin:
                reader = csv.DictReader(fin)
                if reader.fieldnames is None:
                    raise ValueError(f"empty CSV file: {source_path}")
                for row in reader:
                    source_rows += 1
                    uid = (row.get("uid") or "").strip()
                    selection = selected.get(uid)
                    if selection is None:
                        continue
                    split = selection["split"]
                    output_row = {field: row.get(field, "") for field in fieldnames}
                    output_row["fold"] = str(selection["fold"])
                    writers[split].writerow(output_row)
                    output_rows[split] += 1
                    seen_counts[uid] += 1
                    if spec.collect_statistics:
                        statistics_collector.add(
                            output_row, split, selection["sequence_length"]
                        )

    missing = sorted(set(selected).difference(seen_counts), key=_uid_sort_key)
    if missing:
        raise ValueError(
            f"{spec.name} is missing {len(missing)} selected users; first={missing[:10]}"
        )
    if spec.one_row_per_user:
        duplicates = sorted(
            (uid for uid, count in seen_counts.items() if count != 1), key=_uid_sort_key
        )
        if duplicates:
            raise ValueError(
                f"{spec.name} does not have exactly one row for selected users: "
                f"{duplicates[:10]}"
            )

    print(
        f"[{spec.name}] wrote {output_rows['train_valid']:,} train_valid and "
        f"{output_rows['test']:,} test rows",
        flush=True,
    )
    return {
        "source_rows_scanned": source_rows,
        "train_valid_rows": output_rows["train_valid"],
        "test_rows": output_rows["test"],
        "columns": fieldnames,
    }


def _write_selected_users(
    path: Path,
    selected: dict[str, dict[str, Any]],
    uid_inverse: dict[int | str, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for uid in sorted(selected, key=_uid_sort_key):
        selection = selected[uid]
        json_uid = _json_uid(uid)
        rows.append(
            {
                "uid": json_uid,
                "source_uid": uid_inverse.get(json_uid),
                "sequence_length": selection["sequence_length"],
                "stratum": selection["stratum"],
                "split": selection["split"],
                "fold": selection["fold"],
            }
        )
    with path.open("w", encoding="utf-8", newline="\n") as fout:
        json.dump(rows, fout, ensure_ascii=False, indent=2)
        fout.write("\n")
    return rows


def _write_statistics_csv(path: Path, statistics_data: dict[str, dict[str, Any]]) -> None:
    fieldnames = [
        "split",
        "students",
        "questions",
        "interactions",
        "knowledge_areas",
        "knowledge_concepts",
        "correct_rate",
        "sequence_length_min",
        "sequence_length_p25",
        "sequence_length_median",
        "sequence_length_mean",
        "sequence_length_p75",
        "sequence_length_max",
    ]
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        for split in ("overall", "train_valid", "test"):
            values = statistics_data[split]
            length = values["sequence_length"]
            writer.writerow(
                {
                    "split": split,
                    "students": values["students"],
                    "questions": values["questions"],
                    "interactions": values["interactions"],
                    "knowledge_areas": values["knowledge_areas"],
                    "knowledge_concepts": values["knowledge_concepts"],
                    "correct_rate": values["correct_rate"],
                    "sequence_length_min": length["min"],
                    "sequence_length_p25": length["p25"],
                    "sequence_length_median": length["median"],
                    "sequence_length_mean": length["mean"],
                    "sequence_length_p75": length["p75"],
                    "sequence_length_max": length["max"],
                }
            )


def _publish_directory(
    staging_dir: Path,
    output_dir: Path,
    *,
    attempts: int = 10,
    delay_seconds: float = 0.5,
) -> None:
    """Atomically publish a directory, tolerating transient Windows handles."""

    if attempts < 1:
        raise ValueError("publish attempts must be positive")
    for attempt in range(1, attempts + 1):
        try:
            os.replace(staging_dir, output_dir)
            return
        except PermissionError:
            if attempt == attempts:
                raise
            print(
                f"[publish] directory is temporarily busy; retrying "
                f"({attempt}/{attempts})",
                flush=True,
            )
            time.sleep(delay_seconds)


def _validate_arguments(
    *, students: int, strata: int, min_seq_len: int, test_ratio: float, folds: int
) -> None:
    if students <= 0:
        raise ValueError("students must be positive")
    if strata <= 0:
        raise ValueError("strata must be positive")
    if min_seq_len <= 0:
        raise ValueError("min_seq_len must be positive")
    if not 0 < test_ratio < 1:
        raise ValueError("test_ratio must be between 0 and 1")
    if folds <= 1:
        raise ValueError("folds must be greater than 1")


def build_subset(
    *,
    source_dir: str | Path,
    output_dir: str | Path,
    students: int = 5000,
    strata: int = 5,
    min_seq_len: int = 3,
    test_ratio: float = 0.2,
    folds: int = 5,
    seed: int = 3407,
) -> dict[str, Any]:
    """Build and atomically publish a Junyi subset, returning its manifest."""

    _validate_arguments(
        students=students,
        strata=strata,
        min_seq_len=min_seq_len,
        test_ratio=test_ratio,
        folds=folds,
    )
    source_dir = Path(source_dir)
    output_dir = Path(output_dir)
    _validate_source_files(source_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a fresh path"
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    before_snapshot = _source_snapshot(source_dir)
    source_hashes = {
        name: _file_sha256(source_dir / name) for name in HASHED_SOURCE_FILES
    }
    profiles = _read_user_profiles(source_dir)
    selected, population = _sample_users(
        profiles,
        students=students,
        strata=strata,
        min_seq_len=min_seq_len,
        seed=seed,
    )
    split_summary = _assign_splits_and_folds(
        selected, test_ratio=test_ratio, folds=folds, seed=seed
    )
    uid_inverse, question_to_area, metadata_summary = _load_metadata(source_dir)

    staging_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        collector = StatisticsCollector(question_to_area)
        view_summaries: dict[str, dict[str, Any]] = {}
        for spec in VIEW_SPECS:
            view_summaries[spec.name] = _filter_view(
                source_dir, staging_dir, spec, selected, collector
            )

        for name in METADATA_FILES:
            shutil.copy2(source_dir / name, staging_dir / name)
        selected_rows = _write_selected_users(
            staging_dir / "selected_uids.json", selected, uid_inverse
        )
        statistics_data = collector.finalize()
        _write_statistics_csv(staging_dir / "subset_stats.csv", statistics_data)

        after_snapshot = _source_snapshot(source_dir)
        if before_snapshot != after_snapshot:
            raise RuntimeError("Junyi source files changed while the subset was built")

        source_files: dict[str, dict[str, Any]] = {}
        for name, file_info in before_snapshot.items():
            source_files[name] = {"path": name, **file_info}
            if name in source_hashes:
                source_files[name]["sha256"] = source_hashes[name]

        manifest: dict[str, Any] = {
            "schema_version": 1,
            "dataset_name": DATASET_NAME,
            "display_name": DISPLAY_NAME,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "selection": {
                "algorithm": "equal-rank-strata-random-sample",
                "algorithm_version": ALGORITHM_VERSION,
                "students": students,
                "strata_count": strata,
                "students_per_stratum": students // strata,
                "min_sequence_length": min_seq_len,
                "test_ratio": test_ratio,
                "folds": folds,
                "seed": seed,
                "split_seed": seed + 1,
                "fold_seed": seed + 2,
                **population,
            },
            "splits": split_summary,
            "statistics": statistics_data,
            "metadata": {
                **metadata_summary,
                "selected_users_with_source_uid": sum(
                    row["source_uid"] is not None for row in selected_rows
                ),
                "selected_question_ids_without_area": sorted(
                    collector.unmapped_question_ids,
                    key=_uid_sort_key,
                ),
            },
            "source_files": source_files,
            "outputs": view_summaries,
            "integrity": {
                "source_unchanged": True,
                "all_views_cover_selected_users": True,
                "train_test_user_overlap": 0,
            },
        }
        with (staging_dir / "subset_manifest.json").open(
            "w", encoding="utf-8", newline="\n"
        ) as fout:
            json.dump(manifest, fout, ensure_ascii=False, indent=2)
            fout.write("\n")

        _publish_directory(staging_dir, output_dir)
        print(f"[done] published subset at {output_dir}", flush=True)
        return manifest
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/junyi2015"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/junyi_sub5k"))
    parser.add_argument("--students", type=int, default=5000)
    parser.add_argument("--strata", type=int, default=5)
    parser.add_argument("--min-seq-len", type=int, default=3)
    parser.add_argument("--test-ratio", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=3407)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    manifest = build_subset(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        students=args.students,
        strata=args.strata,
        min_seq_len=args.min_seq_len,
        test_ratio=args.test_ratio,
        folds=args.folds,
        seed=args.seed,
    )
    overall = manifest["statistics"]["overall"]
    print(
        json.dumps(
            {
                "students": overall["students"],
                "questions": overall["questions"],
                "interactions": overall["interactions"],
                "knowledge_areas": overall["knowledge_areas"],
                "knowledge_concepts": overall["knowledge_concepts"],
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
