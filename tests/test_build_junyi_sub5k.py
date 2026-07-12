import csv
import json
import tempfile
import unittest
from pathlib import Path

from scripts.build_junyi_sub5k import build_subset


RAW_SCHEMAS = {
    "train_valid_quelevel.csv": [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
    ],
    "test_quelevel.csv": [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
    ],
    "train_valid.csv": [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
        "is_repeat",
    ],
    "test.csv": [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
        "is_repeat",
    ],
}

SEQUENCE_SCHEMAS = {
    "train_valid_sequences_quelevel.csv": [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
        "selectmasks",
    ],
    "test_sequences_quelevel.csv": [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
        "selectmasks",
    ],
    "train_valid_sequences.csv": [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
        "selectmasks",
        "is_repeat",
    ],
    "test_sequences.csv": [
        "fold",
        "uid",
        "questions",
        "concepts",
        "responses",
        "timestamps",
        "usetimes",
        "selectmasks",
        "is_repeat",
    ],
}


def _sequence_values(uid, length):
    questions = [str((uid + index) % 7) for index in range(length)]
    concepts = [str((uid + index) % 5) for index in range(length)]
    responses = [str((uid + index) % 2) for index in range(length)]
    timestamps = [str(1_000_000 + uid * 100 + index) for index in range(length)]
    usetimes = [str(10 + index) for index in range(length)]
    return questions, concepts, responses, timestamps, usetimes


def _row(uid, length, *, sequence=False, start=0, stop=None, concept_level=False):
    stop = length if stop is None else stop
    values = _sequence_values(uid, length)
    questions, concepts, responses, timestamps, usetimes = [
        items[start:stop] for items in values
    ]
    row = {
        "fold": str(uid % 5) if uid % 4 else "-1",
        "uid": str(uid),
        "questions": ",".join(questions),
        "concepts": ",".join(concepts),
        "responses": ",".join(responses),
        "timestamps": ",".join(timestamps),
        "usetimes": ",".join(usetimes),
    }
    if sequence:
        row["selectmasks"] = ",".join("1" for _ in questions)
    if concept_level:
        row["is_repeat"] = ",".join("0" for _ in questions)
    return row


def _write_rows(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as fout:
        writer = csv.DictWriter(fout, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_synthetic_source(source_dir):
    source_dir.mkdir(parents=True)
    raw_rows = {name: [] for name in RAW_SCHEMAS}
    sequence_rows = {name: [] for name in SEQUENCE_SCHEMAS}

    for uid in range(50):
        length = uid + 3
        source_prefix = "test" if uid % 4 == 0 else "train_valid"
        for suffix, concept_level in (("_quelevel.csv", False), (".csv", True)):
            name = f"{source_prefix}{suffix}"
            raw_rows[name].append(
                _row(uid, length, concept_level=concept_level)
            )

        for start in range(0, length, 8):
            stop = min(start + 8, length)
            for suffix, concept_level in (
                ("_sequences_quelevel.csv", False),
                ("_sequences.csv", True),
            ):
                name = f"{source_prefix}{suffix}"
                sequence_rows[name].append(
                    _row(
                        uid,
                        length,
                        sequence=True,
                        start=start,
                        stop=stop,
                        concept_level=concept_level,
                    )
                )

    for name, fieldnames in RAW_SCHEMAS.items():
        _write_rows(source_dir / name, fieldnames, raw_rows[name])
    for name, fieldnames in SEQUENCE_SCHEMAS.items():
        _write_rows(source_dir / name, fieldnames, sequence_rows[name])

    mapping = {
        "questions": {f"exercise####{qid}": qid for qid in range(7)},
        "concepts": {f"topic-{cid}": cid for cid in range(5)},
        "uid": {str(uid): uid for uid in range(50)},
    }
    (source_dir / "keyid2idx.json").write_text(
        json.dumps(mapping, ensure_ascii=False), encoding="utf-8"
    )
    with (source_dir / "junyi_Exercise_table.csv").open(
        "w", encoding="utf-8", newline=""
    ) as fout:
        writer = csv.DictWriter(fout, fieldnames=["name", "topic", "area"])
        writer.writeheader()
        for qid in range(7):
            writer.writerow(
                {
                    "name": f"exercise_{qid}",
                    "topic": f"topic-{qid % 5}",
                    "area": f"area-{qid % 3}",
                }
            )


def _read_rows(path):
    with path.open("r", encoding="utf-8", newline="") as fin:
        return list(csv.DictReader(fin))


class JunyiSubsetBuilderTest(unittest.TestCase):
    def test_builds_deterministic_leakage_free_multi_view_subset(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_dir = root / "source"
            output_a = root / "subset-a"
            output_b = root / "subset-b"
            _write_synthetic_source(source_dir)

            kwargs = {
                "source_dir": source_dir,
                "students": 25,
                "strata": 5,
                "min_seq_len": 3,
                "test_ratio": 0.2,
                "folds": 5,
                "seed": 3407,
            }
            manifest_a = build_subset(output_dir=output_a, **kwargs)
            manifest_b = build_subset(output_dir=output_b, **kwargs)

            selected_a = json.loads(
                (output_a / "selected_uids.json").read_text(encoding="utf-8")
            )
            selected_b = json.loads(
                (output_b / "selected_uids.json").read_text(encoding="utf-8")
            )
            self.assertEqual(selected_a, selected_b)
            self.assertEqual(len(selected_a), 25)
            self.assertEqual(
                {stratum: sum(row["stratum"] == stratum for row in selected_a)
                 for stratum in range(5)},
                {stratum: 5 for stratum in range(5)},
            )

            train_uids = {
                row["uid"] for row in selected_a if row["split"] == "train_valid"
            }
            test_uids = {row["uid"] for row in selected_a if row["split"] == "test"}
            self.assertEqual(len(train_uids), 20)
            self.assertEqual(len(test_uids), 5)
            self.assertTrue(train_uids.isdisjoint(test_uids))
            self.assertEqual(
                {fold: sum(row.get("fold") == fold for row in selected_a)
                 for fold in range(5)},
                {fold: 4 for fold in range(5)},
            )

            required = set(RAW_SCHEMAS) | set(SEQUENCE_SCHEMAS) | {
                "keyid2idx.json",
                "junyi_Exercise_table.csv",
                "selected_uids.json",
                "subset_stats.csv",
                "subset_manifest.json",
            }
            self.assertTrue(all((output_a / name).is_file() for name in required))

            for prefix, expected_uids, expected_folds in (
                ("train_valid", train_uids, set(range(5))),
                ("test", test_uids, {-1}),
            ):
                for suffix in (
                    ".csv",
                    "_quelevel.csv",
                    "_sequences.csv",
                    "_sequences_quelevel.csv",
                ):
                    rows = _read_rows(output_a / f"{prefix}{suffix}")
                    self.assertEqual({int(row["uid"]) for row in rows}, expected_uids)
                    self.assertEqual({int(row["fold"]) for row in rows}, expected_folds)

            overall = manifest_a["statistics"]["overall"]
            self.assertEqual(overall["students"], 25)
            self.assertEqual(overall["questions"], 7)
            self.assertEqual(overall["knowledge_concepts"], 5)
            self.assertEqual(overall["knowledge_areas"], 3)
            self.assertEqual(
                overall["interactions"],
                sum(row["sequence_length"] for row in selected_a),
            )
            self.assertEqual(manifest_a["selection"]["seed"], 3407)
            self.assertEqual(manifest_a["selection"]["algorithm_version"], 1)


if __name__ == "__main__":
    unittest.main()
