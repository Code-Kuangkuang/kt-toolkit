import csv
import json
import tempfile
import unittest
from pathlib import Path

import yaml

from cleaning.adapters import ADAPTERS


ROOT = Path(__file__).resolve().parents[1]


class SlepemapyIntegrationTest(unittest.TestCase):
    def test_repository_config_and_adapter_registration(self):
        self.assertIn("slepemapy", ADAPTERS)

        yaml_path = ROOT / "configs" / "dataset" / "slepemapy.yaml"
        with yaml_path.open("r", encoding="utf-8") as fin:
            dataset_yaml = yaml.safe_load(fin)
        self.assertEqual(dataset_yaml["dataset_name"], "slepemapy")
        self.assertEqual(dataset_yaml["raw_path"], "data/slepemapy/answer.csv")
        self.assertEqual(dataset_yaml["dpath"], "data/slepemapy")
        self.assertTrue(dataset_yaml["gen_question_level"])

        with (ROOT / "configs" / "data_config.json").open("r", encoding="utf-8") as fin:
            dataset_cfg = json.load(fin)["slepemapy"]
        expected = {
            "dpath": "data/slepemapy",
            "num_q": 2913,
            "num_c": 1458,
            "max_concepts": 1,
            "train_valid_file_quelevel": "train_valid_sequences_quelevel.csv",
            "test_file_quelevel": "test_sequences_quelevel.csv",
        }
        for key, value in expected.items():
            self.assertEqual(dataset_cfg[key], value)

    def test_adapter_generates_aligned_question_level_artifacts(self):
        self.assertIn("slepemapy", ADAPTERS)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            raw_path = tmp_path / "answer.csv"
            config_path = tmp_path / "data_config.json"
            config_path.write_text("{}", encoding="utf-8")

            fieldnames = [
                "id",
                "user",
                "place_asked",
                "place_answered",
                "type",
                "inserted",
                "response_time",
            ]
            rows = []
            answer_id = 1
            for offset in range(10):
                user = 1000 + offset
                rows.extend(
                    [
                        {
                            "id": answer_id,
                            "user": user,
                            "place_asked": 2,
                            "place_answered": 1,
                            "type": 2,
                            "inserted": "2015-01-01 00:00:02",
                            "response_time": 2000,
                        },
                        {
                            "id": answer_id + 1,
                            "user": user,
                            "place_asked": 1,
                            "place_answered": 1,
                            "type": 1,
                            "inserted": "2015-01-01 00:00:01",
                            "response_time": 1000,
                        },
                        {
                            "id": answer_id + 2,
                            "user": user,
                            "place_asked": 3,
                            "place_answered": "",
                            "type": 1,
                            "inserted": "2015-01-01 00:00:03",
                            "response_time": 3000,
                        },
                    ]
                )
                answer_id += 3

            with raw_path.open("w", encoding="utf-8", newline="") as fout:
                writer = csv.DictWriter(fout, fieldnames=fieldnames, delimiter=";")
                writer.writeheader()
                writer.writerows(rows)

            cfg = {
                "dataset_name": "slepemapy",
                "raw_path": raw_path.as_posix(),
                "dpath": tmp_path.as_posix(),
                "configf": config_path.as_posix(),
                "min_seq_len": 3,
                "maxlen": 8,
                "kfold": 5,
                "gen_question_level": True,
            }
            ADAPTERS["slepemapy"](cfg)

            data_lines = (tmp_path / "data.txt").read_text(encoding="utf-8").splitlines()
            self.assertEqual(data_lines[0], "1000,3")
            self.assertEqual(data_lines[1], "1----1,2----2,3----1")
            self.assertEqual(data_lines[2], "1,2,3")
            self.assertEqual(data_lines[3], "1,0,0")

            core_outputs = {
                "train_valid.csv",
                "train_valid_sequences.csv",
                "test.csv",
                "test_sequences.csv",
                "train_valid_quelevel.csv",
                "train_valid_sequences_quelevel.csv",
                "test_quelevel.csv",
                "test_sequences_quelevel.csv",
                "keyid2idx.json",
            }
            self.assertTrue(all((tmp_path / name).is_file() for name in core_outputs))

            with (tmp_path / "train_valid_quelevel.csv").open(
                "r", encoding="utf-8", newline=""
            ) as fin:
                train_rows = list(csv.DictReader(fin))
            with (tmp_path / "test_quelevel.csv").open(
                "r", encoding="utf-8", newline=""
            ) as fin:
                test_rows = list(csv.DictReader(fin))

            train_users = {row["uid"] for row in train_rows}
            test_users = {row["uid"] for row in test_rows}
            self.assertTrue(train_users.isdisjoint(test_users))
            self.assertEqual({int(row["fold"]) for row in train_rows}, set(range(5)))
            self.assertEqual({int(row["fold"]) for row in test_rows}, {-1})

            for row in train_rows + test_rows:
                lengths = {
                    len(row["questions"].split(",")),
                    len(row["concepts"].split(",")),
                    len(row["responses"].split(",")),
                }
                self.assertEqual(lengths, {3})
                self.assertLessEqual(set(row["responses"].split(",")), {"0", "1"})

            generated_cfg = json.loads(config_path.read_text(encoding="utf-8"))[
                "slepemapy"
            ]
            self.assertEqual(generated_cfg["num_q"], 3)
            self.assertEqual(generated_cfg["num_c"], 3)
            self.assertEqual(generated_cfg["max_concepts"], 1)
            self.assertEqual(
                generated_cfg["train_valid_file_quelevel"],
                "train_valid_sequences_quelevel.csv",
            )
            self.assertEqual(
                generated_cfg["test_file_quelevel"],
                "test_sequences_quelevel.csv",
            )


if __name__ == "__main__":
    unittest.main()
