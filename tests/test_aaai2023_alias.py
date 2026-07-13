import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AAAI2023DatasetAliasTest(unittest.TestCase):
    def test_peiyou_normalizes_to_aaai2023(self):
        from core.dataset_names import normalize_dataset_name

        self.assertEqual(normalize_dataset_name("peiyou"), "aaai2023")
        self.assertEqual(normalize_dataset_name("AAAI2023"), "aaai2023")
        self.assertEqual(normalize_dataset_name(" assist2017 "), "assist2017")

    def test_both_names_are_hidden_label_datasets(self):
        from core.dataset_names import is_hidden_label_dataset

        self.assertTrue(is_hidden_label_dataset("peiyou"))
        self.assertTrue(is_hidden_label_dataset("aaai2023"))
        self.assertFalse(is_hidden_label_dataset("assist2017"))

    def test_both_config_names_share_canonical_path(self):
        config = json.loads(
            (ROOT / "configs" / "data_config.json").read_text(encoding="utf-8")
        )

        self.assertEqual(config["aaai2023"]["dpath"], "data/aaai2023")
        self.assertEqual(config["peiyou"], config["aaai2023"])
        self.assertTrue((ROOT / config["aaai2023"]["dpath"]).is_dir())

    def test_prediction_entry_points_share_one_app(self):
        from scripts import predict_aaai2023, predict_peiyou

        self.assertIs(predict_peiyou.app, predict_aaai2023.app)


if __name__ == "__main__":
    unittest.main()
