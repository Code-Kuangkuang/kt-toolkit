"""Resuming a cross-validation run must not mix protocols or seeds.

`--skip-completed 1` reuses a finished fold from the CV directory. It matched on
dataset, model, fold and the label-flip settings and stopped there, so changing
anything else -- `score_repeated_kc`, `concept_mode`, `pykt_transductive`, the
seed, or regenerating the data -- and resuming would reuse the old folds, run the
remaining ones under the new settings, and average the two together into a single
reported mean.

That is the protocol mixing `protocol_stamp` exists to prevent, happening in the
one place that combines folds automatically, and announcing itself only as
"completed run found".
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from datasets.init_dataset import protocol_stamp

import importlib.util

_spec = importlib.util.spec_from_file_location("kt_train_cli", ROOT / "scripts" / "train.py")
train_cli = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_cli)


def base_protocol(**overrides):
    stamp = protocol_stamp("dkt", "all_in_one", 4)
    stamp.update(overrides)
    return stamp


def fold_result(fold, protocol=None, **extra):
    result = {"fold": fold, "best_metrics": {"valid_auc": 0.8}}
    if protocol is not None:
        result["protocol"] = protocol
    result.update(extra)
    return result


def write_run(cv_dir, name, **config):
    run_dir = Path(cv_dir) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run_config.json").write_text(json.dumps(config), encoding="utf-8")
    (run_dir / "best_metrics.json").write_text(
        json.dumps({"valid_auc": 0.8}), encoding="utf-8"
    )
    return run_dir


class AssertOneProtocolTest(unittest.TestCase):
    def test_matching_folds_pass(self):
        train_cli._assert_one_protocol([
            fold_result(0, base_protocol()),
            fold_result(1, base_protocol()),
        ])

    def test_a_differing_scope_stops_the_average(self):
        with self.assertRaises(SystemExit) as err:
            train_cli._assert_one_protocol([
                fold_result(0, base_protocol(feature_fit_scope="train_folds")),
                fold_result(1, base_protocol(feature_fit_scope="train_valid_test")),
            ])
        message = str(err.exception)
        self.assertIn("feature_fit_scope", message)
        self.assertIn("different protocols", message)

    def test_a_differing_repeat_setting_stops_the_average(self):
        """The leak the whole mechanism was built for."""
        with self.assertRaises(SystemExit) as err:
            train_cli._assert_one_protocol([
                fold_result(0, base_protocol(score_repeated_kc=False)),
                fold_result(1, base_protocol(score_repeated_kc=True)),
            ])
        self.assertIn("score_repeated_kc", str(err.exception))

    def test_a_differing_concept_mode_stops_the_average(self):
        with self.assertRaises(SystemExit):
            train_cli._assert_one_protocol([
                fold_result(0, base_protocol(concept_mode="multi", concepts_visible="all")),
                fold_result(1, base_protocol(concept_mode="first", concepts_visible="first_of_4")),
            ])

    def test_the_message_names_both_folds_and_both_values(self):
        with self.assertRaises(SystemExit) as err:
            train_cli._assert_one_protocol([
                fold_result(0, base_protocol(graph_scope="train_folds")),
                fold_result(3, base_protocol(graph_scope="train_valid_test")),
            ])
        message = str(err.exception)
        self.assertIn("fold 0", message)
        self.assertIn("fold 3", message)
        self.assertIn("train_valid_test", message)

    def test_a_single_fold_needs_no_agreement(self):
        train_cli._assert_one_protocol([fold_result(0, base_protocol())])

    def test_unstamped_results_are_ignored_rather_than_compared(self):
        """Nothing to compare is not the same as a mismatch."""
        train_cli._assert_one_protocol([
            fold_result(0),
            fold_result(1, base_protocol()),
        ])


class LoadCompletedFoldTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.cv_dir = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def _load(self, seed=3407):
        return train_cli._load_completed_fold(
            self.cv_dir, 0, "assist2009", "dkt",
            train_label_flip_ratio=0.0, train_label_flip_seed=3407, seed=seed,
        )

    def _write(self, **overrides):
        config = {
            "fold": 0, "dataset_name": "assist2009", "model_name": "dkt",
            "seed": 3407, "protocol": base_protocol(),
        }
        config.update(overrides)
        write_run(self.cv_dir, "run0", **config)

    def test_a_matching_run_is_reused(self):
        self._write()
        result = self._load()
        self.assertIsNotNone(result)
        self.assertTrue(result["skipped"])
        self.assertEqual(result["protocol"], base_protocol())

    def test_a_different_seed_is_not_reused(self):
        """The seed produced the number, so another seed is another experiment."""
        self._write(seed=1234)
        self.assertIsNone(self._load(seed=3407))

    def test_a_run_without_a_protocol_block_is_not_reused(self):
        """Predates the stamp, so it cannot be shown to match anything."""
        self._write(protocol=None)
        self.assertIsNone(self._load())

    def test_a_different_model_is_not_reused(self):
        self._write(model_name="akt")
        self.assertIsNone(self._load())

    def test_the_reused_result_carries_its_protocol_for_the_later_check(self):
        """Without this the cross-fold comparison has nothing to compare."""
        self._write(protocol=base_protocol(feature_fit_scope="train_valid_test"))
        result = self._load()
        self.assertEqual(result["protocol"]["feature_fit_scope"], "train_valid_test")


if __name__ == "__main__":
    unittest.main()
