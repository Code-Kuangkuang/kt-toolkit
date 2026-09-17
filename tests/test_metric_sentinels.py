"""An unavailable metric must be None, never a number.

`_score_loader` used to return -1 whenever AUC could not be computed. -1 is a
number, so it flowed into best_metrics.json and was then averaged by
aggregate_fold_metrics like a real score: one unscorable fold out of five drags
the reported mean down by roughly 0.35 while the fold count still reads 5/5.

It also collapsed two very different situations into the same value -- a split
that happens to be single-class, which is benign, and NaN predictions or
mismatched shapes, which are model bugs that must not be swallowed.
"""

import sys
import unittest
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cv_results import aggregate_fold_metrics
from core.hooks import BestMetricsHook, SaveBestHook
from core.trainer import BaseTrainer


class FakeTrainer(BaseTrainer):
    """Scores a fixed list of (pred, target) pairs, bypassing any model."""

    def __init__(self, batches):
        super().__init__(num_epochs=1)
        self._batches = batches
        self.model = torch.nn.Linear(1, 1)  # only needs .eval()

    def _forward_batch(self, batch):
        return batch

    def score(self, prefix="test"):
        return self._score_loader(list(range(len(self._batches))), prefix=prefix)

    def _score_loader(self, loader, prefix):
        # Feed the canned batches through the real implementation.
        return BaseTrainer._score_loader(self, self._batches, prefix)


def batch(preds, targets):
    return (torch.tensor(preds, dtype=torch.float32),
            torch.tensor(targets, dtype=torch.float32))


class ScoreLoaderTest(unittest.TestCase):
    def test_normal_split_scores(self):
        got = FakeTrainer([batch([0.1, 0.9, 0.3, 0.8], [0, 1, 0, 1])]).score()
        self.assertAlmostEqual(got["test_auc"], 1.0)
        self.assertAlmostEqual(got["test_acc"], 1.0)

    def test_single_class_split_returns_none_not_minus_one(self):
        got = FakeTrainer([batch([0.1, 0.9, 0.3], [1, 1, 1])]).score()
        self.assertIsNone(got["test_auc"])
        # Accuracy is still well defined with one class present.
        self.assertIsNotNone(got["test_acc"])

    def test_absent_loader_returns_none(self):
        got = BaseTrainer._score_loader(FakeTrainer([]), None, "test")
        self.assertIsNone(got["test_auc"])
        self.assertIsNone(got["test_acc"])

    def test_nan_predictions_raise_rather_than_scoring(self):
        trainer = FakeTrainer([batch([0.1, float("nan"), 0.3, 0.8], [0, 1, 0, 1])])
        with self.assertRaises(ValueError) as err:
            trainer.score()
        self.assertIn("non-finite", str(err.exception))

    def test_inf_predictions_raise(self):
        trainer = FakeTrainer([batch([0.1, float("inf"), 0.3, 0.8], [0, 1, 0, 1])])
        with self.assertRaises(ValueError):
            trainer.score()

    def test_shape_mismatch_raises(self):
        trainer = FakeTrainer([(torch.tensor([0.1, 0.9, 0.5]), torch.tensor([0.0, 1.0]))])
        with self.assertRaises(ValueError) as err:
            trainer.score()
        self.assertIn("smasks", str(err.exception))


class AggregationOfNoneTest(unittest.TestCase):
    def test_none_counts_as_missing_rather_than_as_a_score(self):
        """The whole point: None shortens n, -1 would have moved the mean."""
        folds = [
            {"fold": i, "best_metrics": {"best_test_auc": 0.78 if i != 2 else None}}
            for i in range(5)
        ]
        entry = aggregate_fold_metrics(folds)["best_test_auc"]
        self.assertEqual(entry["n"], 4)
        self.assertEqual(entry["n_folds"], 5)
        self.assertEqual(entry["missing_folds"], [2])
        self.assertAlmostEqual(entry["mean"], 0.78)

    def test_the_old_sentinel_would_have_gone_unnoticed(self):
        """Documents the bug this change removes, so it cannot come back."""
        folds = [
            {"fold": i, "best_metrics": {"best_test_auc": 0.78 if i != 2 else -1}}
            for i in range(5)
        ]
        entry = aggregate_fold_metrics(folds)["best_test_auc"]
        self.assertEqual(entry["n"], 5)          # looks complete
        self.assertLess(entry["mean"], 0.47)     # but the mean is destroyed


class HooksIgnoreNoneTest(unittest.TestCase):
    def test_save_best_hook_skips_a_none_metric(self):
        hook = SaveBestHook(save_dir="/nonexistent", metric_key="valid_auc")
        hook.on_epoch_end(FakeTrainer([]), {"valid_auc": None})
        self.assertIsNone(hook.best)  # no checkpoint attempted

    def test_best_metrics_hook_skips_a_none_metric(self):
        hook = BestMetricsHook(metric_key="valid_auc")
        hook.on_epoch_end(FakeTrainer([]), {"valid_auc": None})
        self.assertIsNone(hook.best_value)


if __name__ == "__main__":
    unittest.main()
