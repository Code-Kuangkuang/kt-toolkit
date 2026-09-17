"""CV aggregation must not present a partial mean as a full one.

`train_one_fold` swallows a test-loader failure so that one bad fold does not
kill a five-fold sweep. The cost is that the fold finishes without
`best_test_auc`, and the aggregate then averages whatever folds did report it.
Without a count, a four-fold mean is indistinguishable from a five-fold one --
and because the std is computed over the same short list, it also looks tighter.

These tests pin the count, the missing-fold ids, and the fact that the
pyKT-comparable window metrics reach cv_summary.csv.
"""

import csv
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.cv_results import aggregate_fold_metrics, save_cv_summary


def fold(fid, **metrics):
    return {
        "fold": fid,
        "run_name": f"run{fid}",
        "ckpt_dir": f"/ckpt/{fid}",
        "best_path": f"/ckpt/{fid}/best.pt",
        "best_metrics": metrics,
    }


class AggregateFoldMetricsTest(unittest.TestCase):
    def test_complete_run_reports_full_count(self):
        agg = aggregate_fold_metrics([fold(i, valid_auc=0.7 + i / 100) for i in range(5)])
        entry = agg["valid_auc"]
        self.assertEqual(entry["n"], 5)
        self.assertEqual(entry["n_folds"], 5)
        self.assertEqual(entry["missing_folds"], [])

    def test_missing_fold_is_counted_and_named(self):
        folds = [fold(i, valid_auc=0.8, best_test_auc=0.75) for i in range(5)]
        del folds[2]["best_metrics"]["best_test_auc"]  # test loader failed here

        agg = aggregate_fold_metrics(folds)
        self.assertEqual(agg["valid_auc"]["n"], 5)

        partial = agg["best_test_auc"]
        self.assertEqual(partial["n"], 4)
        self.assertEqual(partial["n_folds"], 5)
        self.assertEqual(partial["missing_folds"], [2])
        # The mean itself is still over the folds that reported.
        self.assertAlmostEqual(partial["mean"], 0.75)
        self.assertEqual(len(partial["values"]), 4)

    def test_fold_with_no_metrics_at_all_is_counted_as_missing(self):
        folds = [fold(i, valid_auc=0.8) for i in range(3)]
        folds[1]["best_metrics"] = None

        agg = aggregate_fold_metrics(folds)
        self.assertEqual(agg["valid_auc"]["n"], 2)
        self.assertEqual(agg["valid_auc"]["missing_folds"], [1])

    def test_non_numeric_values_do_not_become_metrics(self):
        agg = aggregate_fold_metrics([fold(0, valid_auc=0.8, note="ok")])
        self.assertIn("valid_auc", agg)
        self.assertNotIn("note", agg)

    def test_empty_run(self):
        self.assertEqual(aggregate_fold_metrics([]), {})


class SaveCvSummaryTest(unittest.TestCase):
    def test_csv_carries_the_window_metrics(self):
        folds = [
            fold(
                i,
                valid_auc=0.80,
                best_test_auc=0.75,
                best_window_test_auc=0.7541,
                best_window_test_acc=0.71,
                epoch=12,
            )
            for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            save_cv_summary(tmp, {"aggregate": aggregate_fold_metrics(folds)}, folds)
            with open(Path(tmp) / "cv_summary.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))

        self.assertIn("best_window_test_auc", rows[0])
        self.assertEqual(rows[0]["best_window_test_auc"], "0.7541")
        self.assertEqual(rows[0]["best_window_test_acc"], "0.71")

    def test_missing_metrics_leave_blank_cells_rather_than_failing(self):
        folds = [fold(0, valid_auc=0.8)]
        with tempfile.TemporaryDirectory() as tmp:
            save_cv_summary(tmp, {"aggregate": aggregate_fold_metrics(folds)}, folds)
            with open(Path(tmp) / "cv_summary.csv", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        self.assertEqual(rows[0]["best_window_test_auc"], "")


if __name__ == "__main__":
    unittest.main()
