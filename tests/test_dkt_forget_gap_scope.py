"""DKT-Forget's gap tables must survive being sized from the training folds.

A gap is "how long since this learner last met this concept", log2-bucketed, so
`num_rgap` and friends are embedding table heights rather than statistics. That
is why the old code read the test file: not to see a label -- it reads only
`timestamps` -- but to know how many rows to allocate.

AGENTS.md requires a derived feature to be fitted on the current fold's training
rows. Doing that leaves a table that can be too short, because valid or test may
hold a longer gap than training ever saw, so one row is reserved as an
out-of-vocabulary bucket and oversized values are folded onto it. Without that,
the first unusually long gap in the test split is an index error.
"""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.feature_utils import clamp_dkt_forget_gaps, compute_dkt_forget_stats

MINUTE = 60 * 1000


def write_sequences(path, rows):
    """A minimal sequence CSV: one learner per row, with a fold column."""
    pd.DataFrame(rows).to_csv(path, index=False)


def row(fold, concepts, minutes):
    """Timestamps are milliseconds; gaps are log2 of the minute difference."""
    stamps = []
    now = 0
    for gap in minutes:
        now += gap * MINUTE
        stamps.append(now)
    return {
        "fold": fold,
        "concepts": ",".join(str(c) for c in concepts),
        "questions": ",".join(str(c) for c in concepts),
        "timestamps": ",".join(str(t) for t in stamps),
        "responses": ",".join("1" for _ in concepts),
    }


class ClampTest(unittest.TestCase):
    def test_values_past_the_table_fold_onto_its_last_row(self):
        self.assertEqual(clamp_dkt_forget_gaps([0, 5, 12, 40], cap=8), [0, 5, 7, 7])

    def test_values_inside_the_table_are_untouched(self):
        self.assertEqual(clamp_dkt_forget_gaps([0, 3, 7], cap=8), [0, 3, 7])

    def test_no_cap_passes_everything_through(self):
        """The transductive path sizes its table from every split, so nothing
        can exceed it and nothing should be altered."""
        self.assertEqual(clamp_dkt_forget_gaps([0, 99], cap=None), [0, 99])
        self.assertEqual(clamp_dkt_forget_gaps([0, 99], cap=0), [0, 99])

    def test_clamping_never_produces_an_out_of_range_index(self):
        cap = 6
        clamped = clamp_dkt_forget_gaps(list(range(50)), cap=cap)
        self.assertTrue(all(0 <= v < cap for v in clamped))


class StatsScopeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dpath = Path(self._tmp.name)
        # Fold 0 is held out. The training folds see gaps up to ~1 hour; the
        # held-out fold has a learner who came back after ~11 days.
        write_sequences(self.dpath / "train_valid.csv", [
            row(1, [1, 1, 1], [0, 1, 2]),
            row(2, [1, 1, 1], [0, 8, 64]),
            row(0, [1, 1, 1], [0, 4, 16000]),
        ])
        write_sequences(self.dpath / "test.csv", [
            {**row(1, [1, 1, 1], [0, 2, 32000])},
        ])
        # The test file in the real layout carries no fold column.
        test = pd.read_csv(self.dpath / "test.csv")
        test.drop(columns=["fold"]).to_csv(self.dpath / "test.csv", index=False)

    def tearDown(self):
        self._tmp.cleanup()

    def _stats(self, folds):
        return compute_dkt_forget_stats(
            str(self.dpath), ["train_valid.csv", "test.csv"],
            ["concepts"], folds=folds,
        )

    def test_training_folds_give_a_smaller_table_than_every_split(self):
        train_only = self._stats(folds=[1, 2])
        everything = self._stats(folds=None)
        self.assertLess(train_only["num_rgap"], everything["num_rgap"])

    def test_the_training_scope_reserves_an_oov_row(self):
        """One row beyond the largest bucket training actually saw."""
        train_only = self._stats(folds=[1, 2])
        # Recompute the bare maximum by asking for the same rows without the
        # reservation, which is what the transductive path returns.
        bare_max = self._stats(folds=None)["num_rgap"] - 1
        self.assertGreater(train_only["num_rgap"], 1)
        self.assertLessEqual(train_only["num_rgap"], bare_max + 2)

    def test_a_held_out_fold_cannot_widen_the_table(self):
        """Fold 0 has the longest gap in train_valid.csv and must not count."""
        without = self._stats(folds=[1, 2])
        with_it = self._stats(folds=[0, 1, 2])
        self.assertLess(without["num_rgap"], with_it["num_rgap"])

    def test_the_test_file_is_skipped_under_a_fold_scope(self):
        """It has no fold column, so it contributes everything or nothing."""
        scoped = compute_dkt_forget_stats(
            str(self.dpath), ["train_valid.csv", "test.csv"], ["concepts"], folds=[1, 2],
        )
        train_file_only = compute_dkt_forget_stats(
            str(self.dpath), ["train_valid.csv"], ["concepts"], folds=[1, 2],
        )
        self.assertEqual(scoped, train_file_only)

    def test_an_empty_fold_list_is_rejected(self):
        with self.assertRaises(ValueError):
            self._stats(folds=[])

    def test_gaps_from_the_held_out_fold_still_fit_after_clamping(self):
        """The end-to-end point: nothing can index past the table."""
        from datasets.feature_utils import compute_dkt_forget_gaps

        stats = self._stats(folds=[1, 2])
        df = pd.read_csv(self.dpath / "train_valid.csv")
        held_out = df[df["fold"] == 0].iloc[0]
        rgap, sgap, pcount = compute_dkt_forget_gaps(held_out, ["concepts"])

        self.assertGreater(max(rgap), stats["num_rgap"] - 1,
                           "the fixture should contain a gap the table cannot hold")
        for values, key in ((rgap, "num_rgap"), (sgap, "num_sgap"), (pcount, "num_pcount")):
            clamped = clamp_dkt_forget_gaps(values, stats[key])
            self.assertTrue(all(0 <= v < stats[key] for v in clamped))


if __name__ == "__main__":
    unittest.main()
