"""A misaligned supervision mask must stop the run, not be guessed around.

`_resolve_selectmasks` used to fall back silently twice. The `is_repeat` half was
the dangerous one: a row whose flags did not line up with its responses kept its
repeated-KC positions scored, and returned `dropped=0`, so `_report_repeat_filter`
stayed quiet too. That is the leak the function exists to close, coming back
without a word on exactly the malformed input where guessing is least safe.

One-by-one mode expands a multi-KC question into consecutive rows carrying the
same response. Scoring those rows asks the model to predict a label its own input
history already contains -- 15.6% of scored positions on assist2009, which is why
concept-level AUC used to read far above the question-level number.
"""

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.kt_dataset import _resolve_selectmasks


def row(**fields):
    return pd.Series(fields)


class AlignedInputTest(unittest.TestCase):
    def test_repeat_positions_are_dropped_from_supervision(self):
        mask, dropped = _resolve_selectmasks(
            row(selectmasks="1,1,1", is_repeat="0,1,0"), [1, 0, 1]
        )
        self.assertEqual(mask, [1, -1, 1])
        self.assertEqual(dropped, 1)

    def test_an_already_unscored_position_is_not_counted_twice(self):
        """`dropped` measures supervision actually removed."""
        mask, dropped = _resolve_selectmasks(
            row(selectmasks="1,-1,1", is_repeat="0,1,0"), [1, -1, 1]
        )
        self.assertEqual(mask, [1, -1, 1])
        self.assertEqual(dropped, 0)

    def test_a_missing_selectmasks_column_derives_from_responses(self):
        mask, dropped = _resolve_selectmasks(row(), [1, -1, 0])
        self.assertEqual(mask, [1, -1, 1])
        self.assertEqual(dropped, 0)

    def test_question_level_rows_without_is_repeat_pass_through(self):
        """Quelevel files have one row per question, so nothing is repeated."""
        mask, dropped = _resolve_selectmasks(row(selectmasks="1,1,1"), [1, 0, 1])
        self.assertEqual(mask, [1, 1, 1])
        self.assertEqual(dropped, 0)


class MisalignedInputTest(unittest.TestCase):
    def test_a_short_selectmasks_raises(self):
        with self.assertRaises(ValueError) as err:
            _resolve_selectmasks(row(selectmasks="1,1"), [1, 0, 1])
        self.assertIn("selectmasks", str(err.exception))
        self.assertIn("2 entries against 3 responses", str(err.exception))

    def test_a_short_is_repeat_raises_rather_than_scoring_the_repeats(self):
        """The half that used to reopen the leak in silence."""
        with self.assertRaises(ValueError) as err:
            _resolve_selectmasks(row(selectmasks="1,1,1", is_repeat="0,1"), [1, 0, 1])
        self.assertIn("is_repeat", str(err.exception))

    def test_a_long_is_repeat_also_raises(self):
        with self.assertRaises(ValueError):
            _resolve_selectmasks(
                row(selectmasks="1,1,1", is_repeat="0,1,0,1"), [1, 0, 1]
            )

    def test_the_message_names_the_file(self):
        """A row number would be better, but the file is what the caller has."""
        with self.assertRaises(ValueError) as err:
            _resolve_selectmasks(row(selectmasks="1,1"), [1, 0, 1], "train_valid.csv")
        self.assertIn("train_valid.csv", str(err.exception))

    def test_the_message_says_what_to_do(self):
        with self.assertRaises(ValueError) as err:
            _resolve_selectmasks(row(selectmasks="1,1"), [1, 0, 1])
        self.assertIn("Regenerate", str(err.exception))


class RealDataIsAlignedTest(unittest.TestCase):
    """Raising costs nothing today: the shipped files all line up.

    Measured over assist2009 and algebra2005 when the check was added -- zero
    mismatching rows. This guards the claim, so a preprocessing change that
    breaks alignment is caught here rather than at the first training run.
    """

    def test_shipped_sequence_files_have_aligned_masks(self):
        import json

        data_config = json.loads(
            (ROOT / "configs" / "data_config.json").read_text(encoding="utf-8")
        )
        checked = 0
        for name in ("assist2009", "algebra2005"):
            cfg = data_config.get(name)
            if not cfg:
                continue
            path = Path(cfg["dpath"]) / cfg["train_valid_file"]
            if not path.is_absolute():
                path = ROOT / path
            if not path.exists():
                continue
            frame = pd.read_csv(path, dtype=str, keep_default_na=False, nrows=500)
            for _, record in frame.iterrows():
                responses = record["responses"].split(",")
                with self.subTest(dataset=name):
                    _resolve_selectmasks(record, responses, str(path))
            checked += 1

        if checked == 0:
            self.skipTest("no preprocessed sequence files available")


if __name__ == "__main__":
    unittest.main()
