"""What a redirected run must write, and what it must not.

Every sweep redirects stdout to a file, and the terminal-shaped output did not
survive the trip: `_print_progress` redrew a bar with `\\r`, which a file cannot
overwrite, so each redraw became another line. One hd_akt log reached 447 KB
across 10,250 lines, of which roughly 8,900 were the bar and about 300 carried
information. `grep` was slow and `tail` showed nothing but bar.

`isatty` was already being consulted at that call site -- to decide whether to
flush, not whether to draw. That is the exact shape of mistake that comes back,
so it is pinned here rather than left to review.
"""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from core.hooks import MetricsJsonlHook
from core.trainer import BaseTrainer


def _bare_trainer():
    """Only `_print_progress` and `_log_epoch` are under test; neither touches
    trainer state beyond class attributes, so no loaders are needed."""
    return BaseTrainer.__new__(BaseTrainer)


def _capture(fn):
    """Run `fn` with stdout redirected, which is what makes isatty() false."""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        fn()
    return buffer.getvalue()


class RedirectedProgressTest(unittest.TestCase):
    def test_no_carriage_returns_when_stdout_is_a_file(self):
        """A `\\r` in a file is not an overwrite, it is a stray control byte."""
        trainer = _bare_trainer()
        out = _capture(
            lambda: [trainer._print_progress(i, 40, 0.5) for i in range(40)]
        )
        self.assertNotIn("\r", out)

    def test_progress_is_bounded_per_epoch(self):
        """The bug was unbounded growth: one line per batch, forever."""
        trainer = _bare_trainer()
        for total in (40, 400, 4000):
            with self.subTest(batches=total):
                out = _capture(
                    lambda: [
                        trainer._print_progress(i, total, 0.5)
                        for i in range(total)
                    ]
                )
                lines = [l for l in out.splitlines() if l.strip()]
                self.assertLessEqual(
                    len(lines), BaseTrainer.PROGRESS_CHECKPOINTS + 1,
                    f"{total} batches produced {len(lines)} progress lines",
                )

    def test_progress_still_reports_the_last_batch(self):
        """Bounded must not mean silent: a long epoch needs a heartbeat, and the
        final batch is the one that says the epoch finished rather than hung."""
        trainer = _bare_trainer()
        out = _capture(lambda: [trainer._print_progress(i, 40, 0.5) for i in range(40)])
        self.assertIn("40/40", out)
        self.assertIn("100%", out)

    def test_epoch_summary_is_one_line(self):
        """Nine lines of frame per epoch is 1,500 lines on a 171-epoch run."""
        trainer = _bare_trainer()
        out = _capture(
            lambda: trainer._log_epoch(
                {"epoch": 7, "train_loss": 0.5, "valid_auc": 0.8, "valid_acc": 0.7}
            )
        )
        lines = [l for l in out.splitlines() if l.strip()]
        self.assertEqual(len(lines), 1, out)
        for token in ("epoch 7", "train_loss 0.5000", "valid_auc 0.8000"):
            self.assertIn(token, lines[0])

    def test_epoch_summary_survives_a_missing_metric(self):
        """A single-class split scores None rather than a float."""
        trainer = _bare_trainer()
        out = _capture(
            lambda: trainer._log_epoch(
                {"epoch": 1, "train_loss": 0.5, "valid_auc": None}
            )
        )
        self.assertIn("valid_auc N/A", out)


class MetricsJsonlIdentityTest(unittest.TestCase):
    def test_every_row_can_be_attributed_without_its_path(self):
        """Rows from different runs must survive being concatenated.

        Without this, cross-run analysis has to recover identity by parsing the
        directory name -- and `saved_model/baseline_table`'s naming does not
        encode the label-flip ratio, so two noise levels would be
        indistinguishable once merged.
        """
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            identity = {
                "dataset": "assist2009", "model": "hd_dkt",
                "fold": 2, "seed": 3407, "train_label_flip_ratio": 0.1,
            }
            hook = MetricsJsonlHook(str(path), identity=identity)
            hook.on_train_start(None)
            hook.on_epoch_end(None, {"epoch": 1, "valid_auc": 0.76})
            hook.on_epoch_end(None, {"epoch": 2, "valid_auc": 0.77})

            rows = [json.loads(l) for l in path.read_text().splitlines()]
            self.assertEqual(len(rows), 2)
            for row in rows:
                for key, value in identity.items():
                    self.assertEqual(row[key], value)
                self.assertIsInstance(row["epoch_seconds"], float)
                self.assertIn("finished_at", row)

    def test_identity_cannot_overwrite_a_metric(self):
        """Metrics are merged after identity, so a name clash keeps the metric."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metrics.jsonl"
            hook = MetricsJsonlHook(str(path), identity={"epoch": "wrong"})
            hook.on_train_start(None)
            hook.on_epoch_end(None, {"epoch": 3})
            row = json.loads(path.read_text().splitlines()[0])
            self.assertEqual(row["epoch"], 3)


if __name__ == "__main__":
    unittest.main()
