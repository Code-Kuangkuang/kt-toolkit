import sys

import numpy as np
import torch
from sklearn import metrics

from core.hooks import HookList


class BaseTrainer:
    def __init__(self, num_epochs, hooks=None, test_loader=None):
        self.num_epochs = num_epochs
        self.hooks = HookList(hooks)
        self.best_epoch = -1
        self.best_metric = None
        self.test_loader = test_loader

    def run(self):
        self.hooks.train_start(self)
        for epoch in range(1, self.num_epochs + 1):
            train_loss = self._train_epoch(epoch)
            metrics_dict = self._eval_epoch(epoch)
            metrics_dict["train_loss"] = train_loss
            metrics_dict["epoch"] = epoch
            self.hooks.epoch_end(self, metrics_dict)

            # The test set is deliberately not scored inside this loop. It used
            # to be printed every 10 epochs behind a "read only" banner, but a
            # number you can see is a number you can tune against: watching test
            # AUC move while adjusting hyperparameters is selection on the test
            # set regardless of what the banner says. Test scoring happens once,
            # in train_runner, against the best-validation checkpoint.
            self._log_epoch(metrics_dict)
            if self._should_stop(epoch, metrics_dict):
                break
        self.hooks.train_end(self)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _eval_epoch(self, epoch):
        return self._score_loader(self.valid_loader, prefix="valid")

    def _should_stop(self, epoch, metrics_dict):
        metric_key = getattr(self, "metric_key", "valid_auc")
        metric = metrics_dict.get(metric_key)
        if metric is None:
            return False
        if self.best_metric is None or metric > self.best_metric + 1e-3:
            self.best_metric = metric
            self.best_epoch = epoch
            return False
        patience = getattr(self, "patience", None)
        if patience is None:
            return False
        return (epoch - self.best_epoch) >= patience

    def _score_loader(self, loader, prefix):
        # `None`, not -1, for an unavailable metric. -1 is a number: it flows
        # into best_metrics.json and is then averaged by aggregate_fold_metrics
        # like any other value, so a single unscorable fold drags a five-fold
        # mean down by ~0.35 while the fold count still reads 5/5. `None` is
        # skipped by the aggregator and shows up as a short count instead.
        if loader is None:
            return {f"{prefix}_auc": None, f"{prefix}_acc": None}

        self.model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for batch in loader:
                result = self._forward_batch(batch)
                pred, target = result[0], result[1]
                if pred.numel() == 0:
                    continue
                y_score.append(pred.detach().cpu().numpy())
                y_true.append(target.detach().cpu().numpy())

        if not y_true:
            print(f"Warning: {prefix} loader produced no scored positions.")
            return {f"{prefix}_auc": None, f"{prefix}_acc": None}

        ts = np.concatenate(y_true, axis=0)
        ps = np.concatenate(y_score, axis=0)

        # Two very different reasons roc_auc_score raises, previously collapsed
        # into the same -1: a split that happens to be single-class (benign, and
        # expected on tiny debug splits), versus NaN predictions or mismatched
        # shapes (a bug that must not be swallowed).
        if not np.isfinite(ps).all():
            raise ValueError(
                f"{prefix}: model produced {np.count_nonzero(~np.isfinite(ps))} "
                f"non-finite predictions out of {ps.size}. This is a model bug, "
                "not a scoring edge case."
            )
        if ts.shape[0] != ps.shape[0]:
            raise ValueError(
                f"{prefix}: {ts.shape[0]} targets against {ps.shape[0]} predictions. "
                "They must be flattened through the same smasks."
            )

        classes = np.unique(ts)
        if classes.size < 2:
            print(
                f"Warning: {prefix} split is single-class (all {classes.tolist()}); "
                "AUC is undefined and reported as null."
            )
            auc = None
        else:
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)

        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        return {f"{prefix}_auc": auc, f"{prefix}_acc": acc}

    #: Progress lines per epoch when stdout is a file rather than a terminal.
    PROGRESS_CHECKPOINTS = 4

    def _print_progress(self, batch_idx, total_batches, loss):
        """Show training progress, in whichever form the destination can use.

        A terminal redraws one line with `\\r`. A file cannot: every redraw
        becomes another line, and a sweep always redirects to a file. That is
        what made a single hd_akt log 447 KB across 10,250 lines, of which
        roughly 8,900 were this bar and about 300 carried information -- so the
        log was effectively unreadable by `grep` and useless by `tail`.

        `isatty` was already consulted here, but only to decide whether to
        flush. It decides whether to draw a bar at all.
        """
        last = batch_idx == total_batches - 1
        pct = int(100 * (batch_idx + 1) / total_batches)

        if not sys.stdout.isatty():
            # A handful of plain lines per epoch: enough to see a long run is
            # alive and where it is, few enough to read around.
            step = max(1, total_batches // self.PROGRESS_CHECKPOINTS)
            if last or (batch_idx + 1) % step == 0:
                print(
                    f"  batch {batch_idx + 1}/{total_batches} "
                    f"({pct:3d}%) | loss {loss:.4f}",
                    flush=True,
                )
            return

        bar_len = 25
        filled = int(bar_len * (batch_idx + 1) / total_batches)
        bar = "#" * filled + "-" * (bar_len - filled)
        line = f"  [{bar}] {pct:3d}% | Loss: {loss:.4f}"
        if last:
            print("\r" + line + " " * 8, flush=True)
        else:
            print("\r" + line, end="", flush=True)

    @staticmethod
    def _fmt_metric(value):
        return "N/A" if value is None else f"{value:.4f}"

    def _log_epoch(self, metrics_dict):
        if not sys.stdout.isatty():
            # One greppable, sortable line per epoch. The boxed form below costs
            # nine lines each, which on a 171-epoch run is 1,500 lines of frame
            # around 700 lines of number.
            parts = [f"epoch {metrics_dict.get('epoch', '?')}"]
            if "train_loss" in metrics_dict:
                parts.append(f"train_loss {metrics_dict['train_loss']:.4f}")
            for key in ("valid_auc", "valid_acc"):
                if key in metrics_dict:
                    parts.append(f"{key} {self._fmt_metric(metrics_dict[key])}")
            print(" | ".join(parts), flush=True)
            return

        # Beautify output
        print("")
        print("=" * 50)
        print("  Epoch Summary")
        print("=" * 50)

        lines = []
        if "epoch" in metrics_dict:
            lines.append(f"  Epoch:      {metrics_dict['epoch']}")
        if "train_loss" in metrics_dict:
            lines.append(f"  Train Loss: {metrics_dict['train_loss']:.4f}")
        if "valid_auc" in metrics_dict:
            auc = metrics_dict['valid_auc']
            lines.append(f"  Valid AUC:  {'N/A' if auc is None else f'{auc:.4f}'}")
        if "valid_acc" in metrics_dict:
            acc = metrics_dict['valid_acc']
            lines.append(f"  Valid Acc:  {'N/A' if acc is None else f'{acc:.4f}'}")

        for line in lines:
            print(line)
        print("=" * 50)
        print("")

    def evaluate_test(self):
        """Evaluate model on the labeled test set when one is available."""
        return self._score_loader(self.test_loader, prefix="test")

    def evaluate_window_test(self):
        """Score the windowed test set -- the protocol pykt reports.

        The plain test file chops a learner into non-overlapping chunks, so a
        position sitting near a chunk boundary is scored with almost no
        history.  The windowed file instead emits one row per position, each
        carrying the full preceding window, and scores only that last position.
        Same predictions, far more history, so the numbers are not
        interchangeable -- both are reported rather than one replacing the
        other.

        The loader is attached by the runner rather than taken through
        __init__, because every trainer subclass declares its own constructor
        and a new required keyword would break all of them.
        """
        return self._score_loader(
            getattr(self, "window_test_loader", None), prefix="window_test"
        )
