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

            # Evaluate on test set every 10 epochs only when a labeled test
            # loader is available. Some datasets, such as Peiyou, provide a
            # prediction-only test file with hidden labels.
            if self.test_loader is not None and epoch % 10 == 0:
                test_metrics = self.evaluate_test()
                if test_metrics is not None:
                    print("")
                    print("=" * 50)
                    print("  [Test Set - Read Only] Every-10-Epochs Check")
                    print("=" * 50)
                    tauc = test_metrics.get("test_auc", -1)
                    tacc = test_metrics.get("test_acc", -1)
                    print(f"  Test AUC:  {tauc:.4f}" if tauc >= 0 else "  Test AUC:  N/A")
                    print(f"  Test Acc:  {tacc:.4f}" if tacc >= 0 else "  Test Acc:  N/A")
                    print("=" * 50)
                    print("  WARNING: This is a read-only check. Do NOT use these numbers")
                    print("     for hyperparameter tuning or final results (Data Leakage).")
                    print("     Final results must use the best-valid model.")
                    print("=" * 50)
                    print("")

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
        if loader is None:
            return {f"{prefix}_auc": -1, f"{prefix}_acc": -1}

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
            return {f"{prefix}_auc": -1, f"{prefix}_acc": -1}

        ts = np.concatenate(y_true, axis=0)
        ps = np.concatenate(y_score, axis=0)
        try:
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        except Exception:
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        return {f"{prefix}_auc": auc, f"{prefix}_acc": acc}

    def _print_progress(self, batch_idx, total_batches, loss):
        """Print progress bar for training (one line, updates in place)."""
        bar_len = 25
        pct = int(100 * (batch_idx + 1) / total_batches)
        filled = int(bar_len * (batch_idx + 1) / total_batches)
        bar = "#" * filled + "-" * (bar_len - filled)
        line = f"  [{bar}] {pct:3d}% | Loss: {loss:.4f}"
        if batch_idx == total_batches - 1:
            print("\r" + line + " " * 8, flush=True)
        else:
            print("\r" + line, end="", flush=True)
            if not sys.stdout.isatty():
                sys.stdout.flush()

    def _log_epoch(self, metrics_dict):
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
            auc_str = f"{auc:.4f}" if auc >= 0 else "N/A"
            lines.append(f"  Valid AUC:  {auc_str}")
        if "valid_acc" in metrics_dict:
            acc = metrics_dict['valid_acc']
            acc_str = f"{acc:.4f}" if acc >= 0 else "N/A"
            lines.append(f"  Valid Acc:  {acc_str}")

        for line in lines:
            print(line)
        print("=" * 50)
        print("")

    def evaluate_test(self):
        """Evaluate model on the labeled test set when one is available."""
        return self._score_loader(self.test_loader, prefix="test")
