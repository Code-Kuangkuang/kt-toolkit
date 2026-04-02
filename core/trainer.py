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
            self._log_epoch(metrics_dict)
            if self._should_stop(epoch, metrics_dict):
                break
        self.hooks.train_end(self)

    def _train_epoch(self, epoch):
        raise NotImplementedError

    def _eval_epoch(self, epoch):
        raise NotImplementedError

    def _should_stop(self, epoch, metrics_dict):
        return False

    def _print_progress(self, batch_idx, total_batches, loss):
        """Print progress bar for training (one line, updates in place)."""
        bar_len = 25
        pct = int(100 * (batch_idx + 1) / total_batches)
        bar = "█" * int(bar_len * (batch_idx + 1) / total_batches) + "░" * (bar_len - int(bar_len * (batch_idx + 1) / total_batches))
        if batch_idx == total_batches - 1:
            # Clear the line first, then print 100%
            print("\r" + " " * 60 + "\r" + f"  │{bar}│ 100% | Loss: {loss:.4f}", flush=True)
        else:
            print(f"\r  │{bar}│ {pct:3d}% | Loss: {loss:.4f}", end="", flush=True)

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
        """Evaluate model on test set. Override in subclass."""
        return None
