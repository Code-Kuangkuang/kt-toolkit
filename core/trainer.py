from core.hooks import HookList


class BaseTrainer:
    def __init__(self, num_epochs, hooks=None):
        self.num_epochs = num_epochs
        self.hooks = HookList(hooks)
        self.best_epoch = -1
        self.best_metric = None

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

    def _log_epoch(self, metrics_dict):
        keys = ["epoch", "train_loss", "valid_auc", "valid_acc"]
        parts = []
        for key in keys:
            if key not in metrics_dict:
                continue
            value = metrics_dict[key]
            if isinstance(value, (int, float)):
                parts.append(f"{key}={value:.4f}")
            else:
                parts.append(f"{key}={value}")
        if parts:
            print(" | ".join(parts))
