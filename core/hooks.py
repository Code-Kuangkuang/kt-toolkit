import os

import torch


class Hook:
    def on_train_start(self, trainer):
        pass

    def on_epoch_end(self, trainer, metrics):
        pass

    def on_train_end(self, trainer):
        pass

class HookList:
    def __init__(self, hooks=None):
        self._hooks = hooks or []

    def train_start(self, trainer):
        for h in self._hooks: h.on_train_start(trainer)

    def epoch_end(self, trainer, metrics):
        for h in self._hooks: h.on_epoch_end(trainer, metrics)

    def train_end(self, trainer):
        for h in self._hooks: h.on_train_end(trainer)


class SaveBestHook(Hook):
    def __init__(self, save_dir, filename="model.pt", metric_key="valid_auc", mode="max"):
        self.save_dir = save_dir
        self.filename = filename
        self.metric_key = metric_key
        self.mode = mode
        self.best = None

    def _is_better(self, value):
        if self.best is None:
            return True
        if self.mode == "min":
            return value < self.best
        return value > self.best

    def on_epoch_end(self, trainer, metrics):
        if self.metric_key not in metrics:
            return
        value = metrics[self.metric_key]
        if value is None:
            return
        if self._is_better(value):
            self.best = value
            os.makedirs(self.save_dir, exist_ok=True)
            save_path = os.path.join(self.save_dir, self.filename)
            torch.save(trainer.model.state_dict(), save_path)
            trainer.best_path = save_path


class WandbHook(Hook):
    def __init__(
        self,
        enabled=False,
        run_name=None,
        config=None,
        project=None,
        entity=None,
        tags=None,
        group=None,
        api_key=None,
        mode=None,
    ):
        self.enabled = enabled
        self.run_name = run_name
        self.config = config or {}
        self.project = project
        self.entity = entity
        self.tags = tags
        self.group = group
        self.api_key = api_key
        self.mode = mode
        self._run = None

    def _disable(self, reason):
        if self.enabled:
            print(f"[WandbHook] {reason}")
        self.enabled = False
        self._run = None

    def on_train_start(self, trainer):
        if not self.enabled:
            return
        try:
            import wandb

            if self.api_key:
                wandb.login(key=self.api_key)
            init_kwargs = {
                "name": self.run_name,
                "config": self.config,
                "project": self.project,
                "entity": self.entity,
                "tags": self.tags,
                "group": self.group,
            }
            if self.mode is not None:
                init_kwargs["mode"] = self.mode
            self._run = wandb.init(**init_kwargs)
        except Exception as exc:
            self._disable(f"wandb init failed, disabling logging. Reason: {exc}")

    def on_epoch_end(self, trainer, metrics):
        if not self.enabled or self._run is None:
            return
        try:
            import wandb

            wandb.log(metrics)
        except Exception as exc:
            self._disable(f"wandb log failed, disabling logging. Reason: {exc}")

    def on_train_end(self, trainer):
        if not self.enabled or self._run is None:
            return
        try:
            self._run.finish()
        except Exception as exc:
            self._disable(f"wandb finish failed. Reason: {exc}")
