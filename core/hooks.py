import copy
import datetime
import json
import os
import time

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


class BestMetricsHook(Hook):
    def __init__(self, metric_key="valid_auc", mode="max"):
        self.metric_key = metric_key
        self.mode = mode
        self.best_value = None
        self.best_metrics = None

    def _is_better(self, value):
        if self.best_value is None:
            return True
        if self.mode == "min":
            return value < self.best_value
        return value > self.best_value

    def on_epoch_end(self, trainer, metrics):
        if self.metric_key not in metrics:
            return
        value = metrics.get(self.metric_key)
        if value is None:
            return
        if self._is_better(value):
            self.best_value = value
            self.best_metrics = copy.deepcopy(metrics)
            trainer.best_metrics = self.best_metrics
            trainer.best_metric_key = self.metric_key
            trainer.best_metric_value = self.best_value


class MetricsJsonlHook(Hook):
    """One JSON object per epoch, carrying enough identity to be concatenated.

    Each line used to be `{valid_auc, valid_acc, train_loss, epoch, time}` and
    nothing else, so `cat */metrics.jsonl` produced a file in which no row could
    be attributed to a run. The only way to analyse across runs was to parse the
    directory path, which makes every analysis script depend on a naming
    convention -- and `saved_model/baseline_table`'s convention does not even
    encode the label-flip ratio.

    `time` was a bare unix float. What a reader actually wants is how long the
    epoch took, which previously required subtracting adjacent rows.
    """

    def __init__(self, path, identity=None):
        self.path = path
        #: dataset / model / fold / seed, merged into every row.
        self.identity = dict(identity or {})
        self._epoch_started = None

    @staticmethod
    def _json_safe(value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, dict):
            return {str(k): MetricsJsonlHook._json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [MetricsJsonlHook._json_safe(v) for v in value]
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return str(value)

    def on_train_start(self, trainer):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._epoch_started = time.time()

    def on_epoch_end(self, trainer, metrics):
        now = time.time()
        payload = dict(self.identity)
        payload.update(self._json_safe(metrics))
        payload["epoch_seconds"] = (
            round(now - self._epoch_started, 3)
            if self._epoch_started is not None
            else None
        )
        payload["finished_at"] = datetime.datetime.fromtimestamp(now).isoformat(
            timespec="seconds"
        )
        # Kept so anything already reading `time` keeps working.
        payload["time"] = now
        self._epoch_started = now
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=True) + "\n")


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
