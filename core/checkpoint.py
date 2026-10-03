"""Epoch-boundary recovery, separate from best-validation model selection."""
import os
import random
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch


def rng_state():
    state = np.random.get_state()
    return {"python": random.getstate(), "numpy": (state[0], state[1].tolist(), *state[2:]),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None}


def restore_rng(state):
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state((np_state[0], np.array(np_state[1], dtype=np.uint32), *np_state[2:]))
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None:
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


@contextmanager
def preserve_rng():
    state = rng_state()
    try:
        yield
    finally:
        restore_rng(state)


def atomic_torch_save(payload, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    os.close(fd)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_recovery(trainer):
    path = getattr(trainer, "recovery_path", None)
    if not path:
        return
    hooks = []
    for hook in trainer.hooks._hooks:
        hooks.append({key: getattr(hook, key) for key in ("best", "best_value", "best_metrics")
                      if hasattr(hook, key)})
    atomic_torch_save({
        "version": 1, "experiment_key": trainer.experiment_key,
        "epoch": trainer.current_epoch, "model": trainer.model.state_dict(),
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict() if hasattr(trainer, "scheduler") else None,
        "rng": rng_state(), "best_epoch": trainer.best_epoch, "best_metric": trainer.best_metric,
        "best_metrics": getattr(trainer, "best_metrics", None),
        "best_path": getattr(trainer, "best_path", None), "hooks": hooks,
        "stopped_early": getattr(trainer, "stopped_early", False),
    }, path)


def load_recovery(trainer, path, experiment_key):
    payload = torch.load(path, map_location=trainer.device, weights_only=True)
    if payload.get("version") != 1 or payload.get("experiment_key") != experiment_key:
        raise ValueError("Recovery checkpoint does not match this experiment's configuration/data/code.")
    trainer.model.load_state_dict(payload["model"])
    trainer.optimizer.load_state_dict(payload["optimizer"])
    if payload.get("scheduler") is not None:
        if not hasattr(trainer, "scheduler"):
            raise ValueError("Recovery checkpoint requires a scheduler.")
        trainer.scheduler.load_state_dict(payload["scheduler"])
    trainer.current_epoch = payload["epoch"]
    trainer.start_epoch = payload["epoch"] + 1
    trainer.best_epoch = payload["best_epoch"]
    trainer.best_metric = payload["best_metric"]
    trainer.best_metrics = payload["best_metrics"]
    trainer.stopped_early = payload.get("stopped_early", False)
    old_best = payload.get("best_path")
    if old_best:
        old_best = Path(old_best)
        if not old_best.is_file():
            raise ValueError("Recovery checkpoint's best-validation weights are missing.")
        new_best = Path(trainer.recovery_path).parent / old_best.name
        atomic_torch_save(torch.load(old_best, map_location="cpu", weights_only=True), new_best)
        trainer.best_path = str(new_best)
    if len(payload["hooks"]) != len(trainer.hooks._hooks):
        raise ValueError("Recovery hook configuration differs.")
    for hook, state in zip(trainer.hooks._hooks, payload["hooks"]):
        for key, value in state.items():
            setattr(hook, key, value)
    trainer._resume_rng = payload["rng"]
    payload["best_path"] = getattr(trainer, "best_path", None)
    atomic_torch_save(payload, trainer.recovery_path)
