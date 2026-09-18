"""Trainer adapter for the HCGKT port.

HCGKT does not train with a plain forward/backward. Upstream's
`pykt/models/train_model.py` gives it a FLAG-style adversarial loop, and the
loop is not incidental: `BGRL.forward` returns a zero contrastive loss whenever
`perb` is None, so running HCGKT through an ordinary trainer would silently drop
the contrastive half of the paper and train a graph-attention model instead.

The loop, as upstream has it:

  * a perturbation of shape `[num_q, emb_size]`, drawn uniform in
    `[-step_size, step_size]` and marked `requires_grad`;
  * `step_m` passes, each scaling its loss by `1 / step_m` and taking a sign-
    gradient ascent step on the perturbation between passes;
  * gradient clipping at `grad_clip` before the optimiser step;
  * an EMA update of BGRL's target encoder at rate `mm`, after the step.

Evaluation passes no perturbation, which is also what upstream does.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer

# Upstream reads these off the config and publishes no point values, only the
# sweep ranges in its examples/seedwandb/hcgkt.yaml:
#   step_size 1e-2..1e-1, step_m 1..5, grad_clip 5..20, mm 0.9..0.99
# These are mid-range picks from that, not authors' defaults. configs/kt_config.json
# carries the same numbers and overrides these.
DEFAULT_STEP_SIZE = 5e-2
DEFAULT_STEP_M = 3
DEFAULT_GRAD_CLIP = 10.0
DEFAULT_MM = 0.99


@TRAINER_REGISTRY.register("hcgkt")
class HCGKTTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        valid_loader,
        optimizer,
        num_epochs,
        device,
        hooks=None,
        metric_key="valid_auc",
        patience=10,
        test_loader=None,
        other_config=None,
    ):
        super().__init__(num_epochs=num_epochs, hooks=hooks, test_loader=test_loader)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience

        config = other_config or {}
        self.step_size = float(_setting(model, config, "step_size", DEFAULT_STEP_SIZE))
        self.step_m = max(1, int(_setting(model, config, "step_m", DEFAULT_STEP_M)))
        self.grad_clip = float(_setting(model, config, "grad_clip", DEFAULT_GRAD_CLIP))
        self.mm = float(_setting(model, config, "mm", DEFAULT_MM))

    def _train_epoch(self, epoch):
        self.model.train()
        losses = []
        total_batches = len(self.train_loader)

        print(f"\n== Epoch {epoch}/{self.num_epochs} ==")
        print("=" * 50)

        for batch_idx, batch in enumerate(self.train_loader):
            loss = self._adversarial_step(batch)
            if loss is None:
                continue
            losses.append(loss)
            self._print_progress(batch_idx, total_batches, loss)

        return float(np.mean(losses)) if losses else 0.0

    def _adversarial_step(self, batch):
        dcur = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        rshft = dcur["shft_rseqs"].float()
        sm = dcur["smasks"].bool()
        if sm.sum() == 0:
            return None

        num_q, emb_size = self.model.sfm_cl.pro_embed.shape
        perturb = torch.empty(
            num_q, emb_size, device=self.device
        ).uniform_(-self.step_size, self.step_size)
        perturb.requires_grad_()

        loss = self._batch_loss(dcur, rshft, sm, perturb) / self.step_m
        self.optimizer.zero_grad()
        for _ in range(self.step_m - 1):
            loss.backward()
            # Sign-gradient ascent on the perturbation; `perturb.grad` is reset
            # rather than zero_grad()-ed because it is not an optimiser tensor.
            perturb.data = perturb.detach() + self.step_size * torch.sign(
                perturb.grad.detach()
            )
            perturb.grad[:] = 0
            loss = self._batch_loss(dcur, rshft, sm, perturb) / self.step_m

        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        self.model.sfm_cl.gcl.update_target_network(self.mm)
        return float(loss.item()) * self.step_m

    def _batch_loss(self, dcur, rshft, sm, perturb):
        preds, _, _, contrast_loss = self.model(dcur, train=True, perb=perturb)
        loss = _masked_bce(preds[:, 1:], rshft, sm)
        if torch.is_tensor(contrast_loss):
            loss = loss + contrast_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("HCGKT produced a NaN/Inf loss.")
        return loss

    def _forward_batch(self, batch, train=False):
        """Evaluation path: no perturbation, so no contrastive term."""
        dcur = {k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)}
        rshft = dcur["shft_rseqs"].float()
        sm = dcur["smasks"].bool()

        output = self.model(dcur, train=False)
        preds = output[0] if isinstance(output, tuple) else output
        y = torch.masked_select(preds[:, 1:], sm)
        t = torch.masked_select(rshft, sm)
        if y.numel() == 0:
            return y, t, preds.sum() * 0.0
        return y, t, binary_cross_entropy(y.double(), t.double())


def _setting(model, config, name, default):
    """Config value, else what the model was constructed with, else the default."""
    if name in config and config[name] is not None:
        return config[name]
    value = getattr(model, name, None)
    return default if value is None else value


def _masked_bce(preds, target, mask):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    return binary_cross_entropy(y, t)
