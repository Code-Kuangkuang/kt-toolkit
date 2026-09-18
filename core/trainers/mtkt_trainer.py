"""Trainer adapter for the MTKT port.

Same shape as core/trainers/stablekt_trainer.py -- `dcur` in, a tuple out whose
first element is the prediction -- with one difference: MTKT also takes `dgaps`,
the three log2-bucketed time features, assembled from the batch exactly as
core/trainers/dkt_forget_trainer.py does.
"""

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer

GAP_KEYS = ("rgaps", "sgaps", "pcounts", "shft_rgaps", "shft_sgaps", "shft_pcounts")


@TRAINER_REGISTRY.register("mtkt")
class MTKTTrainer(BaseTrainer):
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
    ):
        super().__init__(num_epochs=num_epochs, hooks=hooks, test_loader=test_loader)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience

    def _train_epoch(self, epoch):
        self.model.train()
        losses = []
        total_batches = len(self.train_loader)

        print(f"\n== Epoch {epoch}/{self.num_epochs} ==")
        print("=" * 50)

        for batch_idx, batch in enumerate(self.train_loader):
            pred, target, loss = self._forward_batch(batch, train=True)
            if pred.numel() == 0:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            self._print_progress(batch_idx, total_batches, loss.item())

        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch, train=False):
        dcur = {
            k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)
        }
        missing = [k for k in GAP_KEYS if k not in dcur]
        if missing:
            raise ValueError(
                f"MTKT requires the gap features {missing}; the dataloader was "
                f"built without include_dkt_forget=True."
            )
        dgaps = {k: dcur[k] for k in GAP_KEYS}

        rshft = dcur["shft_rseqs"].float()
        sm = dcur["smasks"].bool()

        result = self.model(dcur, dgaps, train=train)
        preds = result[0] if isinstance(result, tuple) else result
        preds_for_loss = preds[:, 1:] if preds.size(1) == rshft.size(1) + 1 else preds
        if preds_for_loss.shape != rshft.shape:
            raise ValueError(
                f"MTKT prediction shape {tuple(preds.shape)} does not align with "
                f"shifted targets {tuple(rshft.shape)}."
            )

        pred = torch.masked_select(preds_for_loss, sm)
        target = torch.masked_select(rshft, sm)
        if pred.numel() == 0:
            return pred, target, preds.sum() * 0.0
        loss = binary_cross_entropy(pred.double(), target.double())
        if not torch.isfinite(loss):
            raise FloatingPointError("MTKT produced a NaN/Inf loss.")
        return pred, target, loss
