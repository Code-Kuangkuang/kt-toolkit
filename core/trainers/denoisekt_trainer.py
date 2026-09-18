"""Trainer adapter for the DenoiseKT port.

`DenoiseKTNet.forward(cq, cc, cr)` wants the full-length sequences and returns
predictions already sliced to `[:, 1:]`, so unlike the AKT-family trainers there
is no shift to undo here. `cc` must keep its concept axis -- `boost_focus`
compares whole concept sets between positions and `get_avg_skill_emb` averages
over them -- which is why "denoisekt" is in MULTI_CONCEPT_MODELS.
"""

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("denoisekt")
class DenoiseKTTrainer(BaseTrainer):
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
        cq = _full_sequence(batch, "qseqs", "shft_qseqs", self.device)
        cc = _full_sequence(batch, "cseqs", "shft_cseqs", self.device)
        cr = _full_sequence(batch, "rseqs", "shft_rseqs", self.device)
        if cq is None or cc is None or cr is None:
            raise ValueError("DenoiseKT requires question, concept, and response sequences.")
        if cc.dim() == 2:
            # boost_focus unpacks three axes; a 2-D concept tensor would fail
            # there with a shape error rather than here with a reason.
            cc = cc.unsqueeze(-1)

        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        preds, contrast_loss = self.model(cq.long(), cc.long(), cr.long())
        if preds.shape != rshft.shape:
            raise ValueError(
                f"DenoiseKT prediction shape {tuple(preds.shape)} does not align "
                f"with shifted targets {tuple(rshft.shape)}."
            )

        y = torch.masked_select(preds.double(), sm)
        t = torch.masked_select(rshft.double(), sm)
        if y.numel() == 0:
            return y, t, preds.sum() * 0.0
        loss = binary_cross_entropy(y, t)
        if torch.is_tensor(contrast_loss):
            loss = loss + contrast_loss
        if not torch.isfinite(loss):
            raise FloatingPointError("DenoiseKT produced a NaN/Inf loss.")
        return y, t, loss


def _full_sequence(batch, seq_key, shft_key, device, dtype=torch.long):
    seqs = batch.get(seq_key)
    shft = batch.get(shft_key)
    if seqs is None or seqs.numel() == 0 or shft is None or shft.numel() == 0:
        return None
    seqs = seqs.to(device).to(dtype)
    shft = shft.to(device).to(dtype)
    return torch.cat((seqs[:, 0:1], shft), dim=1)
