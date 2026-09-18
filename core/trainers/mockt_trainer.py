"""Trainer adapter for the MoC-KT port.

Identical to core/trainers/robustkt_trainer.py except for one thing: MoC-KT's
`forward` takes a leading `s`, the true (unpadded) length of each sequence,
which `FrequencyLayer` uses to pick one of three convolution kernels.

`s` is computed from the padding mask only. It never touches `rseqs`, so no
response value reaches the bucket choice.
"""

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("mockt")
class MoCKTTrainer(BaseTrainer):
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
        c_full = _full_sequence(batch, "cseqs", "shft_cseqs", self.device)
        q_full = _full_sequence(batch, "qseqs", "shft_qseqs", self.device)
        r_full = _full_sequence(batch, "rseqs", "shft_rseqs", self.device)
        if c_full is None or q_full is None or r_full is None:
            raise ValueError("MoC-KT requires question, concept, and response sequences.")

        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)
        lengths = _true_lengths(batch, self.device)
        preds, reg_loss = self.model(lengths, c_full.long(), r_full.long(), q_full.long())
        y = _align_shifted_preds(preds, rshft)
        loss = _masked_bce(y, rshft, sm, reg_loss)
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _true_lengths(batch, device):
    """Unpadded length of each full sequence, as MoC-KT's bucket key.

    pyKT computes this in its own `mockt_data_loader.py` as
    `(rseqs != -1).sum(dim=1, keepdim=True).float()`, on the raw padded tensor.
    That formula CANNOT be carried over literally: this repo pads `rseqs` with 0,
    not -1 (0 is a valid id, which is why validity is read from `masks` -- see
    AGENTS.md), so `!= -1` is true everywhere and every sequence would report the
    full 200. All three length buckets would collapse into one and the mixture of
    convolutions -- the entire contribution -- would silently become a single
    kernel. Measured on assist2009 fold 0: pyKT's formula gives 200 for every row
    where the true lengths are 23, 45, 30, 180, ...

    `masks` covers the shifted sequence, so the full sequence is one longer.

    Shape and dtype match pyKT's `[B, 1]` float deliberately: `FrequencyLayer`
    branches on `s.shape[0] != 1` and calls `.squeeze(dim=0)` in the B=1 case,
    which turns a `[B]` tensor into a 0-dim mask and breaks the batch indexing.
    """
    masks = batch.get("masks")
    if masks is None or masks.numel() == 0:
        raise ValueError("MoC-KT requires the sequence mask to bucket by length.")
    masks = masks.to(device)
    return (masks.sum(dim=1, keepdim=True) + 1).float()


def _full_sequence(batch, seq_key, shft_key, device, dtype=torch.long):
    seqs = batch.get(seq_key)
    shft = batch.get(shft_key)
    if seqs is None or seqs.numel() == 0 or shft is None or shft.numel() == 0:
        return None
    seqs = seqs.to(device).to(dtype)
    shft = shft.to(device).to(dtype)
    return torch.cat((seqs[:, 0:1], shft), dim=1)


def _align_shifted_preds(preds, target):
    preds_for_loss = preds[:, 1:] if preds.size(1) == target.size(1) + 1 else preds
    if preds_for_loss.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(preds.shape)} does not align with shifted targets {tuple(target.shape)}."
        )
    return preds_for_loss


def _masked_bce(preds, target, mask, extra_loss=None):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    loss = binary_cross_entropy(y, t)
    if extra_loss is not None:
        loss = loss + extra_loss
    return loss
