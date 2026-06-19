import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("dtransformer")
class DTransformerTrainer(BaseTrainer):
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
            raise ValueError("DTransformer requires question, concept, and response sequences.")

        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)
        if train and getattr(self.model, "emb_type", "") == "qid_cl":
            preds, reg_loss = self.model.get_cl_loss(c_full.long(), r_full.long(), q_full.long())
        else:
            preds, reg_loss = self.model.get_loss(c_full.long(), r_full.long(), q_full.long())
        y = _align_shifted_preds(preds, rshft)
        loss = _masked_bce(y, rshft, sm, reg_loss)
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


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
