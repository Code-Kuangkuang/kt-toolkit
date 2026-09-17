import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer
from models.multi_concept import pool_concept_predictions


@TRAINER_REGISTRY.register("atdkt")
class ATDKTTrainer(BaseTrainer):
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
        dcur = _to_device_dict(batch, self.device)
        rshft = dcur["shft_rseqs"].float()
        sm = dcur["smasks"]

        result = self.model(dcur, train=train)
        if isinstance(result, tuple):
            y_full, y2, y3 = result
        else:
            y_full, y2, y3 = result, None, None

        if y_full.dim() == 3:
            cshft = dcur["shft_cseqs"].long()
            # A plain gather breaks on [B,T,K] concepts; pooling the per-KC
            # predictions averages over the question's KCs, as pykt does in
            # qikt.py.  Identity for the single-concept [B,T] case.
            y, target_has_concept = pool_concept_predictions(
                y_full, cshft, y_full.size(-1)
            )
            if torch.any(sm.bool() & ~target_has_concept):
                raise ValueError(
                    "AT-DKT found a scored question without a valid concept id."
                )
        else:
            y = _align_shifted_preds(y_full, rshft)

        loss = _masked_bce(y, rshft, sm)
        emb_type = getattr(self.model, "emb_type", "")
        if train and "predcurc" in emb_type:
            loss = self.model.l1 * loss + self.model.l2 * y2
            if "his" in emb_type:
                loss = loss + self.model.l3 * y3
        elif train and "predhis" in emb_type:
            loss = self.model.l1 * loss + self.model.l2 * y2

        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _to_device_dict(batch, device):
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


def _align_shifted_preds(preds, target):
    preds_for_loss = preds[:, 1:] if preds.size(1) == target.size(1) + 1 else preds
    if preds_for_loss.shape != target.shape:
        raise ValueError(
            f"Prediction shape {tuple(preds.shape)} does not align with shifted targets {tuple(target.shape)}."
        )
    return preds_for_loss


def _masked_bce(preds, target, mask):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    return binary_cross_entropy(y, t)
