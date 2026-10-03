import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer
from models.multi_concept import pool_concept_predictions


@TRAINER_REGISTRY.register("dkt_forget")
@TRAINER_REGISTRY.register("dkt-forget")
class DKTForgetTrainer(BaseTrainer):
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

    training_forward_kwargs = {'train': True}

    def _forward_batch(self, batch, train=False):
        qseqs = batch.get("cseqs")
        qshft = batch.get("shft_cseqs")
        if qseqs is None or qseqs.numel() == 0:
            qseqs = batch.get("qseqs")
            qshft = batch.get("shft_qseqs")
        if qseqs is None or qseqs.numel() == 0 or qshft is None or qshft.numel() == 0:
            raise ValueError("DKT-forget requires concept or question sequences.")

        qseqs = qseqs.to(self.device).long()
        qshft = qshft.to(self.device).long()
        rseqs = batch["rseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)
        dgaps = {
            key: batch[key].to(self.device).long()
            for key in ("rgaps", "sgaps", "pcounts", "shft_rgaps", "shft_sgaps", "shft_pcounts")
        }

        y_full = self.model(qseqs, rseqs, dgaps)
        # A plain gather breaks on [B,T,K] concepts; pooling the per-KC
        # predictions averages over the question's KCs, as pykt does in
        # qikt.py.  Identity for the single-concept [B,T] case.
        y, target_has_concept = pool_concept_predictions(
            y_full, qshft, y_full.size(-1)
        )
        if torch.any(sm.bool() & ~target_has_concept):
            raise ValueError(
                "DKT-forget found a scored question without a valid concept id."
            )
        loss = _masked_bce(y, rshft, sm)
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _masked_bce(preds, target, mask):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    return binary_cross_entropy(y, t)
