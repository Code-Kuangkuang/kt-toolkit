import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


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
        y = y_full.gather(-1, qshft.unsqueeze(-1)).squeeze(-1)
        loss = _masked_bce(y, rshft, sm)
        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _masked_bce(preds, target, mask):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    return binary_cross_entropy(y, t)
