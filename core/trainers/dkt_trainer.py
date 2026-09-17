import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer
from models.multi_concept import pool_concept_predictions


@TRAINER_REGISTRY.register("dkt")
class DKTTrainer(BaseTrainer):
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
            pred, target, loss = self._forward_batch(batch)
            if pred.numel() == 0:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            # Progress bar
            self._print_progress(batch_idx, total_batches, loss.item())


        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch):
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")

        if qseqs is not None and qseqs.numel() > 0:
            qseqs = qseqs.to(self.device).long()
        if cseqs is not None and cseqs.numel() > 0:
            cseqs = cseqs.to(self.device).long()
        if qshft is not None and qshft.numel() > 0:
            qshft = qshft.to(self.device).long()
        if cshft is not None and cshft.numel() > 0:
            cshft = cshft.to(self.device).long()

        rseqs = batch["rseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        if cseqs is not None and cseqs.numel() > 0:
            base_seqs = cseqs
        elif qseqs is not None and qseqs.numel() > 0:
            base_seqs = qseqs
        else:
            raise ValueError("DKTTrainer requires qseqs or cseqs.")

        if cshft is not None and cshft.numel() > 0:
            target_idx = cshft
        elif qshft is not None and qshft.numel() > 0:
            target_idx = qshft
        else:
            raise ValueError("DKTTrainer requires shft_cseqs or shft_qseqs.")

        y = self.model(base_seqs, rseqs)

        y, target_has_concept = pool_concept_predictions(
            y, target_idx, y.size(-1)
        )
        if torch.any(sm.bool() & ~target_has_concept):
            raise ValueError("DKT found a scored question without a valid concept id.")

        loss = cal_loss(self.model, [y], rseqs, rshft, sm)

        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def cal_loss(model, ys, r, rshft, sm, preloss=None):
    y = torch.masked_select(ys[0], sm)
    t = torch.masked_select(rshft, sm)
    loss = binary_cross_entropy(y.double(), t.double())
    return loss

