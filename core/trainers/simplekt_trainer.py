import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("simplekt")
class SimpleKTTrainer(BaseTrainer):
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
            self._print_progress(batch_idx, total_batches, loss.item())


        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch):
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"].to(self.device).long()
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)
        pidseqs = batch.get("pidseqs")

        base_seqs = cseqs if cseqs is not None and cseqs.numel() > 0 else qseqs
        base_shft = cshft if cshft is not None and cshft.numel() > 0 else qshft

        if base_seqs is None or base_seqs.numel() == 0:
            raise ValueError("SimpleKTTrainer requires question or concept sequences.")

        base_seqs = base_seqs.to(self.device).long()
        base_shft = base_shft.to(self.device).long()

        # Prepare pidseqs
        if pidseqs is not None:
            pidseqs = pidseqs.to(self.device).long()

        preds = self.model(
            qseqs=qseqs.to(self.device).long() if qseqs is not None else None,
            rseqs=rseqs,
            cseqs=base_seqs,
            qshft=qshft.to(self.device).long() if qshft is not None else None,
            cshft=base_shft,
            rshft=rshft,
            pidseqs=pidseqs,
        )

        loss = cal_loss(preds, rshft, sm)
        pred = torch.masked_select(preds, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def cal_loss(preds, rshft, sm):
    y = torch.masked_select(preds.double(), sm)
    t = torch.masked_select(rshft.double(), sm)
    loss = binary_cross_entropy(y, t)
    return loss
