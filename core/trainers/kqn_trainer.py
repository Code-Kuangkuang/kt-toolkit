import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("kqn")
class KQNTrainer(BaseTrainer):
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
        rseqs = batch["rseqs"].to(self.device).long()
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        # KQN one-hot indices are bounded by num_c, so prefer concept ids first.
        base_seqs = cseqs if cseqs is not None and cseqs.numel() > 0 else qseqs
        base_shft = cshft if cshft is not None and cshft.numel() > 0 else qshft

        if (
            base_seqs is None
            or base_seqs.numel() == 0
            or base_shft is None
            or base_shft.numel() == 0
        ):
            raise ValueError("KQNTrainer requires valid sequences and shifted sequences.")

        max_id = int(max(base_seqs.max().item(), base_shft.max().item()))
        if max_id >= self.model.num_c:
            raise ValueError(
                f"KQN input id {max_id} is out of range for num_c={self.model.num_c}. "
                "Use concept sequences (cseqs/shft_cseqs) or align ids with num_c."
            )

        base_seqs = base_seqs.to(self.device).long()
        base_shft = base_shft.to(self.device).long()

        # Forward through KQN model
        preds = self.model(base_seqs, rseqs, base_shft)

        # Compute loss
        loss = cal_loss(self.model, [preds], rseqs, rshft, sm)

        pred = torch.masked_select(preds, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def cal_loss(model, ys, r, rshft, sm, preloss=None):
    y = torch.masked_select(ys[0], sm)
    t = torch.masked_select(rshft, sm)
    loss = binary_cross_entropy(y.double(), t.double())
    return loss
