import numpy as np
import torch

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("iekt")
class IEKTTrainer(BaseTrainer):
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
        other_config=None,
        test_loader=None,
    ):
        if other_config is None:
            other_config = {}
        super().__init__(num_epochs=num_epochs, hooks=hooks, test_loader=test_loader)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience
        self.other_config = other_config

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

        # Ensure 100% is shown at the end
        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch):
        # Prepare data
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"]
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"]
        sm = batch["smasks"]
        masks = batch.get("masks")

        # Move to device
        if qseqs is not None:
            qseqs = qseqs.to(self.device)
        if cseqs is not None:
            cseqs = cseqs.to(self.device)
        if rseqs is not None:
            rseqs = rseqs.to(self.device)
        if qshft is not None:
            qshft = qshft.to(self.device)
        if cshft is not None:
            cshft = cshft.to(self.device)
        if rshft is not None:
            rshft = rshft.to(self.device)
        if sm is not None:
            sm = sm.to(self.device)
        if masks is not None:
            masks = masks.to(self.device)

        # Build data dict for IEKT model
        data = {
            "qseqs": qseqs,
            "cseqs": cseqs,
            "rseqs": rseqs,
            "shft_qseqs": qshft,
            "shft_cseqs": cshft,
            "shft_rseqs": rshft,
            "masks": masks if masks is not None else torch.zeros_like(sm),
            "smasks": sm,
        }

        # Forward through model
        pred, target, loss = self.model.train_one_step(data, process=True)

        # Some implementations return [B, T] logits, while IEKT currently
        # returns flattened valid positions; only mask in the matrix case.
        if pred.dim() > 1:
            pred = torch.masked_select(pred, sm)
            target = torch.masked_select(target, sm)
        return pred, target, loss
