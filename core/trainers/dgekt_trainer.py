import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy_with_logits

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("dgekt")
class DGEKTTrainer(BaseTrainer):
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
        super().__init__(num_epochs=num_epochs, hooks=hooks, test_loader=test_loader)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience
        config = other_config or {}
        self.kd_lambda = float(config.get("kd_lambda", 5e-6))
        self.kd_temperature = float(config.get("kd_temperature", 0.5))
        if self.kd_temperature <= 0:
            raise ValueError("DGEKT kd_temperature must be positive.")

    def _train_epoch(self, epoch):
        self.model.train()
        losses = []
        total_batches = len(self.train_loader)

        print(f"\n== Epoch {epoch}/{self.num_epochs} ==")
        print("=" * 50)
        for batch_idx, batch in enumerate(self.train_loader):
            pred, _, loss = self._forward_batch(batch)
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
        qshft = batch.get("shft_qseqs")
        if qseqs is None or qseqs.numel() == 0:
            raise ValueError("DGEKTTrainer requires question sequences (qseqs).")
        if qshft is None or qshft.numel() == 0:
            raise ValueError("DGEKTTrainer requires shifted question sequences (shft_qseqs).")

        qseqs = qseqs.to(self.device).long()
        qshft = qshft.to(self.device).long()
        rseqs = batch["rseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        smasks = batch["smasks"].to(self.device).bool()

        outputs = self.model(qseqs, rseqs)
        branch_logits = [
            outputs["concept_logits"],
            outputs["transition_logits"],
            outputs["ensemble_logits"],
        ]
        max_target = int(qshft[smasks].max().item()) if smasks.any() else -1
        if max_target >= self.model.num_q:
            raise ValueError(
                f"DGEKT target question id {max_target} exceeds num_q={self.model.num_q}."
            )

        safe_qshft = qshft.masked_fill(~smasks, 0)
        gathered = [
            logits.gather(-1, safe_qshft.unsqueeze(-1)).squeeze(-1)
            for logits in branch_logits
        ]
        if not smasks.any():
            empty = gathered[0][smasks]
            return empty, rshft[smasks], sum(logits.sum() * 0.0 for logits in branch_logits)

        target = rshft[smasks]
        supervised_loss = sum(
            binary_cross_entropy_with_logits(logits[smasks], target)
            for logits in gathered
        )

        temperature = self.kd_temperature
        valid_concept = torch.sigmoid(branch_logits[0][smasks] / temperature)
        valid_transition = torch.sigmoid(branch_logits[1][smasks] / temperature)
        valid_ensemble = torch.sigmoid(branch_logits[2][smasks] / temperature)
        kd_loss = self.kd_lambda * (
            torch.abs(valid_ensemble - valid_concept).sum()
            + torch.abs(valid_ensemble - valid_transition).sum()
        )
        loss = supervised_loss + kd_loss

        pred = torch.stack([torch.sigmoid(logits[smasks]) for logits in gathered], dim=0).mean(0)
        return pred, target, loss
