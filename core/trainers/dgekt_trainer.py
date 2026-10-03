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
        # `/ num_students` is what keeps kd_lambda meaning what the paper's does.
        # Upstream's eval.py accumulates the supervised term ACROSS the batch --
        # `for student in ...: loss = loss + crossEntropy(...)`, three per student
        # -- while the distillation term is a raw `.sum()`. Both therefore grow
        # with the batch, so their ratio does not. Here the supervised term is one
        # mean over every valid position and does not grow with the batch, so
        # without this division the distillation term is batch_size times
        # over-weighted. Measured on assist2009 fold 0 at batch 64: first-batch
        # supervised 2.08 vs distillation 27.93, i.e. 93% of the gradient pushed
        # the three branches towards agreement instead of towards the labels. The
        # cheapest way to agree is to emit a constant, and that is what training
        # did: loss pinned at 3*ln2 = 2.0794 (chance), every head bias at
        # p = 0.5009, the hypergraph branch dead (0% positive pre-activations, from
        # 47.5% at init), and the model unable to overfit even a single batch.
        # Test AUC 0.5856; with this division, 0.7401 and still improving.
        num_students = max(int(smasks.shape[0]), 1)
        kd_loss = (
            self.kd_lambda
            * (
                torch.abs(valid_ensemble - valid_concept).sum()
                + torch.abs(valid_ensemble - valid_transition).sum()
            )
            / num_students
        )
        loss = supervised_loss + kd_loss

        pred = torch.stack([torch.sigmoid(logits[smasks]) for logits in gathered], dim=0).mean(0)
        return pred, target, loss
