import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, mse_loss

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("gbktv5")
@TRAINER_REGISTRY.register("cgbkt")
class CGBKTTrainer(BaseTrainer):
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

        warmup_epochs = int(other_config.get("warmup_epochs", 3))
        use_scheduler = bool(other_config.get("use_scheduler", True))
        self.scheduler = None
        if use_scheduler:
            from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

            warmup = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
            cosine = CosineAnnealingLR(optimizer, T_max=max(num_epochs - warmup_epochs, 1), eta_min=1e-6)
            self.scheduler = SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

    @staticmethod
    def _concat_full(seqs, shft):
        if seqs is None or seqs.numel() == 0:
            return None
        return torch.cat((seqs[:, :1], shft), dim=1)

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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(loss.item())
            self._print_progress(batch_idx, total_batches, loss.item())

        if self.scheduler is not None:
            self.scheduler.step()
        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch):
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"]
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"]
        sm = batch["smasks"]

        qseqs = qseqs.to(self.device) if qseqs is not None else None
        cseqs = cseqs.to(self.device) if cseqs is not None else None
        rseqs = rseqs.to(self.device)
        qshft = qshft.to(self.device) if qshft is not None else None
        cshft = cshft.to(self.device) if cshft is not None else None
        rshft = rshft.to(self.device)
        sm = sm.to(self.device)

        q_full = self._concat_full(qseqs, qshft)
        c_full = self._concat_full(cseqs, cshft)
        r_full = self._concat_full(rseqs, rshft)
        if q_full is None or c_full is None:
            raise ValueError("CGBKTTrainer requires both question and concept sequences.")

        outputs = self.model(q_full.long(), c_full.long(), r_full.float())
        y = outputs["y"]
        common_len = min(y.size(1), rshft.size(1), sm.size(1))
        if common_len <= 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty, torch.tensor(0.0, device=self.device)

        y = y[:, :common_len]
        rshft = rshft[:, :common_len]
        sm = sm[:, :common_len].bool() & (rshft[:, :common_len] >= 0)

        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm).float()
        if pred.numel() == 0:
            return pred, target, torch.tensor(0.0, device=self.device)

        loss_pred = binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), target)

        loss_center = torch.tensor(0.0, device=self.device)
        y_center = outputs.get("y_center")
        if y_center is not None:
            center_pred = torch.masked_select(y_center[:, :common_len], sm)
            if center_pred.numel() > 0:
                loss_center = binary_cross_entropy(center_pred.clamp(1e-5, 1.0 - 1e-5), target)

        loss_coverage = torch.tensor(0.0, device=self.device)
        y_coverage = outputs.get("y_coverage")
        if y_coverage is not None:
            coverage_pred = torch.masked_select(y_coverage[:, :common_len], sm)
            if coverage_pred.numel() > 0:
                loss_coverage = binary_cross_entropy(coverage_pred.clamp(1e-5, 1.0 - 1e-5), target)

        loss_conf = torch.tensor(0.0, device=self.device)
        conf = outputs.get("confidence")
        if conf is not None and float(self.other_config.get("lambda_conf", 0.0)) > 0.0:
            conf_sel = torch.masked_select(conf[:, :common_len], sm)
            if conf_sel.numel() > 0:
                pred_error = torch.abs(pred.detach() - target)
                conf_target = (1.0 - pred_error).clamp(min=0.0, max=1.0)
                loss_conf = mse_loss(conf_sel.float(), conf_target.float())

        loss_radius = torch.tensor(0.0, device=self.device)
        r_h = outputs.get("r_h_mean")
        r_d = outputs.get("r_d_mean")
        if r_h is not None and r_d is not None:
            rh = torch.masked_select(r_h[:, :common_len], sm).clamp_min(1e-6)
            rd = torch.masked_select(r_d[:, :common_len], sm).clamp_min(1e-6)
            if rh.numel() > 0 and rd.numel() > 0:
                loss_radius = rh.log().pow(2).mean() + rd.log().pow(2).mean()

        loss_gate = torch.tensor(0.0, device=self.device)
        gate = outputs.get("coverage_gate")
        if gate is not None:
            gate_sel = torch.masked_select(gate[:, :common_len], sm)
            if gate_sel.numel() > 0:
                loss_gate = gate_sel.mean()

        lambda_center = float(self.other_config.get("lambda_center", 0.05))
        lambda_coverage = float(self.other_config.get("lambda_coverage", 0.05))
        lambda_radius = float(self.other_config.get("lambda_radius", 1e-4))
        lambda_gate = float(self.other_config.get("lambda_gate", 1e-4))
        lambda_conf = float(self.other_config.get("lambda_conf", 0.0))

        loss = (
            loss_pred
            + lambda_center * loss_center
            + lambda_coverage * loss_coverage
            + lambda_radius * loss_radius
            + lambda_gate * loss_gate
            + lambda_conf * loss_conf
        )
        return pred, target, loss
