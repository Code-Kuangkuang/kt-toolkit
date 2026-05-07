import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, mse_loss

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("gbktv2")
class GBKTV2Trainer(BaseTrainer):
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

            warmup = LinearLR(
                optimizer,
                start_factor=0.1,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(num_epochs - warmup_epochs, 1),
                eta_min=float(other_config.get("eta_min", 1e-6)),
            )
            self.scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )

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
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=float(self.other_config.get("grad_clip", 1.0)),
            )
            self.optimizer.step()
            losses.append(loss.item())
            self._print_progress(batch_idx, total_batches, loss.item())

        if self.scheduler is not None:
            self.scheduler.step()
        return float(np.mean(losses)) if losses else 0.0

    @staticmethod
    def _concat_full(seqs, shft):
        if seqs is None or seqs.numel() == 0:
            return None
        return torch.cat((seqs[:, :1], shft), dim=1)

    def _forward_batch(self, batch):
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"]
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"]
        sm = batch["smasks"]

        if qseqs is not None:
            qseqs = qseqs.to(self.device)
        if cseqs is not None:
            cseqs = cseqs.to(self.device)
        rseqs = rseqs.to(self.device)
        if qshft is not None:
            qshft = qshft.to(self.device)
        if cshft is not None:
            cshft = cshft.to(self.device)
        rshft = rshft.to(self.device)
        sm = sm.to(self.device)

        q_full = self._concat_full(qseqs, qshft)
        c_full = self._concat_full(cseqs, cshft)
        r_full = self._concat_full(rseqs, rshft)

        if q_full is None or c_full is None:
            raise ValueError("GBKTV2Trainer requires both question and concept sequences.")

        outputs = self.model(q_full.long(), c_full.long(), r_full.float())
        y = outputs["y"]
        y_ball = outputs.get("y_ball")
        y_concept_next = outputs.get("y_concept_next")
        theta = outputs["theta"]
        conf = outputs["confidence"]
        r_h_mean = outputs["r_h_mean"]
        r_d_mean = outputs["r_d_mean"]

        common_len = min(y.size(1), rshft.size(1), sm.size(1))
        if common_len <= 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty, torch.tensor(0.0, device=self.device)

        y = y[:, :common_len]
        if y_ball is not None:
            y_ball = y_ball[:, :common_len]
        if y_concept_next is not None:
            y_concept_next = y_concept_next[:, :common_len]
        theta = theta[:, :common_len]
        conf = conf[:, :common_len]
        r_h_mean = r_h_mean[:, :common_len]
        r_d_mean = r_d_mean[:, :common_len]
        rshft = rshft[:, :common_len]
        sm = sm[:, :common_len]

        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm).float()
        if pred.numel() == 0:
            return pred, target, torch.tensor(0.0, device=self.device)

        loss_pred = binary_cross_entropy(pred.clamp(1e-5, 1.0 - 1e-5), target)
        loss_ball = torch.tensor(0.0, device=self.device)
        if y_ball is not None:
            ball_pred = torch.masked_select(y_ball, sm)
            loss_ball = binary_cross_entropy(ball_pred.clamp(1e-5, 1.0 - 1e-5), target)
        loss_concept_next = torch.tensor(0.0, device=self.device)
        if y_concept_next is not None:
            concept_pred = torch.masked_select(y_concept_next, sm)
            loss_concept_next = binary_cross_entropy(concept_pred.clamp(1e-5, 1.0 - 1e-5), target)

        theta_prob = torch.sigmoid(theta)
        theta_pred = torch.masked_select(theta_prob, sm)
        loss_theta = (
            binary_cross_entropy(theta_pred.clamp(1e-5, 1.0 - 1e-5), target)
            if theta_pred.numel() > 0
            else torch.tensor(0.0, device=self.device)
        )

        rh = torch.masked_select(r_h_mean, sm)
        rd = torch.masked_select(r_d_mean, sm)
        loss_radius = torch.tensor(0.0, device=self.device)
        if rh.numel() > 0 and rd.numel() > 0:
            loss_radius = -(torch.log(rh + 1e-6).mean() + torch.log(rd + 1e-6).mean())

        conf_sel = torch.masked_select(conf, sm)
        loss_conf = torch.tensor(0.0, device=self.device)
        if conf_sel.numel() > 0:
            pred_error = torch.abs(pred.detach() - target)
            conf_target = (1.0 - pred_error).clamp(min=0.0, max=1.0)
            loss_conf = mse_loss(conf_sel.float(), conf_target.float())

        lambda_theta = float(self.other_config.get("lambda_theta", 0.1))
        lambda_radius = float(self.other_config.get("lambda_radius", 0.001))
        lambda_conf = float(self.other_config.get("lambda_conf", 0.02))
        lambda_ball = float(self.other_config.get("lambda_ball", 0.2))
        lambda_concept_next = float(self.other_config.get("lambda_concept_next", 0.2))

        loss = (
            loss_pred
            + lambda_ball * loss_ball
            + lambda_concept_next * loss_concept_next
            + lambda_theta * loss_theta
            + lambda_radius * loss_radius
            + lambda_conf * loss_conf
        )
        return pred, target, loss
