import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, mse_loss

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("gbsvkt_fix")
class GBSVKTFixTrainer(BaseTrainer):
    """Trainer for GBSVKTFix model.

    Fixes from original GBSVKTTrainer:
    1. Uses nn.Linear layers instead of bare Parameters for W_* layers
    2. SVM-style loss functions operate on raw normalized margins (no pre-sigmoid)
    3. Consistent normalized margin computation across all losses
    """

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
        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience
        self.other_config = other_config
        self.C1 = float(other_config.get("C1", 1.0))
        self.C2 = float(other_config.get("C2", 1.0))
        self.C4 = float(other_config.get("C4", 1.0))
        self.epsilon = float(other_config.get("epsilon", 0.01))

        # LR schedule: linear warmup then cosine decay.
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
                eta_min=1e-6,
            )
            self.scheduler = SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )

    @staticmethod
    def _normalized_margin(y, w, b):
        """Compute SVM-style normalized margin: (w*x + b) / ||w||"""
        norm_w = torch.norm(w)
        return (torch.matmul(y, w) + b) / (norm_w + 1e-8)

    def cal_loss_shortcut(self, w, b, y, t, sm):
        """SVM-style hinge loss for shortcut prediction.

        y: shortcut_input features [B, T, 4*d_p]
        t: target response [B, T] in {0, 1}
        """
        t = t * 2 - 1  # convert to {-1, +1}

        # Compute normalized SVM margin
        wxeb = self._normalized_margin(y, w, b)  # [B, T]
        wxeb_masked = torch.masked_select(wxeb, sm)

        # Hinge loss: 0.5*||w||^2 + C * max(0, 1 - t * (w*x + b))^2
        loss = 0.5 * w.pow(2).sum() + self.C1 * torch.relu(1 - t * wxeb_masked).pow(2).mean()

        # For evaluation: convert margin to probability via sigmoid
        pred = torch.sigmoid(wxeb)

        return loss, pred

    def cal_loss_theta(self, w, b, y, t, sm):
        """SVM-style loss for theta (ability parameter).

        y: effective_diff features [B, T, d_p]
        t: target response [B, T] in {0, 1}
        """
        t = t * 2 - 1  # convert to {-1, +1}

        # Compute normalized SVM margin
        wxeb = self._normalized_margin(y, w, b)  # [B, T]
        wxeb_masked = torch.masked_select(wxeb, sm)

        # Hinge loss for theta
        loss = 0.5 * w.pow(2).sum() + self.C2 * torch.relu(1 - t * wxeb_masked).pow(2).mean()

        # For evaluation: convert margin to probability via sigmoid
        pred = torch.sigmoid(wxeb)

        return loss, pred

    def cal_loss_conf(self, w, b, y, t, sm):
        """Epsilon-insensitive loss for confidence estimation.

        y: effective_diff features [B, T, d_p]
        t: confidence target [B, T] in [0, 1]
        """
        # Compute normalized margin
        wxeb = self._normalized_margin(y, w, b)  # [B, T]
        wxeb_masked = torch.masked_select(wxeb, sm)
        t_masked = torch.masked_select(t, sm)

        # Epsilon-insensitive loss: 0.5*||w||^2 + C * max(0, |t - (w*x + b)| - epsilon)^2
        loss = 0.5 * w.pow(2).sum() + self.C4 * torch.relu(
            torch.abs(t_masked - wxeb_masked) - self.epsilon
        ).pow(2).mean()

        return loss, wxeb

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

        q_full = self._concat_full(qseqs, qshft)
        c_full = self._concat_full(cseqs, cshft)
        r_full = self._concat_full(rseqs, rshft)

        if q_full is None or c_full is None:
            raise ValueError("GBSVKTFixTrainer requires both question and concept sequences.")

        outputs = self.model(q_full.long(), c_full.long(), r_full.float())
        y = outputs["y"]
        conf = outputs["confidence"]
        r_h_mean = outputs["r_h_mean"]
        r_d_mean = outputs["r_d_mean"]
        feature1 = outputs["feature1"]
        feature2 = outputs["feature2"]
        feature4 = outputs["feature4"]

        common_len = min(y.size(1), rshft.size(1), sm.size(1))
        if common_len <= 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty, torch.tensor(0.0, device=self.device)

        feature1 = feature1[:, :, :common_len]
        feature2 = feature2[:, :, :common_len]
        feature4 = feature4[:, :, :common_len]
        conf = conf[:, :common_len]
        r_h_mean = r_h_mean[:, :common_len]
        r_d_mean = r_d_mean[:, :common_len]
        rshft = rshft[:, :common_len]
        sm = sm[:, :common_len]

        target = torch.masked_select(rshft, sm)
        target = target.float()

        # Get model layers
        model_ref = self.model.module if hasattr(self.model, "module") else self.model

        # Loss for shortcut prediction
        loss_shortcut, pred_shortcut = self.cal_loss_shortcut(
            model_ref.W_shortcut.weight,
            model_ref.b_shortcut,
            feature1.permute(0, 2, 1),  # [B, T, 4*d_p] -> [B, 4*d_p, T]
            target,
            sm
        )

        if pred_shortcut.numel() == 0:
            return pred_shortcut, target, torch.tensor(0.0, device=self.device)

        # Loss for theta (using raw theta margin)
        loss_theta, pred_theta = self.cal_loss_theta(
            model_ref.W_theta.weight,
            model_ref.b_theta,
            feature2.permute(0, 2, 1),  # [B, T, d_p] -> [B, d_p, T]
            target,
            sm
        )
        if pred_theta.numel() <= 0:
            loss_theta = torch.tensor(0.0, device=self.device)

        # Radius loss (log-barrier to prevent collapse)
        rh = torch.masked_select(r_h_mean, sm)
        rd = torch.masked_select(r_d_mean, sm)
        loss_radius = torch.tensor(0.0, device=self.device)
        if rh.numel() > 0 and rd.numel() > 0:
            loss_radius = -(torch.log(rh + 1e-6).mean() + torch.log(rd + 1e-6).mean())

        # Confidence loss
        conf_sel = torch.masked_select(conf, sm)
        loss_conf = torch.tensor(0.0, device=self.device)
        if conf_sel.numel() > 0 and feature4.numel() > 0:
            pred_error = torch.abs(pred_shortcut.detach() - target)
            conf_target = (1.0 - pred_error).clamp(min=0.0, max=1.0)
            loss_conf, pred_conf = self.cal_loss_conf(
                model_ref.W_conf.weight,
                model_ref.b_conf,
                feature4.permute(0, 2, 1),
                conf_target,
                sm
            )

        lambda_theta = float(self.other_config.get("lambda_theta", 0.3))
        lambda_radius = float(self.other_config.get("lambda_radius", 0.001))
        lambda_conf = float(self.other_config.get("lambda_conf", 0.05))

        loss = loss_shortcut + lambda_theta * loss_theta + lambda_radius * loss_radius + lambda_conf * loss_conf
        return pred_shortcut, target, loss
