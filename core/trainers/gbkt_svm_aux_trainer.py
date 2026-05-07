import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy, mse_loss

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("gbkt_svm_aux")
class GBKTSVMAuxTrainer(BaseTrainer):
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
        self.C_svm = float(other_config.get("C_svm", 1.0))
        self.C_svm_theta = float(other_config.get("C_svm_theta", 1.0))
        self.C_svm_conf = float(other_config.get("C_svm_conf", 1.0))
        self.epsilon = float(other_config.get("epsilon", 0.1))

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
    def _concat_full(seqs, shft):
        if seqs is None or seqs.numel() == 0:
            return None
        return torch.cat((seqs[:, :1], shft), dim=1)

    @staticmethod
    def _normalized_linear(linear, feature):
        margin = linear(feature).squeeze(-1)
        norm_w = torch.norm(linear.weight)
        return margin / (norm_w + 1e-8)

    def _svm_classification_loss(self, linear, feature, target, mask, c_value):
        label = torch.masked_select(target, mask).float().mul(2.0).sub(1.0)
        margin = self._normalized_linear(linear, feature)
        margin_sel = torch.masked_select(margin, mask)
        if margin_sel.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        reg = 0.5 * linear.weight.pow(2).sum()
        hinge = torch.relu(1.0 - label * margin_sel).pow(2).mean()
        return reg + c_value * hinge

    def _svm_regression_loss(self, linear, feature, target, mask, c_value):
        pred = self._normalized_linear(linear, feature)
        pred_sel = torch.masked_select(pred, mask)
        target_sel = torch.masked_select(target, mask).float()
        if pred_sel.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        reg = 0.5 * linear.weight.pow(2).sum()
        eps_loss = torch.relu(torch.abs(target_sel - pred_sel) - self.epsilon).pow(2).mean()
        return reg + c_value * eps_loss

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
            raise ValueError("GBKTSVMAuxTrainer requires both question and concept sequences.")

        outputs = self.model(q_full.long(), c_full.long(), r_full.float())
        y = outputs["y"]
        theta = outputs["theta"]
        conf = outputs["confidence"]
        r_h_mean = outputs["r_h_mean"]
        r_d_mean = outputs["r_d_mean"]
        feature_svm_shortcut = outputs["feature_svm_shortcut"]
        feature_svm_theta = outputs["feature_svm_theta"]
        feature_svm_conf = outputs["feature_svm_conf"]

        common_len = min(y.size(1), rshft.size(1), sm.size(1))
        if common_len <= 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty, torch.tensor(0.0, device=self.device)

        y = y[:, :common_len]
        theta = theta[:, :common_len]
        conf = conf[:, :common_len]
        r_h_mean = r_h_mean[:, :common_len]
        r_d_mean = r_d_mean[:, :common_len]
        feature_svm_shortcut = feature_svm_shortcut[:, :common_len, :]
        feature_svm_theta = feature_svm_theta[:, :common_len, :]
        feature_svm_conf = feature_svm_conf[:, :common_len, :]
        rshft = rshft[:, :common_len]
        sm = sm[:, :common_len]

        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        if pred.numel() == 0:
            return pred, target, torch.tensor(0.0, device=self.device)
        target = target.float()

        loss_pred = binary_cross_entropy(pred, target)

        theta_prob = torch.sigmoid(theta)
        theta_pred = torch.masked_select(theta_prob, sm)
        loss_theta = (
            binary_cross_entropy(theta_pred, target)
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
        conf_target_full = None
        if conf_sel.numel() > 0:
            pred_error = torch.abs(pred.detach() - target)
            conf_target = (1.0 - pred_error).clamp(min=0.0, max=1.0)
            loss_conf = mse_loss(conf_sel.float(), conf_target.float())
            conf_target_full = torch.zeros_like(conf)
            conf_target_full[sm] = conf_target

        model_ref = self.model.module if hasattr(self.model, "module") else self.model
        loss_svm_shortcut = self._svm_classification_loss(
            model_ref.svm_shortcut,
            feature_svm_shortcut,
            rshft,
            sm,
            self.C_svm,
        )
        loss_svm_theta = self._svm_classification_loss(
            model_ref.svm_theta,
            feature_svm_theta,
            rshft,
            sm,
            self.C_svm_theta,
        )
        loss_svm_conf = torch.tensor(0.0, device=self.device)
        if conf_target_full is not None:
            loss_svm_conf = self._svm_regression_loss(
                model_ref.svm_conf,
                feature_svm_conf,
                conf_target_full.detach(),
                sm,
                self.C_svm_conf,
            )

        lambda_theta = float(self.other_config.get("lambda_theta", 0.1))
        lambda_radius = float(self.other_config.get("lambda_radius", 0.001))
        lambda_conf = float(self.other_config.get("lambda_conf", 0.05))
        lambda_svm = float(self.other_config.get("lambda_svm", 0.01))
        lambda_svm_theta = float(self.other_config.get("lambda_svm_theta", 0.005))
        lambda_svm_conf = float(self.other_config.get("lambda_svm_conf", 0.0))

        loss = (
            loss_pred
            + lambda_theta * loss_theta
            + lambda_radius * loss_radius
            + lambda_conf * loss_conf
            + lambda_svm * loss_svm_shortcut
            + lambda_svm_theta * loss_svm_theta
            + lambda_svm_conf * loss_svm_conf
        )
        return pred, target, loss
