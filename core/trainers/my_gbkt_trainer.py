import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy, mse_loss

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("gbkt")
class GBKTTrainer(BaseTrainer):
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

        # --- LR Scheduler: linear warmup + cosine decay ---
        warmup_epochs = int(other_config.get("warmup_epochs", 3))
        use_scheduler = bool(other_config.get("use_scheduler", True))
        self.scheduler = None
        if use_scheduler:
            from torch.optim.lr_scheduler import (
                CosineAnnealingLR,
                LinearLR,
                SequentialLR,
            )

            warmup = LinearLR(
                optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
            )
            cosine = CosineAnnealingLR(
                optimizer,
                T_max=max(num_epochs - warmup_epochs, 1),
                eta_min=1e-6,
            )
            self.scheduler = SequentialLR(
                optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
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
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            losses.append(loss.item())
            self._print_progress(batch_idx, total_batches, loss.item())

        if losses:
            bar = "\u2588" * 25
            print(f"  \u2502{bar}\u2502 100% | Loss: {losses[-1]:.4f}")

        # Step LR scheduler per epoch.
        if self.scheduler is not None:
            self.scheduler.step()

        return float(np.mean(losses)) if losses else 0.0

    def _eval_epoch(self, epoch):
        self.model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for batch in self.valid_loader:
                pred, target, _ = self._forward_batch(batch)
                if pred.numel() == 0:
                    continue
                y_score.append(pred.detach().cpu().numpy())
                y_true.append(target.detach().cpu().numpy())

        if not y_true:
            return {"valid_auc": -1, "valid_acc": -1}

        ts = np.concatenate(y_true, axis=0)
        ps = np.concatenate(y_score, axis=0)
        try:
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        except Exception:
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        return {"valid_auc": auc, "valid_acc": acc}

    def _should_stop(self, epoch, metrics_dict):
        metric = metrics_dict.get(self.metric_key, None)
        if metric is None:
            return False
        # Changed: 1e-4 instead of 1e-3, avoid premature stopping.
        if self.best_metric is None or metric > self.best_metric + 1e-4:
            self.best_metric = metric
            self.best_epoch = epoch
            return False
        if self.patience is None:
            return False
        return (epoch - self.best_epoch) >= self.patience

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
            raise ValueError("GBKTTrainer requires both question and concept sequences.")

        outputs = self.model(q_full.long(), c_full.long(), r_full.float())
        y = outputs["y"]
        theta = outputs["theta"]
        conf = outputs["confidence"]
        r_h_mean = outputs["r_h_mean"]
        r_d_mean = outputs["r_d_mean"]

        common_len = min(y.size(1), rshft.size(1), sm.size(1))
        if common_len <= 0:
            empty = torch.empty(0, device=self.device)
            return empty, empty, torch.tensor(0.0, device=self.device)

        y = y[:, :common_len]
        theta = theta[:, :common_len]
        conf = conf[:, :common_len]
        r_h_mean = r_h_mean[:, :common_len]
        r_d_mean = r_d_mean[:, :common_len]
        rshft = rshft[:, :common_len]
        sm = sm[:, :common_len]

        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
        if pred.numel() == 0:
            return pred, target, torch.tensor(0.0, device=self.device)
        target = target.float()

        # ---- L_pred: main prediction BCE ----
        loss_pred = binary_cross_entropy(pred, target)

        # ---- L_theta: IRT ability supervision ----
        theta_prob = torch.sigmoid(theta)
        theta_pred = torch.masked_select(theta_prob, sm)
        loss_theta = (
            binary_cross_entropy(theta_pred, target)
            if theta_pred.numel() > 0
            else torch.tensor(0.0, device=self.device)
        )

        # ---- L_radius: log-barrier regularization ----
        # CHANGED: L2 -> log-barrier.
        # L2 pushes radius toward 0, killing uncertainty modeling.
        # Log-barrier only penalizes when radius approaches 0,
        # allowing the model to learn meaningful radius values.
        rh = torch.masked_select(r_h_mean, sm)
        rd = torch.masked_select(r_d_mean, sm)
        loss_radius = torch.tensor(0.0, device=self.device)
        if rh.numel() > 0 and rd.numel() > 0:
            loss_radius = -(torch.log(rh + 1e-6).mean() + torch.log(rd + 1e-6).mean())

        # ---- L_conf: confidence calibration ----
        conf_sel = torch.masked_select(conf, sm)
        loss_conf = torch.tensor(0.0, device=self.device)
        if conf_sel.numel() > 0:
            pred_error = torch.abs(pred.detach() - target)
            conf_target = (1.0 - pred_error).clamp(min=0.0, max=1.0)
            loss_conf = mse_loss(conf_sel.float(), conf_target.float())

        # CHANGED defaults: lambda_theta 0.1 -> 0.3, lambda_radius kept 0.001.
        lambda_theta = float(self.other_config.get("lambda_theta", 0.3))
        lambda_radius = float(self.other_config.get("lambda_radius", 0.001))
        lambda_conf = float(self.other_config.get("lambda_conf", 0.05))

        loss = (
            loss_pred
            + lambda_theta * loss_theta
            + lambda_radius * loss_radius
            + lambda_conf * loss_conf
        )
        return pred, target, loss

    def evaluate_test(self):
        if self.test_loader is None:
            return {"test_auc": -1, "test_acc": -1}

        self.model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for batch in self.test_loader:
                pred, target, _ = self._forward_batch(batch)
                if pred.numel() == 0:
                    continue
                y_score.append(pred.detach().cpu().numpy())
                y_true.append(target.detach().cpu().numpy())

        if not y_true:
            return {"test_auc": -1, "test_acc": -1}

        ts = np.concatenate(y_true, axis=0)
        ps = np.concatenate(y_score, axis=0)
        try:
            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        except Exception:
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        acc = metrics.accuracy_score(ts, prelabels)
        return {"test_auc": auc, "test_acc": acc}
