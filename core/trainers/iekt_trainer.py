import sys
import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy

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

        for batch_idx, batch in enumerate(self.train_loader):
            pred, target, loss = self._forward_batch(batch)
            if pred.numel() == 0:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            # Progress bar
            progress = (batch_idx + 1) / total_batches
            bar_len = 25
            filled = int(bar_len * progress)
            bar = "█" * filled + "░" * (bar_len - filled)
            pct = int(progress * 100)
            print(f"  │{bar}│ {pct:3d}% | Loss: {loss.item():.4f}", flush=True)

        # Ensure 100% is shown at the end
        if losses:
            bar = "█" * bar_len
            print(f"  │{bar}│ 100% | Loss: {losses[-1]:.4f}")
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
        if self.best_metric is None or metric > self.best_metric + 1e-3:
            self.best_metric = metric
            self.best_epoch = epoch
            return False
        if self.patience is None:
            return False
        return (epoch - self.best_epoch) >= self.patience

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

    def evaluate_test(self):
        """Evaluate model on test set."""
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
