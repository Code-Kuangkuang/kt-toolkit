import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("hawkes")
class HawkesTrainer(BaseTrainer):
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

        # Show final
        if losses:
            bar = "█" * 25
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
        # Hawkes needs: skills, problems, times, labels
        cseqs = batch.get("cseqs")  # skills
        qseqs = batch.get("qseqs")  # problems
        rseqs = batch["rseqs"].to(self.device)  # labels/responses
        tseqs = batch.get("tseqs")  # timestamps
        cshft = batch.get("shft_cseqs")
        qshft = batch.get("shft_qseqs")
        rshft = batch["shft_rseqs"].to(self.device)
        tshft = batch.get("shft_tseqs")
        sm = batch["smasks"].to(self.device)

        # Move to device
        if cseqs is not None:
            cseqs = cseqs.to(self.device)
        if qseqs is not None:
            qseqs = qseqs.to(self.device)
        if tseqs is not None:
            tseqs = tseqs.to(self.device)
        if cshft is not None:
            cshft = cshft.to(self.device)
        if qshft is not None:
            qshft = qshft.to(self.device)
        if tshft is not None:
            tshft = tshft.to(self.device)

        # Build full sequences (prepend first element) - same as pykt
        skills = torch.cat((cseqs[:, 0:1], cshft), dim=1) if cseqs is not None else None
        problems = torch.cat((qseqs[:, 0:1], qshft), dim=1) if qseqs is not None else None
        times = torch.cat((tseqs[:, 0:1], tshft), dim=1) if tseqs is not None else None
        labels = torch.cat((rseqs[:, 0:1], rshft), dim=1)

        # Forward
        predictions = self.model(skills, problems, times, labels)

        # Use predictions from position 1 onwards (same as pykt: y[:, 1:])
        predictions = predictions[:, 1:]

        # Compute loss with clamping to avoid extreme values
        pred = torch.masked_select(predictions, sm)
        target = torch.masked_select(rshft, sm)
        # Clamp predictions to avoid extreme BCE values (same as pykt behavior)
        pred_clamped = pred.clamp(min=1e-7, max=1 - 1e-7)
        loss = binary_cross_entropy(pred_clamped.double(), target.double())

        # Debug: show loss per element
        # element_loss = -target * torch.log(pred + 1e-10) - (1-target) * torch.log(1-pred + 1e-10)
        # print(f"Element loss stats: mean={element_loss.mean():.4f}, max={element_loss.max():.4f}")

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
