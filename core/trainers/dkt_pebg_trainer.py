import numpy as np
import torch
from sklearn import metrics
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("dkt_pebg")
class DKTPEBGTrainer(BaseTrainer):
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
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")

        if qseqs is not None and qseqs.numel() > 0:
            qseqs = qseqs.to(self.device).long()
        if cseqs is not None and cseqs.numel() > 0:
            cseqs = cseqs.to(self.device).long()
        if qshft is not None and qshft.numel() > 0:
            qshft = qshft.to(self.device).long()
        if cshft is not None and cshft.numel() > 0:
            cshft = cshft.to(self.device).long()

        rseqs = batch["rseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        uses_binary_target = bool(getattr(self.model, "uses_binary_target", False))
        if uses_binary_target:
            if qseqs is None or qseqs.numel() == 0:
                raise ValueError("Original PEBG-DKT mode requires qseqs.")
            if qshft is None or qshft.numel() == 0:
                raise ValueError("Original PEBG-DKT mode requires shft_qseqs.")

            y = self.model(qseqs, rseqs, q_next=qshft)
            pred = torch.masked_select(y, sm)
            target = torch.masked_select(rshft, sm)
            loss = binary_cross_entropy(pred.double(), target.double())
            return pred, target, loss

        use_question_inputs = bool(getattr(self.model, "use_question_inputs", False))
        if use_question_inputs and qseqs is not None and qseqs.numel() > 0:
            base_seqs = qseqs
        elif cseqs is not None and cseqs.numel() > 0:
            base_seqs = cseqs
        elif qseqs is not None and qseqs.numel() > 0:
            base_seqs = qseqs
        else:
            raise ValueError("DKTPEBGTrainer requires qseqs or cseqs.")

        if cshft is not None and cshft.numel() > 0:
            target_idx = cshft
        elif qshft is not None and qshft.numel() > 0:
            target_idx = qshft
        else:
            raise ValueError("DKTPEBGTrainer requires shft_cseqs or shft_qseqs.")

        y = self.model(base_seqs, rseqs)

        max_target = int(target_idx.max().item()) if target_idx.numel() > 0 else -1
        if max_target >= y.size(-1):
            raise ValueError(
                f"DKT target id {max_target} exceeds output dim {y.size(-1)}. "
                "Please provide compatible shft_cseqs/shft_qseqs for current model output."
            )

        y = y.gather(-1, target_idx.unsqueeze(-1)).squeeze(-1)

        loss = cal_loss(self.model, [y], rseqs, rshft, sm)

        pred = torch.masked_select(y, sm)
        target = torch.masked_select(rshft, sm)
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


def cal_loss(model, ys, r, rshft, sm, preloss=None):
    y = torch.masked_select(ys[0], sm)
    t = torch.masked_select(rshft, sm)
    loss = binary_cross_entropy(y.double(), t.double())
    return loss