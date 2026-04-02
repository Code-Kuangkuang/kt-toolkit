import numpy as np
import torch
from torch.nn.functional import one_hot

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("dkt+")
class DKTPlusTrainer(BaseTrainer):
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
            pred, target, loss = self._forward_batch(batch, with_loss=True)
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
                pred, target, _ = self._forward_batch(batch, with_loss=False)
                if pred.numel() == 0:
                    continue
                y_score.append(pred.detach().cpu().numpy())
                y_true.append(target.detach().cpu().numpy())

        if not y_true:
            return {"valid_auc": -1, "valid_acc": -1}

        ts = np.concatenate(y_true, axis=0)
        ps = np.concatenate(y_score, axis=0)
        try:
            auc = float(torch.tensor(0.0))
            from sklearn import metrics

            auc = metrics.roc_auc_score(y_true=ts, y_score=ps)
        except Exception:
            auc = -1
        prelabels = [1 if p >= 0.5 else 0 for p in ps]
        from sklearn import metrics

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

    def _forward_batch(self, batch, with_loss=True):
        cseqs = batch["cseqs"].to(self.device).long()
        rseqs = batch["rseqs"].to(self.device).long()
        cshft = batch["shft_cseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        y = self.model(cseqs, rseqs)
        
        # 💡 优化：用 gather 替代 one_hot，省显存提速度
        y_next = y.gather(-1, cshft.unsqueeze(-1)).squeeze(-1)
        y_curr = y.gather(-1, cseqs.unsqueeze(-1)).squeeze(-1)

        pred = torch.masked_select(y_next, sm)
        target = torch.masked_select(rshft, sm)

        if not with_loss:
            return pred, target, None

        loss = cal_loss(self.model, y_next, y_curr, y, rseqs, rshft, sm)
        return pred, target, loss


def cal_loss(model, y_next, y_curr, y_full, rseqs, rshft, sm):
    # ✅ 1. 主干预测 Loss: 必须先用 sm 过滤！
    y_next_masked = torch.masked_select(y_next, sm)
    rshft_masked = torch.masked_select(rshft, sm)
    loss = torch.nn.functional.binary_cross_entropy(y_next_masked.double(), rshft_masked.double())

    # ✅ 2. 重建预测 Loss (L_r): 同样需要过滤！
    y_curr_masked = torch.masked_select(y_curr, sm)
    rseqs_masked = torch.masked_select(rseqs.float(), sm)
    loss_r = torch.nn.functional.binary_cross_entropy(y_curr_masked.double(), rseqs_masked.double())

    # ✅ 3. 平滑度惩罚 Loss (Waviness): 你原来写的这段是完全正确的！
    diff = y_full[:, 1:] - y_full[:, :-1]
    loss_w1 = torch.masked_select(
        torch.norm(diff, p=1, dim=-1), sm[:, 1:]
    ).mean() / model.num_c
    
    loss_w2 = torch.masked_select(
        torch.norm(diff, p=2, dim=-1) ** 2, sm[:, 1:]
    ).mean() / model.num_c

    # 4. 加权求和
    total_loss = (
        loss
        + model.lambda_r * loss_r
        + model.lambda_w1 * loss_w1
        + model.lambda_w2 * loss_w2
    )
    return total_loss

    def evaluate_test(self):
        """Evaluate model on test set."""
        if self.test_loader is None:
            return {"test_auc": -1, "test_acc": -1}

        from sklearn import metrics
        self.model.eval()
        y_true = []
        y_score = []
        with torch.no_grad():
            for batch in self.test_loader:
                pred, target, _ = self._forward_batch(batch, with_loss=False)
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
