import numpy as np
import torch

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer
from models.multi_concept import pool_concept_predictions


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


        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch, with_loss=True):
        cseqs = batch["cseqs"].to(self.device).long()
        rseqs = batch["rseqs"].to(self.device).long()
        cshft = batch["shft_cseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        y = self.model(cseqs, rseqs)

        # A plain gather breaks on [B,T,K] concepts and would force the dataset
        # to truncate to KC 1.  Pooling the per-KC predictions instead averages
        # over the question's KCs, which is what DKT does here and what pykt
        # does in qikt.py.
        y_next, target_has_concept = pool_concept_predictions(y, cshft, y.size(-1))
        y_curr, _ = pool_concept_predictions(y, cseqs, y.size(-1))
        if torch.any(sm.bool() & ~target_has_concept):
            raise ValueError("DKT+ found a scored question without a valid concept id.")

        pred = torch.masked_select(y_next, sm)
        target = torch.masked_select(rshft, sm)

        if not with_loss:
            return pred, target, None

        loss = cal_loss(self.model, y_next, y_curr, y, rseqs, rshft, sm)
        return pred, target, loss


def cal_loss(model, y_next, y_curr, y_full, rseqs, rshft, sm):

    y_next_masked = torch.masked_select(y_next, sm)
    rshft_masked = torch.masked_select(rshft, sm)
    loss = torch.nn.functional.binary_cross_entropy(y_next_masked.double(), rshft_masked.double())


    y_curr_masked = torch.masked_select(y_curr, sm)
    rseqs_masked = torch.masked_select(rseqs.float(), sm)
    loss_r = torch.nn.functional.binary_cross_entropy(y_curr_masked.double(), rseqs_masked.double())


    diff = y_full[:, 1:] - y_full[:, :-1]
    loss_w1 = torch.masked_select(
        torch.norm(diff, p=1, dim=-1), sm[:, 1:]
    ).mean() / model.num_c
    
    loss_w2 = torch.masked_select(
        torch.norm(diff, p=2, dim=-1) ** 2, sm[:, 1:]
    ).mean() / model.num_c


    total_loss = (
        loss
        + model.lambda_r * loss_r
        + model.lambda_w1 * loss_w1
        + model.lambda_w2 * loss_w2
    )
    return total_loss
