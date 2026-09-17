import numpy as np
import torch
from torch.autograd import grad
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer
from models.multi_concept import pool_concept_predictions


def _l2_normalize_adv(d):
    """Per-sequence L2 normalisation of the adversarial direction.

    pykt's version round-trips through numpy (`d.cpu().numpy()`, divide by the
    norm over axes (1, 2), `torch.from_numpy`).  This is the same arithmetic
    kept on-device, which also keeps the result's dtype and device matching the
    features it will be added to.
    """
    norm = d.pow(2).sum(dim=(1, 2), keepdim=True).sqrt()
    return d / (norm + 1e-16)


@TRAINER_REGISTRY.register("atkt")
class ATKTTrainer(BaseTrainer):
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


        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch):
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"].to(self.device).long()
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        if cseqs is None or cseqs.numel() == 0:
            raise ValueError("ATKTTrainer requires concept sequences.")

        cseqs = cseqs.to(self.device).long()
        if cshft is not None:
            cshft = cshft.to(self.device).long()

        target_idx = cshft if cshft is not None else cseqs

        # ATKT's contribution IS the adversarial step: a clean pass, then a
        # second pass with an epsilon-sized perturbation pointing along the
        # gradient of the clean loss w.r.t. the interaction features, and the
        # two losses summed with weight beta (pykt train_model.py, "atkt"
        # branch).  Without it this model is a plain attention-LSTM and
        # model.epsilon / model.beta are dead parameters.
        preds, features = self.model(cseqs, rseqs)
        # A plain gather breaks on [B,T,K] concepts; pooling the per-KC
        # predictions averages over the question's KCs, as pykt does in qikt.py.
        preds, target_has_concept = pool_concept_predictions(
            preds, target_idx, preds.size(-1)
        )
        if torch.any(sm.bool() & ~target_has_concept):
            raise ValueError("ATKT found a scored question without a valid concept id.")
        loss = cal_loss(self.model, [preds], rseqs, rshft, sm)

        if self.model.training:
            features_grad = grad(loss, features, retain_graph=True)[0]
            p_adv = (self.model.epsilon * _l2_normalize_adv(features_grad.detach())).detach()
            adv_preds, _ = self.model(cseqs, rseqs, p_adv)
            adv_preds, _ = pool_concept_predictions(
                adv_preds, target_idx, adv_preds.size(-1)
            )
            adv_loss = cal_loss(self.model, [adv_preds], rseqs, rshft, sm)
            loss = loss + self.model.beta * adv_loss

        pred = torch.masked_select(preds, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def cal_loss(model, ys, r, rshft, sm, preloss=None):
    y = torch.masked_select(ys[0], sm)
    t = torch.masked_select(rshft, sm)
    loss = binary_cross_entropy(y.double(), t.double())
    return loss
