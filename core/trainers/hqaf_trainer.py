import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("hqaf")
@TRAINER_REGISTRY.register("hqaf_kt")
class HQAFTrainer(BaseTrainer):
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
            self._print_progress(batch_idx, total_batches, loss.item())

        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch):
        qseqs = _required(batch, "qseqs").to(self.device).long()
        cseqs = _required(batch, "cseqs").to(self.device).long()
        rseqs = _required(batch, "rseqs").to(self.device).long()
        qshft = _required(batch, "shft_qseqs").to(self.device).long()
        cshft = _required(batch, "shft_cseqs").to(self.device).long()
        rshft = _required(batch, "shft_rseqs").to(self.device).float()
        sm = _required(batch, "smasks").to(self.device)

        sdseqs = _required(batch, "sdseqs").to(self.device).long()
        sdshft = _required(batch, "shft_sdseqs").to(self.device).long()
        qdseqs = _required(batch, "qdseqs").to(self.device).long()
        qdshft = _required(batch, "shft_qdseqs").to(self.device).long()
        quseqs = _required(batch, "quseqs").to(self.device).long()
        qushft = _required(batch, "shft_quseqs").to(self.device).long()
        ptseqs = _required(batch, "pTseqs").to(self.device).long()
        ptshft = _required(batch, "shft_pTseqs").to(self.device).long()

        q_full = torch.cat((qseqs[:, :1], qshft), dim=1)
        c_full = torch.cat((cseqs[:, :1], cshft), dim=1)
        r_full = torch.cat((rseqs[:, :1], rshft.long()), dim=1)
        sd_full = torch.cat((sdseqs[:, :1], sdshft), dim=1)
        qd_full = torch.cat((qdseqs[:, :1], qdshft), dim=1)
        qu_full = torch.cat((quseqs[:, :1], qushft), dim=1)
        # Do not pass observed per-interaction response time here: for the
        # prediction target at t+1 it is only known after the student answers.
        # The question-average time bucket is fitted on training folds and is
        # available before prediction, so it is safe to use as HQAF's cutT.
        safe_time_full = qu_full
        pt_full = torch.cat((ptseqs[:, :1], ptshft), dim=1)

        preds, _ = self.model(
            c_full,
            r_full,
            q_full,
            cd=sd_full,
            qd=qd_full,
            qu=qu_full,
            cutT=safe_time_full,
            cpT=pt_full,
            sdshft=sdshft,
            sm=sm,
        )
        preds = preds[:, 1:]
        if preds.shape != rshft.shape:
            raise ValueError(
                f"HQAF prediction shape {tuple(preds.shape)} does not align "
                f"with shifted targets {tuple(rshft.shape)}."
            )

        loss = _masked_bce(preds, rshft, sm)
        pred = torch.masked_select(preds, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _required(batch, key):
    value = batch.get(key)
    if value is None or (hasattr(value, "numel") and value.numel() == 0):
        raise ValueError(f"HQAF requires batch field '{key}'.")
    return value


def _masked_bce(preds, target, mask):
    y = torch.masked_select(preds.double(), mask)
    t = torch.masked_select(target.double(), mask)
    return binary_cross_entropy(y, t)
