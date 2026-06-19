import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("ukt")
class UKTTrainer(BaseTrainer):
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
        cl_weight=0.02,
    ):
        super().__init__(num_epochs=num_epochs, hooks=hooks, test_loader=test_loader)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience
        self.cl_weight = cl_weight

    def _train_epoch(self, epoch):
        self.model.train()
        losses = []
        total_batches = len(self.train_loader)

        print(f"\n== Epoch {epoch}/{self.num_epochs} ==")
        print("=" * 50)

        for batch_idx, batch in enumerate(self.train_loader):
            pred, target, loss = self._forward_batch(batch, train=True)
            if pred.numel() == 0:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())
            self._print_progress(batch_idx, total_batches, loss.item())


        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch, train=False):
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"].to(self.device).long()
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)
        masks = batch.get("masks")
        if masks is not None:
            masks = masks.to(self.device)
        pidseqs = batch.get("pidseqs")
        pidshft = batch.get("shft_pidseqs")

        qseqs = qseqs if qseqs is not None and qseqs.numel() > 0 else None
        qshft = qshft if qshft is not None and qshft.numel() > 0 else None
        cseqs = cseqs if cseqs is not None and cseqs.numel() > 0 else None
        cshft = cshft if cshft is not None and cshft.numel() > 0 else None
        pidseqs = pidseqs if pidseqs is not None and pidseqs.numel() > 0 else None
        pidshft = pidshft if pidshft is not None and pidshft.numel() > 0 else None

        base_seqs = cseqs if cseqs is not None and cseqs.numel() > 0 else qseqs
        base_shft = cshft if cshft is not None and cshft.numel() > 0 else qshft

        if base_seqs is None or base_seqs.numel() == 0:
            raise ValueError("UKTTrainer requires question or concept sequences.")
        if base_shft is None or base_shft.numel() == 0:
            raise ValueError("UKTTrainer requires shifted question or concept sequences.")

        base_seqs = base_seqs.to(self.device).long()
        base_shft = base_shft.to(self.device).long()

        if pidseqs is not None:
            pidseqs = pidseqs.to(self.device).long()
        if pidshft is not None:
            pidshft = pidshft.to(self.device).long()

        if train and self.model.use_CL and self.model.use_uncertainty_aug:
            shft_r_aug = batch.get("shft_r_aug")
            r_aug = batch.get("r_aug")
            if shft_r_aug is not None:
                shft_r_aug = shft_r_aug.to(self.device).long()
            if r_aug is not None:
                r_aug = r_aug.to(self.device).long()
        else:
            shft_r_aug = None
            r_aug = None

        result = self.model(
            qseqs=qseqs.to(self.device).long() if qseqs is not None else None,
            rseqs=rseqs,
            cseqs=base_seqs,
            qshft=qshft.to(self.device).long() if qshft is not None else None,
            cshft=base_shft,
            rshft=rshft,
            pidseqs=pidseqs,
            pidshft=pidshft,
            masks=masks,
            train=train,
            shft_r_aug=shft_r_aug,
            r_aug=r_aug,
        )

        if train and self.model.use_CL:
            preds, cl_loss, temp = result
            preds_for_loss = _align_shifted_preds(preds, rshft)
            bce_loss = cal_loss(preds_for_loss, rshft, sm)
            loss = bce_loss + getattr(self.model, "cl_weight", self.cl_weight) * cl_loss
        else:
            preds = result
            preds_for_loss = _align_shifted_preds(preds, rshft)
            loss = cal_loss(preds_for_loss, rshft, sm)

        pred = torch.masked_select(preds_for_loss, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def _align_shifted_preds(preds, rshft):
    preds_for_loss = preds[:, 1:] if preds.size(1) == rshft.size(1) + 1 else preds
    if preds_for_loss.shape != rshft.shape:
        raise ValueError(
            f"UKT prediction shape {tuple(preds.shape)} does not align "
            f"with shifted targets {tuple(rshft.shape)}."
        )
    return preds_for_loss


def cal_loss(preds, rshft, sm):
    y = torch.masked_select(preds.double(), sm)
    t = torch.masked_select(rshft.double(), sm)
    loss = binary_cross_entropy(y, t)
    return loss
