"""Trainer adapter for the FA-KT port.

Identical in shape to core/trainers/mtkt_trainer.py -- both models take
`(dcur, dgaps)` and return a tuple whose first element is the prediction, so the
gap features are assembled the same way core/trainers/dkt_forget_trainer.py
assembles them.

One difference from MTKT: FA-KT's `forward` returns `(preds, y2, y3)` when
`train=True` and `preds` alone otherwise, where `y2`/`y3` are the auxiliary
outputs upstream leaves hard-coded at 0. They carry no gradient, so there is no
auxiliary loss term to add here -- taking `result[0]` is the whole story, and
this note exists so that a future upstream diff that starts populating them is
noticed rather than silently dropped.
"""

import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer

GAP_KEYS = ("rgaps", "sgaps", "pcounts", "shft_rgaps", "shft_sgaps", "shft_pcounts")


@TRAINER_REGISTRY.register("fakt")
class FAKTTrainer(BaseTrainer):
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

    training_forward_kwargs = {'train': True}

    def _forward_batch(self, batch, train=False):
        dcur = {
            k: v.to(self.device) for k, v in batch.items() if torch.is_tensor(v)
        }
        missing = [k for k in GAP_KEYS if k not in dcur]
        if missing:
            raise ValueError(
                f"FA-KT requires the gap features {missing}; the dataloader was "
                f"built without include_dkt_forget=True. A dataset with no "
                f"`timestamps` column cannot supply them at all."
            )
        dgaps = {k: dcur[k] for k in GAP_KEYS}

        rshft = dcur["shft_rseqs"].float()
        sm = dcur["smasks"].bool()

        result = self.model(dcur, dgaps, train=train)
        preds = result[0] if isinstance(result, tuple) else result
        preds_for_loss = preds[:, 1:] if preds.size(1) == rshft.size(1) + 1 else preds
        if preds_for_loss.shape != rshft.shape:
            raise ValueError(
                f"FA-KT prediction shape {tuple(preds.shape)} does not align with "
                f"shifted targets {tuple(rshft.shape)}."
            )

        pred = torch.masked_select(preds_for_loss, sm)
        target = torch.masked_select(rshft, sm)
        if pred.numel() == 0:
            return pred, target, preds.sum() * 0.0
        loss = binary_cross_entropy(pred.double(), target.double())
        if not torch.isfinite(loss):
            raise FloatingPointError("FA-KT produced a NaN/Inf loss.")
        return pred, target, loss
