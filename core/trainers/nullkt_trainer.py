"""Trainer for `nullkt`.

Deliberately the same shape as `SimpleKTTrainer`: same batch keys, same
`preds[:, 1:]` alignment, same float64 BCE over `smasks`. The null is only useful
if the *only* difference between its row and a model's row is the model, so the
loss and the scored positions have to be identical -- the float32/float64 split
between the HD trainers and their baselines is exactly the confound this avoids
(see core/backbone.py).
"""

import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer
from models.multi_concept import concept_validity


@TRAINER_REGISTRY.register("nullkt")
class NullKTTrainer(BaseTrainer):
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
        other_config=None,
    ):
        super().__init__(num_epochs=num_epochs, hooks=hooks, test_loader=test_loader)
        self.model = model
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.optimizer = optimizer
        self.device = device
        self.metric_key = metric_key
        self.patience = patience
        self.other_config = other_config or {}

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
        def get(key):
            value = batch.get(key)
            if value is None or value.numel() == 0:
                return None
            return value.to(self.device).long()

        qseqs, qshft = get("qseqs"), get("shft_qseqs")
        cseqs, cshft = get("cseqs"), get("shft_cseqs")
        pidseqs, pidshft = get("pidseqs"), get("shft_pidseqs")
        rseqs = batch["rseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        base_seqs = cseqs if cseqs is not None else qseqs
        base_shft = cshft if cshft is not None else qshft
        if base_seqs is None or base_shft is None:
            raise ValueError("NullKTTrainer requires question or concept sequences.")

        _, target_has_concept = concept_validity(base_shft, self.model.num_c)
        if torch.any(sm.bool() & ~target_has_concept):
            raise ValueError(
                "nullkt found a scored question without a valid concept id."
            )

        preds = self.model(
            qseqs=qseqs,
            rseqs=rseqs,
            cseqs=base_seqs,
            qshft=qshft,
            cshft=base_shft,
            rshft=rshft.long(),
            pidseqs=pidseqs,
            pidshft=pidshft,
        )
        preds = preds[:, 1:] if preds.size(1) == rshft.size(1) + 1 else preds
        if preds.shape != rshft.shape:
            raise ValueError(
                f"nullkt prediction shape {tuple(preds.shape)} does not align "
                f"with shifted targets {tuple(rshft.shape)}."
            )

        y = torch.masked_select(preds.double(), sm)
        t = torch.masked_select(rshft.double(), sm)
        loss = binary_cross_entropy(y, t)
        return y, t, loss
