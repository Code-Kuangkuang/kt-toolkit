import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer
from models.multi_concept import concept_validity


@TRAINER_REGISTRY.register("lefokt")
@TRAINER_REGISTRY.register("lefokt_akt")
@TRAINER_REGISTRY.register("akt")
class AKTTrainer(BaseTrainer):
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
            pred, target, reg_loss, loss = self._forward_batch(batch)
            if pred.numel() == 0:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(loss.item())

            # Progress bar
            self._print_progress(batch_idx, total_batches, loss.item())


        return float(np.mean(losses)) if losses else 0.0

    def _optional(self, batch, key):
        """A sequence the loader may not have produced at all.

        `KTQueDataset.__getitem__` omits an empty tensor rather than passing it
        through, so a concept-only dataset such as statics2011 arrives with no
        `qseqs` key. Indexing `batch["qseqs"]` raised KeyError on those; the
        hand-written HD-AKT trainer had quietly fixed that for its own copy,
        which is how the difference survived unnoticed.
        """
        value = batch.get(key)
        if value is None or not value.numel():
            return None
        return value.to(self.device)

    def _forward_batch(self, batch):
        qseqs = self._optional(batch, "qseqs")
        cseqs = self._optional(batch, "cseqs")
        rseqs = batch["rseqs"].to(self.device)
        qshft = self._optional(batch, "shft_qseqs")
        cshft = self._optional(batch, "shft_cseqs")
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        q_full = self._concat_full(qseqs, qshft)
        c_full = self._concat_full(cseqs, cshft)
        r_full = self._concat_full(rseqs, rshft)

        if c_full is not None:
            q_data = c_full
            pid_data = q_full
        else:
            q_data = q_full
            pid_data = None

        if q_data is None:
            raise ValueError("AKTTrainer requires concept or question sequences.")
        target_concepts = cshft if cshft is not None else qshft
        _, target_has_concept = concept_validity(
            target_concepts, self.model.n_question
        )
        if torch.any(sm.bool() & ~target_has_concept):
            raise ValueError("AKT found a scored question without a valid concept id.")
        if pid_data is None and getattr(self.model, "n_pid", 0) > 0:
            raise ValueError("AKTTrainer requires question ids when n_pid > 0.")

        if pid_data is None:
            preds, reg_loss = self.model(q_data.long(), r_full.long())
        else:
            preds, reg_loss = self.model(q_data.long(), r_full.long(), pid_data.long())

        preds = preds[:, 1:]
        loss = cal_loss(self.model, [preds], rseqs, rshft, sm, preloss=[reg_loss] if reg_loss is not None else [])
        pred = torch.masked_select(preds, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, reg_loss, loss

    @staticmethod
    def _concat_full(seqs, shft):
        if seqs is None or seqs.numel() == 0:
            return None
        return torch.cat((seqs[:, :1], shft), dim=1)


def cal_loss(model, ys, r, rshft, sm, preloss=None):
    y = torch.masked_select(ys[0], sm)
    t = torch.masked_select(rshft, sm)
    loss = binary_cross_entropy(y.double(), t.double())
    if preloss:
        loss = loss + preloss[0]
    return loss


LEFOKTAKTTrainer = AKTTrainer
