import numpy as np
import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer
from models.multi_concept import concept_validity


@TRAINER_REGISTRY.register("simplekt")
class SimpleKTTrainer(BaseTrainer):
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
        self.lambda_item_l2 = float(self.other_config.get("lambda_item_l2", 0.0))
        if self.lambda_item_l2 < 0.0:
            raise ValueError(
                f"lambda_item_l2 must be non-negative, got {self.lambda_item_l2}."
            )

    def _item_l2_penalty(self):
        """Explicit L2 on the per-item Rasch deviation table.

        This is the control for "why not just use weight decay?".  In the linear
        case an L2 penalty on the item parameter yields an effective shrinkage of
        ``n_q/(n_q+lambda)``, i.e. the same graded form as the frequency gate,
        without counting anything.  The penalty covers every real item row on
        every step, matching what an optimizer's ``weight_decay`` would do, and
        is written as a sum so ``lambda`` keeps the scale of the classical
        derivation; sweep it logarithmically.

        Returns ``None`` when disabled so the default path is untouched.
        """
        if self.lambda_item_l2 <= 0.0:
            return None
        table = getattr(self.model, "difficult_param", None)
        if table is None:
            raise ValueError(
                "lambda_item_l2 was set but the model has no difficult_param "
                "table; this control only applies to Rasch-style backbones."
            )
        num_pid = int(getattr(self.model, "num_pid", 0))
        if num_pid <= 0:
            raise ValueError(
                "lambda_item_l2 requires num_pid > 0 (question-level data)."
            )
        # The final row is the padding slot and indexes no real item.
        return self.lambda_item_l2 * table.weight[:num_pid].square().sum()

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
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        rseqs = batch["rseqs"].to(self.device).long()
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)
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
            raise ValueError("SimpleKTTrainer requires question or concept sequences.")
        if base_shft is None or base_shft.numel() == 0:
            raise ValueError("SimpleKTTrainer requires shifted question or concept sequences.")

        base_seqs = base_seqs.to(self.device).long()
        base_shft = base_shft.to(self.device).long()
        _, target_has_concept = concept_validity(base_shft, self.model.num_c)
        if torch.any(sm.bool() & ~target_has_concept):
            raise ValueError(
                "SimpleKT found a scored question without a valid concept id."
            )

        # Prepare optional explicit problem-id sequences.
        if pidseqs is not None:
            pidseqs = pidseqs.to(self.device).long()
        if pidshft is not None:
            pidshft = pidshft.to(self.device).long()

        preds = self.model(
            qseqs=qseqs.to(self.device).long() if qseqs is not None else None,
            rseqs=rseqs,
            cseqs=base_seqs,
            qshft=qshft.to(self.device).long() if qshft is not None else None,
            cshft=base_shft,
            rshft=rshft,
            pidseqs=pidseqs,
            pidshft=pidshft,
        )
        preds_for_loss = preds[:, 1:] if preds.size(1) == rshft.size(1) + 1 else preds
        if preds_for_loss.shape != rshft.shape:
            raise ValueError(
                f"SimpleKT prediction shape {tuple(preds.shape)} does not align "
                f"with shifted targets {tuple(rshft.shape)}."
            )

        loss = cal_loss(preds_for_loss, rshft, sm)
        penalty = self._item_l2_penalty()
        if penalty is not None:
            loss = loss + penalty
        pred = torch.masked_select(preds_for_loss, sm)
        target = torch.masked_select(rshft, sm)
        return pred, target, loss


def cal_loss(preds, rshft, sm):
    y = torch.masked_select(preds.double(), sm)
    t = torch.masked_select(rshft.double(), sm)
    loss = binary_cross_entropy(y, t)
    return loss
