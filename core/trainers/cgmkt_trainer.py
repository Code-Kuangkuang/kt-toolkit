"""Trainer adapter for the CGMKT port.

CGMKT's `forward` already takes the current and the next step separately --
`(last_pro, last_ans, last_skill, next_pro, next_skill)` -- so unlike the MoC-KT
family there is no full-sequence reassembly here; the batch's shifted tensors go
straight in and `y` comes back aligned with `shft_rseqs` position for position.

The one adaptation that matters is padding. This repo pads `qseqs` with `-1`
(AGENTS.md: `0` is a valid question id, so validity is read from the masks), and
`Questions_Embedding` looks the ids up with `F.embedding`, which rejects a
negative index. Upstream never hits this because pyKT hands it a differently
padded tensor. Question ids are therefore clamped to 0 here.

Clamping is safe rather than merely convenient:

  * padded positions are excluded from the loss and the metric by `smasks`, so
    the substituted embedding never reaches a reported number;
  * padding is trailing, so a contaminated step cannot influence an earlier
    prediction -- the direction the causality contract test checks;
  * the mastery state is separately frozen at invalid steps by the model's own
    `valid` gate, so the substitution does not leak into `m_t` either.

Concept ids are left at `-1` on purpose: `get_kc_embedding` and
`_concept_group_membership` both mask on `>= 0` internally, and clamping them
here would make a padded slot look like concept 0 to that mask.
"""

import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


@TRAINER_REGISTRY.register("cgmkt")
class CGMKTTrainer(BaseTrainer):
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
        for key in ("qseqs", "cseqs", "rseqs", "shft_qseqs", "shft_cseqs", "shft_rseqs"):
            if batch.get(key) is None or batch[key].numel() == 0:
                raise ValueError(
                    f"CGMKT requires {key}; it is an all_in_one multi-concept model, "
                    f"so the dataset needs both question and concept ids."
                )

        # -1 padding would index out of range in F.embedding; see module docstring.
        last_pro = batch["qseqs"].to(self.device).long().clamp(min=0)   # [B, T]
        next_pro = batch["shft_qseqs"].to(self.device).long().clamp(min=0)
        # Concepts keep their -1: the model masks on >= 0 itself.
        last_skill = batch["cseqs"].to(self.device).long()              # [B, T, K]
        next_skill = batch["shft_cseqs"].to(self.device).long()
        last_ans = batch["rseqs"].to(self.device).float()               # [B, T]

        rshft = batch["shft_rseqs"].to(self.device).float()
        sm = batch["smasks"].to(self.device)

        preds, reg_loss = self.model(last_pro, last_ans, last_skill, next_pro, next_skill)
        if preds.shape != rshft.shape:
            raise ValueError(
                f"CGMKT prediction shape {tuple(preds.shape)} does not align with "
                f"shifted targets {tuple(rshft.shape)}."
            )

        y = torch.masked_select(preds.double(), sm)
        t = torch.masked_select(rshft.double(), sm)
        loss = binary_cross_entropy(y, t)
        if reg_loss is not None:
            loss = loss + reg_loss
        return y, t, loss
