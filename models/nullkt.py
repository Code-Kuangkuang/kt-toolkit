"""The row every table in this repository has been missing: no sequence model.

`experiment/baseline_table.md` uses `dkt` as its floor, but DKT is a model, not a
null.  Every delta measured against it -- HD-KT's three pairs, the item-parameter
sweep, the eight ported models when they are finally run -- is a difference
between two sequence models, with no statement of what a run that does no
sequence modelling at all would score.  Without that, "AKT beats DKT by 0.020"
has a numerator and no denominator.

This model is that denominator.  Every feature it computes is a **count or a mean
over the prefix**, so it is order-invariant by construction: shuffling a
student's history before position `t` cannot change the prediction at `t`.
`tests/test_nullkt_is_order_invariant.py` pins that, which is what makes it
usable as a null -- the claim is structural, not "we hope it does not use order".

Three rungs, selected with `emb_type`, each adding one kind of information:

    item      how hard is this question, and how hard is its concept
    student   + how well has this student done so far, over everything
    count     + how well has this student done on *this concept* so far

`count` is the classical logistic baseline (Pfaffian/PFA-shaped: per-skill
correct and incorrect counts), which Gervet et al. 2020 and Wilson et al. 2016
report as competitive with DKT.  Neither ran under this repository's protocol, so
the point of having it here is not the comparison to their numbers but that it
can share a table with `dkt` and `simplekt`: same folds, same windowed test file,
same `is_repeat` filter, same protocol stamp.

The learnable part is one `Linear(F, 1)` -- six weights and a bias at the widest
rung.  Everything else is frozen and derived from the fold's training rows only,
so the run stamps `feature_fit_scope: train_folds`.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from core.backbone import infer_valid_mask
from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY

#: Which features each rung stacks, in order. Names index `_FEATURES`.
LADDER = {
    "item": ("item_diff", "concept_diff"),
    "student": ("item_diff", "concept_diff", "student_acc", "student_n"),
    "count": (
        "item_diff",
        "concept_diff",
        "student_acc",
        "student_n",
        "skill_acc",
        "skill_n",
    ),
}


def exclusive_cumsum(x: torch.Tensor) -> torch.Tensor:
    """Sum over positions strictly before `t`, along dim 1.

    The whole causality argument rests on this one line: `cumsum` is inclusive,
    so subtracting the value at `t` leaves exactly the prefix.  Position 0 gets
    0, which is the correct "nothing observed yet".
    """
    return torch.cumsum(x, dim=1) - x


@MODEL_REGISTRY.register("nullkt")
class NullKT(nn.Module):
    class Inputs(InputSpec):
        """Frozen difficulty tables, counted from this fold's training rows."""

        requires_question_ids = True

        @classmethod
        def prepare(cls, ctx):
            from datasets.feature_utils import compute_difficulty_logodds_tables

            alpha = float(ctx.model_cfg.get("difficulty_alpha", 10.0))
            tables = compute_difficulty_logodds_tables(
                ctx.dataset_cfg["dpath"],
                ctx.resolve_file(
                    ctx.quelevel_key("train_valid_file"), "train_valid_file"
                ),
                folds=ctx.train_folds(),
                num_q=int(ctx.dataset_cfg["num_q"]),
                num_c=int(ctx.dataset_cfg["num_c"]),
                alpha=alpha,
            )
            return ModelInputs(
                model_kwargs={
                    "item_difficulty": tables["items"],
                    "concept_difficulty": tables["concepts"],
                    "base_rate": tables["base_rate"],
                },
                run_config_extras={
                    "difficulty_alpha": alpha,
                    "difficulty_folds": ctx.train_folds(),
                    "base_rate": tables["base_rate"],
                },
                feature_fit_scope="train_folds",
            )

    def __init__(
        self,
        num_c,
        num_q,
        emb_type="count",
        item_difficulty=None,
        concept_difficulty=None,
        base_rate=0.5,
        prior_strength=5.0,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "nullkt"
        self.num_c = int(num_c)
        self.num_q = int(num_q)
        if emb_type not in LADDER:
            raise ValueError(
                f"nullkt emb_type must be one of {sorted(LADDER)}, got {emb_type!r}."
            )
        self.emb_type = emb_type
        self.feature_names = LADDER[emb_type]
        self.prior_strength = float(prior_strength)

        base_rate = float(base_rate)
        if not 0.0 < base_rate < 1.0:
            raise ValueError(f"base_rate must be in (0, 1), got {base_rate}.")
        self.base_rate = base_rate
        self.base_logit = math.log(base_rate / (1.0 - base_rate))

        # Zeros when `prepare` did not run -- the contract test builds every model
        # straight from its config block. The model stays well-defined (it just
        # predicts a constant from the difficulty features), which is what that
        # test needs; a real run always goes through `prepare`.
        self.register_buffer(
            "item_difficulty", self._as_table(item_difficulty, self.num_q + 1)
        )
        self.register_buffer(
            "concept_difficulty", self._as_table(concept_difficulty, self.num_c + 1)
        )

        self.head = nn.Linear(len(self.feature_names), 1)
        nn.init.zeros_(self.head.weight)
        nn.init.constant_(self.head.bias, self.base_logit)

    @staticmethod
    def _as_table(values, size):
        if values is None:
            return torch.zeros(size, dtype=torch.float32)
        table = torch.as_tensor(values, dtype=torch.float32).reshape(-1)
        if table.numel() != size:
            raise ValueError(
                f"difficulty table has {table.numel()} rows, expected {size}."
            )
        return table

    def _acc_logit(self, correct, count):
        """Prefix accuracy as a log-odds offset, shrunk toward the base rate.

        At `count == 0` this is exactly 0, so a student's first position carries
        no claim about them -- the same convention the frozen item table uses for
        an unseen question.
        """
        alpha = self.prior_strength
        rate = (correct + alpha * self.base_rate) / (count + alpha)
        rate = rate.clamp(1e-6, 1.0 - 1e-6)
        return torch.log(rate / (1.0 - rate)) - self.base_logit

    def _multi_hot(self, concepts):
        """[B,T] or [B,T,K] concept ids -> [B,T,num_c] indicator, -1 dropped."""
        if concepts.dim() == 2:
            hot = torch.zeros(
                *concepts.shape, self.num_c, device=concepts.device, dtype=torch.float32
            )
            valid = (concepts >= 0) & (concepts < self.num_c)
            hot.scatter_(2, concepts.clamp(0, self.num_c - 1).unsqueeze(-1),
                         valid.unsqueeze(-1).float())
            return hot
        valid = (concepts >= 0) & (concepts < self.num_c)
        hot = torch.zeros(
            concepts.size(0), concepts.size(1), self.num_c,
            device=concepts.device, dtype=torch.float32,
        )
        hot.scatter_add_(
            2, concepts.clamp(0, self.num_c - 1), valid.float()
        )
        # A question that lists the same concept twice must still count once.
        return hot.clamp(max=1.0)

    def features(self, questions, concepts, responses):
        """[B,T] feature stack. Every column is a prefix count or prefix mean."""
        valid = infer_valid_mask(concepts).float()
        answered = responses.clamp(min=0).float() * valid

        hot = self._multi_hot(concepts) * valid.unsqueeze(-1)
        width = hot.sum(-1).clamp(min=1.0)

        computed = {}
        if "item_diff" in self.feature_names:
            in_range = (questions >= 0) & (questions < self.num_q)
            item_idx = torch.where(
                in_range, questions, torch.full_like(questions, self.num_q)
            )
            computed["item_diff"] = self.item_difficulty[item_idx]
        if "concept_diff" in self.feature_names:
            table = self.concept_difficulty[: self.num_c].view(1, 1, -1)
            computed["concept_diff"] = (hot * table).sum(-1) / width

        if "student_acc" in self.feature_names:
            prior_n = exclusive_cumsum(valid)
            prior_correct = exclusive_cumsum(answered)
            computed["student_acc"] = self._acc_logit(prior_correct, prior_n)
            computed["student_n"] = torch.log1p(prior_n)

        if "skill_acc" in self.feature_names:
            seen_c = exclusive_cumsum(hot)
            right_c = exclusive_cumsum(hot * answered.unsqueeze(-1))
            skill_n = (hot * seen_c).sum(-1) / width
            skill_correct = (hot * right_c).sum(-1) / width
            computed["skill_acc"] = self._acc_logit(skill_correct, skill_n)
            computed["skill_n"] = torch.log1p(skill_n)

        return torch.stack([computed[name] for name in self.feature_names], dim=-1)

    def forward(self, qseqs, rseqs, cseqs, qshft, cshft, rshft,
                pidseqs=None, pidshft=None, **kwargs):
        # Same reconstruction SimpleKT does: the loader hands out (current,
        # shifted) pairs and the prefix counts need the full sequence.
        concepts = cseqs if cseqs is not None else qseqs
        concept_shft = cshft if cshft is not None else qshft
        if concepts is None or concept_shft is None:
            raise ValueError("nullkt requires concept or question sequences.")
        concepts = torch.cat((concepts[:, 0:1], concept_shft), dim=1).long()

        questions = qseqs if qseqs is not None else pidseqs
        question_shft = qshft if qshft is not None else pidshft
        if questions is None or question_shft is None:
            # Concept-only dataset: the item column falls back to the concept id,
            # matching `SeqBatch.items`.
            questions = concepts if concepts.dim() == 2 else concepts[..., 0]
        else:
            questions = torch.cat(
                (questions[:, 0:1], question_shft), dim=1
            ).long()

        responses = torch.cat((rseqs[:, 0:1], rshft), dim=1).long()

        logits = self.head(self.features(questions, concepts, responses))
        return torch.sigmoid(logits.squeeze(-1))
