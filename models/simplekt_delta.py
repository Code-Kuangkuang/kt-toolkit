"""SimpleKT GRU + item shrinkage + personal residual memory (2026-09-23).

Original experiment adapter using SimpleKT's existing embedding/encoder stages.
Memory equations: models/residual_memory.py; no upstream DeltaNet code is copied.

Full-sequence flow: q/r [B,L], c [B,L,K], valid [B,L] -> query [B,L,E]
and causal backbone logits [B,L] -> residual memory [B,D] -> logits [B,L].
The trainer reconstructs L=T+1 and scores logits[:,1:] using smasks [B,T].
"""

import math

import torch
from torch import nn
from torch.nn import functional as F

from core.backbone import SeqBatch
from core.registry import MODEL_REGISTRY
from .residual_memory import ResidualMemory
from .simplekt import SimpleKT


@MODEL_REGISTRY.register("simplekt_delta")
class SimpleKTDelta(nn.Module):
    composed_of = (SimpleKT,)

    class Inputs(SimpleKT.Inputs):
        dataset_mode = "all_in_one"
        requires_question_ids = True

        @classmethod
        def prepare(cls, ctx):
            inputs = super().prepare(ctx)
            inputs.run_config_extras.update({
                "residual_memory_value": "response_minus_raw_backbone_probability",
                "residual_memory_reset": "sequence_chunk",
                "residual_memory_first_position_write": False,
                "residual_memory_training": "joint",
            })
            return inputs

    def __init__(self, num_c, num_q, emb_type="qid_shrunkvec_gru",
                 memory_dim=32, memory_rate=0.1, memory_rule="delta",
                 residual_detach=True, **kwargs):
        super().__init__()
        if emb_type not in ("qid_shrunkvec_gru", "qid_gru"):
            raise ValueError("simplekt_delta supports qid_shrunkvec_gru or qid_gru")
        if num_q < 1:
            raise ValueError("simplekt_delta requires question IDs")
        self.model_name, self.emb_type = "simplekt_delta", emb_type
        self.num_c, self.num_q = int(num_c), int(num_q)
        # Construct the backbone first so equal seeds reproduce SimpleKT init.
        self.backbone = SimpleKT(num_c=num_c, num_q=num_q, emb_type=emb_type, **kwargs)
        self.memory = ResidualMemory(self.backbone.emb_size, num_c,
                                     memory_dim, memory_rate, memory_rule)
        self.residual_detach = bool(residual_detach)
        self.backbone_frozen = False
        self.log_calibration_scale = nn.Parameter(torch.zeros(()))
        self.calibration_bias = nn.Parameter(torch.zeros(()))
        if memory_rule != "none":
            self.raw_memory_gain = nn.Parameter(torch.tensor(math.log(math.expm1(1.0))))
        else:
            self.register_parameter("raw_memory_gain", None)

    def load_backbone(self, state_dict, freeze=True):
        """Load a matching SimpleKT state, including its fitted count buffers.

        The caller owns student isolation and provenance; the diagnostic entry
        point constructs a fresh backbone on disjoint backbone-fit students.
        """
        self.backbone.load_state_dict(state_dict, strict=True)
        self.backbone_frozen = bool(freeze)
        self.backbone.requires_grad_(not self.backbone_frozen)
        self.train(self.training)

    def train(self, mode=True):
        super().train(mode)
        if self.backbone_frozen:
            self.backbone.eval()
        return self

    def backbone_features(self, questions, concepts, responses, valid_mask):
        """Full sequences in; raw logits and item queries out. No label shift."""
        if questions.shape != responses.shape or questions.ndim != 2:
            raise ValueError("questions/responses must align as [B,L]")
        if concepts.ndim not in (2, 3) or concepts.shape[:2] != responses.shape:
            raise ValueError("concepts must align as [B,L] or [B,L,K]")
        if valid_mask.shape != responses.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be explicit bool [B,L]")
        if (valid_mask[:, 1:] & ~valid_mask[:, :-1]).any():
            raise ValueError("GRU expects contiguous valid prefixes followed by padding")
        if (((questions < 0) | (questions >= self.num_q)) & valid_mask).any():
            raise ValueError("Question ID out of range at a valid position")
        if (((responses != 0) & (responses != 1)) & valid_mask).any():
            raise ValueError("Responses must be binary at valid positions")
        active = valid_mask if concepts.ndim == 2 else valid_mask.unsqueeze(-1)
        if (((concepts < -1) | (concepts >= self.num_c)) & active).any():
            raise ValueError("Concept ID out of range at a valid position")
        has_kc = concepts >= 0
        if concepts.ndim == 3:
            has_kc = has_kc.any(-1)
        if (valid_mask & ~has_kc).any():
            raise ValueError("Valid question has no concept")
        batch = SeqBatch(
            concepts=torch.where(active, concepts, -1).long(),
            responses=torch.where(valid_mask, responses, 0).long(),
            questions=torch.where(valid_mask, questions, self.backbone.num_pid).long(),
            valid_mask=valid_mask,
        )
        with torch.set_grad_enabled(torch.is_grad_enabled() and not self.backbone_frozen):
            emb = self.backbone.embed(batch)
            emb.query = emb.query * valid_mask.unsqueeze(-1)
            emb.history = emb.history * valid_mask.unsqueeze(-1)
            hidden = self.backbone.encode(emb)
            logits = self.backbone._logits(hidden, emb)
        return logits, emb.query

    def correct_logits(self, base_logits, features, responses, valid_mask, concepts):
        residual = responses.to(base_logits.dtype) - base_logits.sigmoid()
        if self.residual_detach:
            residual = residual.detach()
        read, final_state = self.memory(features, residual, valid_mask, concepts)
        logits = self.log_calibration_scale.exp() * base_logits + self.calibration_bias
        if self.raw_memory_gain is not None:
            logits = logits + F.softplus(self.raw_memory_gain) * read
        return {"logits": logits, "base_logits": base_logits,
                "memory_read": read, "final_state": final_state}

    def forward(self, questions, concepts, responses, valid_mask, return_details=False):
        base_logits, features = self.backbone_features(questions, concepts, responses, valid_mask)
        details = self.correct_logits(base_logits, features, responses, valid_mask, concepts)
        return details if return_details else details["logits"].sigmoid()
