"""Leakage-safe KeenKT reimplementation for KT-Toolkit.

The model follows the public KeenKT implementation and paper:

    Li et al., "KeenKT: Knowledge Mastery-State Disambiguation for
    Knowledge Tracing", AAAI 2026.

Reference implementation:
https://github.com/HubuKG/KeenKT
commit 911e4e460b0e2ddb25142fb3c7e46123c1043690 (Apache-2.0).

This port keeps the NIG embeddings, distribution-distance attention,
diffusion reconstruction, distributional contrastive learning, squeeze-
excitation gates, and Rasch-style item variation. It deliberately uses an
explicit strict-causal mask and a padding-aware masked softmax. Therefore the
prediction at position ``t`` can use responses only from positions ``< t``;
the response at ``t`` is never visible to its own prediction.
"""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_inputs import InputSpec
from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings


def _pairwise_nig_distance(
    mean_a: torch.Tensor,
    uncertainty_a: torch.Tensor,
    mean_b: torch.Tensor,
    uncertainty_b: torch.Tensor,
) -> torch.Tensor:
    """Return the non-negative pairwise distance used by KeenKT.

    KeenKT converts the four NIG parameters to a mean and a positive
    uncertainty stream. Following Eq. (6) of the paper, the distance is the
    squared mean distance plus the squared distance between square-root
    uncertainty vectors.
    """

    mean_a_sq = mean_a.square().sum(dim=-1, keepdim=True)
    mean_b_sq = mean_b.square().sum(dim=-1, keepdim=True)
    mean_dist = (
        mean_a_sq
        + mean_b_sq.transpose(-2, -1)
        - 2.0 * mean_a @ mean_b.transpose(-2, -1)
    )

    sqrt_a = torch.sqrt(uncertainty_a.clamp_min(1e-8))
    sqrt_b = torch.sqrt(uncertainty_b.clamp_min(1e-8))
    unc_a_sq = sqrt_a.square().sum(dim=-1, keepdim=True)
    unc_b_sq = sqrt_b.square().sum(dim=-1, keepdim=True)
    uncertainty_dist = (
        unc_a_sq
        + unc_b_sq.transpose(-2, -1)
        - 2.0 * sqrt_a @ sqrt_b.transpose(-2, -1)
    )
    return (mean_dist + uncertainty_dist).clamp_min(0.0)


def _masked_softmax(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Softmax with an exact zero result for rows without valid keys."""

    mask = mask.to(dtype=torch.bool, device=scores.device)
    masked_scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(masked_scores, dim=-1)
    has_key = mask.any(dim=-1, keepdim=True)
    weights = torch.where(has_key, weights, torch.zeros_like(weights))
    return weights.masked_fill(~mask, 0.0)


class SinusoidalPositionEmbedding(nn.Module):
    def __init__(self, d_model: int, max_len: int) -> None:
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        encoding = torch.zeros(max_len, d_model, dtype=torch.float32)
        encoding[:, 0::2] = torch.sin(position * div_term)
        encoding[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=True)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.size(1) > self.encoding.size(1):
            raise ValueError(
                f"KeenKT sequence length {sequence.size(1)} exceeds configured "
                f"maximum {self.encoding.size(1)}."
            )
        return self.encoding[:, : sequence.size(1)].to(
            device=sequence.device,
            dtype=sequence.dtype,
        )


class SqueezeExcitation(nn.Module):
    def __init__(self, d_model: int, reduction: int = 16) -> None:
        super().__init__()
        hidden = max(1, d_model // max(1, int(reduction)))
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden, bias=False),
            nn.ReLU(),
            nn.Linear(hidden, d_model, bias=False),
            nn.Sigmoid(),
        )

    def forward(
        self,
        sequence: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply a prefix-only channel gate.

        The public implementation averages over the complete sequence. That
        would allow a prediction at an early position to depend on future
        hidden states. A cumulative prefix average preserves the intended
        channel recalibration without violating KT chronology.
        """

        if valid_mask is None:
            cumulative = sequence.cumsum(dim=1)
            counts = torch.arange(
                1,
                sequence.size(1) + 1,
                device=sequence.device,
                dtype=sequence.dtype,
            ).view(1, -1, 1)
            summary = cumulative / counts
        else:
            weights = valid_mask.to(sequence.dtype).unsqueeze(-1)
            cumulative = (sequence * weights).cumsum(dim=1)
            counts = weights.cumsum(dim=1).clamp_min(1.0)
            summary = cumulative / counts
        return sequence * self.net(summary)


class DiffusionDenoiser(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
        )

    def forward(self, noisy_state: torch.Tensor) -> torch.Tensor:
        return noisy_state + self.net(noisy_state)


class NIGContrastiveLoss(nn.Module):
    """Symmetric InfoNCE in the NIG-derived distribution space."""

    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("nig_temperature must be positive.")
        self.temperature = float(temperature)

    def forward(
        self,
        mean_a: torch.Tensor,
        uncertainty_a: torch.Tensor,
        mean_b: torch.Tensor,
        uncertainty_b: torch.Tensor,
    ) -> torch.Tensor:
        if mean_a.size(0) <= 1:
            return mean_a.sum() * 0.0
        distance = _pairwise_nig_distance(
            mean_a,
            uncertainty_a,
            mean_b,
            uncertainty_b,
        )
        logits = (1.0 / (1.0 + distance)) / self.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return 0.5 * (
            F.cross_entropy(logits, labels)
            + F.cross_entropy(logits.transpose(0, 1), labels)
        )


class NIGMultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model={d_model} must be divisible by num_attn_heads={num_heads}."
            )
        self.num_heads = int(num_heads)
        self.head_dim = d_model // num_heads
        self.d_model = int(d_model)

        self.q_mean = nn.Linear(d_model, d_model)
        self.k_mean = nn.Linear(d_model, d_model)
        self.v_mean = nn.Linear(d_model, d_model)
        self.q_uncertainty = nn.Linear(d_model, d_model)
        self.k_uncertainty = nn.Linear(d_model, d_model)
        self.v_uncertainty = nn.Linear(d_model, d_model)
        self.out_mean = nn.Linear(d_model, d_model)
        self.out_uncertainty = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for module in (
            self.q_mean,
            self.k_mean,
            self.v_mean,
            self.q_uncertainty,
            self.k_uncertainty,
            self.v_uncertainty,
            self.out_mean,
            self.out_uncertainty,
        ):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, length, _ = value.shape
        return value.view(
            batch,
            length,
            self.num_heads,
            self.head_dim,
        ).transpose(1, 2)

    def _merge_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, _, length, _ = value.shape
        return value.transpose(1, 2).contiguous().view(
            batch,
            length,
            self.d_model,
        )

    def forward(
        self,
        query_mean: torch.Tensor,
        query_uncertainty: torch.Tensor,
        key_mean: torch.Tensor,
        key_uncertainty: torch.Tensor,
        value_mean: torch.Tensor,
        value_uncertainty: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_mean = self._split_heads(self.q_mean(query_mean))
        k_mean = self._split_heads(self.k_mean(key_mean))
        v_mean = self._split_heads(self.v_mean(value_mean))
        q_unc = F.elu(self._split_heads(self.q_uncertainty(query_uncertainty))) + 1.0
        k_unc = F.elu(self._split_heads(self.k_uncertainty(key_uncertainty))) + 1.0
        v_unc = F.elu(self._split_heads(self.v_uncertainty(value_uncertainty))) + 1.0

        distance = _pairwise_nig_distance(q_mean, q_unc, k_mean, k_unc)
        scores = -distance / math.sqrt(self.head_dim)
        weights = _masked_softmax(scores, attention_mask)
        weights = self.dropout(weights)

        mean_output = weights @ v_mean
        uncertainty_output = weights @ v_unc
        return (
            self.out_mean(self._merge_heads(mean_output)),
            self.out_uncertainty(self._merge_heads(uncertainty_output)),
        )


class KeenKTBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.attention = NIGMultiHeadAttention(d_model, num_heads, dropout)
        self.mean_norm1 = nn.LayerNorm(d_model)
        self.uncertainty_norm1 = nn.LayerNorm(d_model)
        self.mean_norm2 = nn.LayerNorm(d_model)
        self.uncertainty_norm2 = nn.LayerNorm(d_model)
        self.mean_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.uncertainty_ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query_mean: torch.Tensor,
        query_uncertainty: torch.Tensor,
        value_mean: torch.Tensor,
        value_uncertainty: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        attended_mean, attended_uncertainty = self.attention(
            query_mean,
            query_uncertainty,
            query_mean,
            query_uncertainty,
            value_mean,
            value_uncertainty,
            attention_mask,
        )
        mean = self.mean_norm1(query_mean + self.dropout(attended_mean))
        uncertainty = self.uncertainty_norm1(
            F.elu(query_uncertainty + self.dropout(attended_uncertainty)) + 1.0
        )

        mean = self.mean_norm2(mean + self.dropout(self.mean_ffn(mean)))
        uncertainty = self.uncertainty_norm2(
            F.elu(
                uncertainty
                + self.dropout(self.uncertainty_ffn(uncertainty))
            )
            + 1.0
        )
        return mean, uncertainty


@MODEL_REGISTRY.register("keenkt")
class KeenKT(nn.Module):
    class Inputs(InputSpec):
        """Declares what this model needs; it derives nothing from the data."""

        dataset_mode = "all_in_one"
        requires_question_ids = True

    """Normal-Inverse-Gaussian knowledge tracing with causal attention."""

    def __init__(
        self,
        num_c: int,
        num_q: int,
        emb_type: str = "stoc_qid",
        d_model: int = 256,
        n_blocks: int = 4,
        d_ff: int = 512,
        num_attn_heads: int = 8,
        dropout: float = 0.2,
        final_fc_dim: int = 256,
        final_fc_dim2: int = 256,
        seq_len: int = 200,
        separate_qa: bool = False,
        use_CL: bool = True,
        cl_weight: float = 0.02,
        use_uncertainty_aug: bool = True,
        use_diffusion: bool = True,
        diffusion_weight: float = 0.08,
        noise_level: float = 0.3,
        nig_temperature: float = 0.07,
        se_ratio: int = 16,
        **_: object,
    ) -> None:
        super().__init__()
        if num_c <= 0 or num_q <= 0:
            raise ValueError("KeenKT requires positive num_c and num_q.")
        if emb_type not in {"stoc_qid", "qid"}:
            raise ValueError(
                "This KeenKT port supports emb_type 'stoc_qid' or 'qid'."
            )

        self.model_name = "keenkt"
        self.num_c = int(num_c)
        self.num_q = int(num_q)
        self.emb_type = emb_type
        self.d_model = int(d_model)
        self.separate_qa = bool(separate_qa)
        self.use_CL = bool(use_CL)
        self.use_uncertainty_aug = bool(use_uncertainty_aug)
        self.use_diffusion = bool(use_diffusion)
        self.cl_weight = float(cl_weight)
        self.diffusion_weight = float(diffusion_weight)
        self.noise_level = float(noise_level)

        self.concept_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(self.num_c, self.d_model)
                for name in ("mu", "alpha", "beta", "delta")
            }
        )
        response_size = 2 * self.num_c + 1 if self.separate_qa else 2
        self.response_embeddings = nn.ModuleDict(
            {
                name: nn.Embedding(response_size, self.d_model)
                for name in ("mu", "alpha", "beta", "delta")
            }
        )

        self.concept_variation = nn.Embedding(self.num_c + 1, self.d_model)
        self.item_difficulty = nn.Embedding(self.num_q + 1, self.d_model)
        nn.init.zeros_(self.item_difficulty.weight)

        self.position = SinusoidalPositionEmbedding(self.d_model, int(seq_len))
        self.blocks = nn.ModuleList(
            [
                KeenKTBlock(
                    self.d_model,
                    int(d_ff),
                    int(num_attn_heads),
                    float(dropout),
                )
                for _ in range(int(n_blocks))
            ]
        )
        self.mean_gate = SqueezeExcitation(self.d_model, se_ratio)
        self.uncertainty_gate = SqueezeExcitation(self.d_model, se_ratio)
        self.predictor = nn.Sequential(
            nn.Linear(self.d_model * 4, int(final_fc_dim)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(final_fc_dim), int(final_fc_dim2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(int(final_fc_dim2), 1),
        )

        self.diffusion = DiffusionDenoiser(self.d_model)
        self.contrastive = NIGContrastiveLoss(nig_temperature)

    @staticmethod
    def _check_ids(name: str, values: torch.Tensor, upper: int) -> None:
        if values.numel() == 0:
            raise ValueError(f"KeenKT received an empty {name} tensor.")
        minimum = int(values.min().item())
        maximum = int(values.max().item())
        if minimum < 0 or maximum >= upper:
            raise ValueError(
                f"KeenKT {name} ids must be in [0, {upper - 1}], "
                f"but observed [{minimum}, {maximum}]."
            )

    @staticmethod
    def _nig_moments(
        embeddings: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = F.softplus(embeddings["alpha"]) + 1e-8
        beta = torch.tanh(embeddings["beta"]) * alpha * 0.999
        delta = F.elu(embeddings["delta"]) + 1.0
        gamma = torch.sqrt((alpha.square() - beta.square()).clamp_min(1e-8))
        mean = embeddings["mu"] + delta * beta / gamma.clamp_min(1e-8)
        uncertainty = (
            torch.sqrt(delta)
            * alpha
            / gamma.clamp_min(1e-8).pow(1.5)
        )
        return mean, uncertainty.clamp_min(1e-8)

    def _embed(
        self,
        concepts: torch.Tensor,
        questions: torch.Tensor,
        responses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Identity on [B,T]; on [B,T,K] mean-pools the question's KCs with
        # -1 padding masked, as pykt's QueEmb.get_avg_skill_emb does.
        concept_raw = {
            name: pool_concept_embeddings(table, concepts, self.num_c)
            for name, table in self.concept_embeddings.items()
        }
        if self.separate_qa:
            qa_raw = {
                name: pool_interaction_embeddings(
                    table, concepts, responses, self.num_c
                )
                for name, table in self.response_embeddings.items()
            }
        else:
            qa_raw = {
                name: concept_raw[name] + table(responses)
                for name, table in self.response_embeddings.items()
            }

        concept_mean, concept_uncertainty = self._nig_moments(concept_raw)
        qa_mean, qa_uncertainty = self._nig_moments(qa_raw)

        item_effect = self.item_difficulty(questions)
        concept_effect = pool_concept_embeddings(
            self.concept_variation, concepts, self.num_c
        )
        concept_mean = concept_mean + item_effect * concept_effect
        concept_uncertainty = concept_uncertainty + item_effect * concept_effect
        return concept_mean, concept_uncertainty, qa_mean, qa_uncertainty

    @staticmethod
    def _attention_mask(valid_mask: torch.Tensor) -> torch.Tensor:
        length = valid_mask.size(1)
        positions = torch.arange(length, device=valid_mask.device)
        strict_past = positions.view(1, 1, length, 1) > positions.view(
            1,
            1,
            1,
            length,
        )
        key_is_valid = valid_mask[:, None, None, :]
        return strict_past & key_is_valid

    def _encode(
        self,
        concept_mean: torch.Tensor,
        concept_uncertainty: torch.Tensor,
        qa_mean: torch.Tensor,
        qa_uncertainty: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        position = self.position(concept_mean)
        mean_state = concept_mean + position
        uncertainty_state = F.elu(concept_uncertainty + position) + 1.0
        qa_mean = qa_mean + position
        qa_uncertainty = F.elu(qa_uncertainty + position) + 1.0
        mask = self._attention_mask(valid_mask)

        for block in self.blocks:
            mean_state, uncertainty_state = block(
                mean_state,
                uncertainty_state,
                qa_mean,
                qa_uncertainty,
                mask,
            )
        mean_state = self.mean_gate(mean_state, valid_mask)
        uncertainty_state = self.uncertainty_gate(
            uncertainty_state,
            valid_mask,
        )
        return mean_state, uncertainty_state

    @staticmethod
    def _masked_pool(
        sequence: torch.Tensor,
        valid_mask: torch.Tensor,
    ) -> torch.Tensor:
        weights = valid_mask.to(sequence.dtype).unsqueeze(-1)
        return (sequence * weights).sum(dim=1) / weights.sum(
            dim=1
        ).clamp_min(1.0)

    def forward(
        self,
        concepts: torch.Tensor,
        questions: torch.Tensor,
        responses: torch.Tensor,
        valid_mask: Optional[torch.Tensor] = None,
        augmented_responses: Optional[torch.Tensor] = None,
        compute_auxiliary: bool = False,
    ) -> Dict[str, torch.Tensor]:
        concepts = concepts.long()
        questions = questions.long()
        responses = responses.long()
        if concepts.shape != questions.shape or concepts.shape != responses.shape:
            raise ValueError(
                "KeenKT expects concepts, questions, and responses with the "
                f"same [B, T] shape, got {tuple(concepts.shape)}, "
                f"{tuple(questions.shape)}, and {tuple(responses.shape)}."
            )
        self._check_ids("concept", concepts, self.num_c)
        self._check_ids("question", questions, self.num_q)
        self._check_ids("response", responses, 2)
        if valid_mask is None:
            valid_mask = torch.ones_like(responses, dtype=torch.bool)
        else:
            valid_mask = valid_mask.to(device=responses.device, dtype=torch.bool)
        if valid_mask.shape != responses.shape:
            raise ValueError("KeenKT valid_mask must match response shape.")

        q_mean, q_uncertainty, qa_mean, qa_uncertainty = self._embed(
            concepts,
            questions,
            responses,
        )
        mean_state, uncertainty_state = self._encode(
            q_mean,
            q_uncertainty,
            qa_mean,
            qa_uncertainty,
            valid_mask,
        )

        features = torch.cat(
            [mean_state, uncertainty_state, q_mean, q_uncertainty],
            dim=-1,
        )
        probabilities = torch.sigmoid(self.predictor(features).squeeze(-1))
        zero = probabilities.sum() * 0.0
        contrastive_loss = zero
        diffusion_loss = zero

        if compute_auxiliary and self.use_diffusion:
            noisy_state = mean_state + torch.randn_like(mean_state) * self.noise_level
            denoised_state = self.diffusion(noisy_state)
            valid_values = valid_mask.unsqueeze(-1).expand_as(mean_state)
            if valid_values.any():
                diffusion_loss = F.mse_loss(
                    denoised_state[valid_values],
                    mean_state.detach()[valid_values],
                )

        if (
            compute_auxiliary
            and self.use_CL
            and augmented_responses is not None
        ):
            augmented_responses = augmented_responses.to(
                device=responses.device,
                dtype=torch.long,
            )
            self._check_ids("augmented response", augmented_responses, 2)
            (
                aug_q_mean,
                aug_q_uncertainty,
                aug_qa_mean,
                aug_qa_uncertainty,
            ) = self._embed(concepts, questions, augmented_responses)
            aug_mean_state, aug_uncertainty_state = self._encode(
                aug_q_mean,
                aug_q_uncertainty,
                aug_qa_mean,
                aug_qa_uncertainty,
                valid_mask,
            )
            contrastive_loss = self.contrastive(
                self._masked_pool(mean_state, valid_mask),
                self._masked_pool(uncertainty_state, valid_mask),
                self._masked_pool(aug_mean_state, valid_mask),
                self._masked_pool(aug_uncertainty_state, valid_mask),
            )

        return {
            "preds": probabilities,
            "contrastive_loss": contrastive_loss,
            "diffusion_loss": diffusion_loss,
            "mean_state": mean_state,
            "uncertainty_state": uncertainty_state,
        }
