"""Causal scalar-value associative memory, independently usable by KT models.

Delta write: m <- m + rate * (value - <m, key>) * key.
This is the classical delta rule used by DeltaNet (arXiv:2406.06484),
implemented here from the equation, not copied from an upstream model.
The value in this study is r - p_backbone, not mastery or the final BCE gradient.
"""

import math

import torch
from torch import nn
from torch.nn import functional as F


def delta_write(memory, key, value, rate):
    """One write with unit keys: [B,D], [B,D], [B] -> [B,D]."""
    innovation = value - (memory * key).sum(-1)
    return memory + rate * innovation.unsqueeze(-1) * key


class ResidualMemory(nn.Module):
    """Read before writing; keep no persistent state between forward calls.

    Dense delta/ema arms have exactly the same parameters and input features.
    kc_ema instead maintains one scalar per KC, averaging distinct KC reads
    and independently updating each involved KC. Its state capacity differs.
    """

    RULES = ("none", "kc_ema", "ema", "delta")

    def __init__(self, input_dim, num_c, memory_dim=32,
                 memory_rate=0.1, memory_rule="delta"):
        super().__init__()
        if memory_rule not in self.RULES:
            raise ValueError(f"memory_rule must be one of {self.RULES}")
        if not math.isfinite(memory_rate) or not 0 <= memory_rate <= 1:
            raise ValueError("memory_rate must be finite and in [0, 1]")
        if memory_dim < 1 or num_c < 1:
            raise ValueError("memory_dim and num_c must be positive")
        self.rule, self.rate = memory_rule, float(memory_rate)
        self.memory_dim, self.num_c = int(memory_dim), int(num_c)
        self.key_projection = (
            nn.Linear(input_dim, memory_dim, bias=False)
            if memory_rule in ("delta", "ema") else None
        )

    def keys(self, features):
        projected = self.key_projection(features)
        # Give a zero projection a deterministic unit address as well. This
        # preserves the unit-key interpretation without dividing by zero.
        norm = projected.norm(dim=-1, keepdim=True)
        fallback = torch.zeros_like(projected)
        fallback[..., 0] = 1
        return torch.where(norm > 1e-8, F.normalize(projected, dim=-1), fallback)

    def forward(self, features, residual, valid_mask, concepts=None):
        """[B,L,E], [B,L], [B,L], optional [B,L,K] -> reads, final state.

        Position zero is context-only and does not write a residual, matching
        the preceding residual-state study. Other valid context positions DO
        write, even when not scored. The score mask must not be used here.
        """
        if features.ndim != 3 or residual.shape != features.shape[:2]:
            raise ValueError("features [B,L,E] and residual [B,L] must align")
        if valid_mask.shape != residual.shape or valid_mask.dtype != torch.bool:
            raise ValueError("valid_mask must be bool [B,L] aligned with residual")
        batch, length = residual.shape
        if length < 1:
            raise ValueError("memory requires a nonempty sequence")
        if self.rule == "none":
            return residual.new_zeros(batch, length), residual.new_zeros(batch, 0)

        features = torch.where(valid_mask.unsqueeze(-1), features, 0)

        if self.rule == "kc_ema":
            if concepts is None or concepts.shape[:2] != residual.shape:
                raise ValueError("kc_ema requires aligned concept IDs")
            if concepts.ndim == 2:
                concepts = concepts.unsqueeze(-1)
            if concepts.ndim != 3:
                raise ValueError("concepts must be [B,L] or [B,L,K]")
            concepts = concepts.sort(dim=-1).values
            slots = (concepts >= 0) & valid_mask.unsqueeze(-1)
            if ((concepts >= self.num_c) & slots).any():
                raise ValueError("KC ID exceeds num_c")
            # A duplicated tag is one KC, not two independent observations.
            slots[..., 1:] &= concepts[..., 1:] != concepts[..., :-1]
            weights = slots.to(features.dtype)
            ids = concepts.clamp(min=0, max=self.num_c - 1).long()
            denominator = weights.sum(-1).clamp(min=1)
            state = features.new_zeros(batch, self.num_c)
        else:
            keys = self.keys(features)                       # [B,L,D]
            state = features.new_zeros(batch, self.memory_dim)

        reads = []
        for t in range(length):
            active = valid_mask[:, t]
            if self.rule == "kc_ema":
                old = state.gather(1, ids[:, t])             # [B,K]
                read = (old * weights[:, t]).sum(-1) / denominator[:, t]
            else:
                key = keys[:, t]                            # [B,D]
                read = (state * key).sum(-1)
            reads.append(torch.where(active, read, torch.zeros_like(read)))
            if t == 0:
                continue
            value = torch.where(active, residual[:, t], torch.zeros_like(read))
            if self.rule == "kc_ema":
                change = self.rate * (value.unsqueeze(-1) - old) * weights[:, t]
                candidate = state.scatter_add(1, ids[:, t], change)
            elif self.rule == "delta":
                candidate = delta_write(state, key, value, self.rate)
            else:
                candidate = (1 - self.rate) * state + self.rate * value.unsqueeze(-1) * key
            state = torch.where(active.unsqueeze(-1), candidate, state)
        return torch.stack(reads, dim=1), state
