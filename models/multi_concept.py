"""Utilities for permutation-invariant multi-concept question encoding."""

import torch


def _validate_concepts(concepts, num_c, name="concepts"):
    if concepts.dim() not in (2, 3):
        raise ValueError(
            f"{name} must have shape [B,T] or [B,T,K], got {tuple(concepts.shape)}."
        )
    invalid = concepts >= int(num_c)
    if invalid.any():
        maximum = int(concepts[invalid].max().item())
        raise ValueError(
            f"{name} contains id {maximum}, but valid concept ids are "
            f"-1 or [0, {int(num_c) - 1}]."
        )


def concept_validity(concepts, num_c):
    """Return valid slots and valid question positions for concept ids."""
    _validate_concepts(concepts, num_c)
    slots = (concepts >= 0) & (concepts < int(num_c))
    questions = slots if concepts.dim() == 2 else slots.any(dim=-1)
    return slots, questions


def pool_concept_embeddings(embedding, concepts, num_c):
    """Embed [B,T] ids or mean-pool valid ids in [B,T,K]."""
    slots, _ = concept_validity(concepts, num_c)
    safe = concepts.long().clamp(min=0, max=int(num_c) - 1)
    values = embedding(safe)
    if concepts.dim() == 2:
        # Preserve the historical one-by-one path exactly. Dataset padding is
        # already masked before it reaches a model.
        return values
    weights = slots.unsqueeze(-1).to(values.dtype)
    denominator = slots.sum(dim=-1, keepdim=True).clamp(min=1).to(values.dtype)
    return (values * weights).sum(dim=-2) / denominator


def pool_interaction_embeddings(embedding, concepts, responses, num_c):
    """Embed concept-response pairs, pooling KCs within the same question."""
    if responses.dim() != 2 or responses.shape != concepts.shape[:2]:
        raise ValueError(
            "responses must have shape [B,T] aligned with the concept sequence."
        )
    slots, _ = concept_validity(concepts, num_c)
    safe = concepts.long().clamp(min=0, max=int(num_c) - 1)
    answers = responses.long().clamp(min=0, max=1)
    if concepts.dim() == 2:
        interaction_ids = safe + int(num_c) * answers
        return embedding(interaction_ids)
    interaction_ids = safe + int(num_c) * answers.unsqueeze(-1)
    values = embedding(interaction_ids)
    weights = slots.unsqueeze(-1).to(values.dtype)
    denominator = slots.sum(dim=-1, keepdim=True).clamp(min=1).to(values.dtype)
    return (values * weights).sum(dim=-2) / denominator


def pool_concept_predictions(predictions, concepts, num_c):
    """Select one KC prediction or mean-pool predictions for a KC set."""
    if predictions.dim() != 3 or predictions.shape[:2] != concepts.shape[:2]:
        raise ValueError(
            "predictions [B,T,C] and concepts [B,T] or [B,T,K] are not aligned."
        )
    slots, questions = concept_validity(concepts, num_c)
    safe = concepts.long().clamp(min=0, max=int(num_c) - 1)
    if concepts.dim() == 2:
        selected = predictions.gather(-1, safe.unsqueeze(-1)).squeeze(-1)
        return selected, questions
    selected = predictions.gather(-1, safe)
    weights = slots.to(selected.dtype)
    denominator = slots.sum(dim=-1).clamp(min=1).to(selected.dtype)
    return (selected * weights).sum(dim=-1) / denominator, questions
