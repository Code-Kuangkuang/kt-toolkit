"""Trainer adapter for the leakage-safe KeenKT port."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from core.registry import TRAINER_REGISTRY
from core.trainer import BaseTrainer


def _full_sequence(
    batch,
    sequence_key: str,
    shifted_key: str,
    device: str,
    dtype=torch.long,
):
    sequence = batch.get(sequence_key)
    shifted = batch.get(shifted_key)
    if (
        sequence is None
        or shifted is None
        or sequence.numel() == 0
        or shifted.numel() == 0
    ):
        return None
    sequence = sequence.to(device=device, dtype=dtype)
    shifted = shifted.to(device=device, dtype=dtype)
    if sequence.dim() == 3:
        sequence = sequence[:, :, 0]
    if shifted.dim() == 3:
        shifted = shifted[:, :, 0]
    return torch.cat((sequence[:, :1], shifted), dim=1)


def _full_valid_mask(batch, device: str) -> torch.Tensor:
    masks = batch.get("masks")
    if masks is None or masks.numel() == 0:
        raise ValueError("KeenKT requires the causal sequence mask.")
    masks = masks.to(device=device, dtype=torch.bool)
    first = torch.ones(
        masks.size(0),
        1,
        dtype=torch.bool,
        device=masks.device,
    )
    return torch.cat((first, masks), dim=1)


def _build_training_augmentation(
    responses: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """Construct the response-perturbed positive view used by KeenKT.

    This is applied only to the training fold and only for the auxiliary
    contrastive objective. For each sequence, interactions before the final
    valid interaction that share its response value are flipped; the final
    response itself is preserved. No validation/test response is used and the
    augmented view is never used by the prediction branch during evaluation.
    """

    augmented = responses.clone()
    for row in range(responses.size(0)):
        valid_positions = torch.nonzero(
            valid_mask[row],
            as_tuple=False,
        ).flatten()
        if valid_positions.numel() <= 1:
            continue
        final_position = int(valid_positions[-1].item())
        anchor = responses[row, final_position]
        earlier = valid_positions[:-1]
        flip_positions = earlier[responses[row, earlier] == anchor]
        augmented[row, flip_positions] = 1 - augmented[row, flip_positions]
    return augmented


@TRAINER_REGISTRY.register("keenkt")
class KeenKTTrainer(BaseTrainer):
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
        super().__init__(
            num_epochs=num_epochs,
            hooks=hooks,
            test_loader=test_loader,
        )
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
            pred, _, loss = self._forward_batch(batch, train=True)
            if pred.numel() == 0:
                continue
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            losses.append(float(loss.item()))
            self._print_progress(
                batch_idx,
                total_batches,
                float(loss.item()),
            )
        return float(np.mean(losses)) if losses else 0.0

    def _forward_batch(self, batch, train=False):
        concepts = _full_sequence(
            batch,
            "cseqs",
            "shft_cseqs",
            self.device,
        )
        questions = _full_sequence(
            batch,
            "qseqs",
            "shft_qseqs",
            self.device,
        )
        responses = _full_sequence(
            batch,
            "rseqs",
            "shft_rseqs",
            self.device,
        )
        if concepts is None or questions is None or responses is None:
            raise ValueError(
                "KeenKT requires question, concept, and response sequences."
            )
        valid_mask = _full_valid_mask(batch, self.device)

        rshft = batch["shft_rseqs"].to(self.device).float()
        select_mask = batch["smasks"].to(self.device).bool()
        augmented = None
        if (
            train
            and self.model.use_CL
            and self.model.use_uncertainty_aug
        ):
            augmented = _build_training_augmentation(
                responses,
                valid_mask,
            )

        output = self.model(
            concepts=concepts,
            questions=questions,
            responses=responses,
            valid_mask=valid_mask,
            augmented_responses=augmented,
            compute_auxiliary=train,
        )
        probabilities = output["preds"]
        shifted_probabilities = probabilities[:, 1:]
        if shifted_probabilities.shape != rshft.shape:
            raise ValueError(
                f"KeenKT prediction shape {tuple(probabilities.shape)} does "
                f"not align with shifted target shape {tuple(rshft.shape)}."
            )

        pred = torch.masked_select(shifted_probabilities, select_mask)
        target = torch.masked_select(rshft, select_mask)
        if pred.numel() == 0:
            zero = probabilities.sum() * 0.0
            return pred, target, zero
        if not torch.isfinite(pred).all():
            raise FloatingPointError("KeenKT produced NaN/Inf predictions.")

        bce_loss = F.binary_cross_entropy(
            pred.double(),
            target.double(),
        )
        loss = (
            bce_loss
            + self.model.cl_weight * output["contrastive_loss"]
            + self.model.diffusion_weight * output["diffusion_loss"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("KeenKT produced a NaN/Inf loss.")
        return pred, target, loss
