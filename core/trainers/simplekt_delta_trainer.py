"""Explicit mask/shift adapter for the residual-memory experiment."""

import torch
from torch.nn import functional as F

from core.registry import TRAINER_REGISTRY
from .simplekt_trainer import SimpleKTTrainer


@TRAINER_REGISTRY.register("simplekt_delta")
class SimpleKTDeltaTrainer(SimpleKTTrainer):
    @staticmethod
    def full_batch(batch, device):
        """Loader [B,T] -> full [B,T+1]; multi-KC keeps its final K axis."""
        masks = batch["masks"].to(device).bool()
        score = batch["smasks"].to(device).bool()
        if masks.ndim != 2 or score.shape != masks.shape or masks.size(1) < 1:
            raise ValueError("masks/smasks must align as nonempty [B,T]")
        if (score & ~masks).any():
            raise ValueError("smasks includes an invalid target")
        full = {}
        for name, field in (("questions", "qseqs"), ("concepts", "cseqs"),
                            ("responses", "rseqs")):
            current, shifted = batch.get(field), batch.get("shft_" + field)
            if current is None or shifted is None or not current.numel():
                raise ValueError(f"simplekt_delta requires {field} and shft_{field}")
            if current.shape != shifted.shape or current.shape[:2] != masks.shape:
                raise ValueError(f"{field} does not align with masks")
            full[name] = torch.cat((current[:, :1], shifted), dim=1).to(device)
        full["valid_mask"] = torch.cat((masks[:, :1], masks), dim=1)
        return full, score

    def _forward_batch(self, batch):
        full, score = self.full_batch(batch, self.device)
        target = full["responses"][:, 1:].float()[score]
        if target.numel() == 0:
            return target, target, self.model.calibration_bias * 0
        details = self.model(**full, return_details=True)
        logits = details["logits"][:, 1:][score]
        if logits.shape != target.shape or not torch.isfinite(logits).all():
            raise ValueError("Non-finite or misaligned residual-memory logits")
        loss = F.binary_cross_entropy_with_logits(logits, target)
        return logits.sigmoid(), target, loss
