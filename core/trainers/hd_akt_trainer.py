import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainers.akt_trainer import AKTTrainer


@TRAINER_REGISTRY.register("hd_akt")
class HDAKTTrainer(AKTTrainer):
    def _forward_batch(self, batch):
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        rseqs = batch["rseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        score_mask = batch["smasks"].to(self.device).bool()
        transition_mask = batch["masks"].to(self.device).bool()

        q_full = self._concat_full(
            qseqs.to(self.device) if qseqs is not None and qseqs.numel() else None,
            qshft.to(self.device) if qshft is not None and qshft.numel() else None,
        )
        c_full = self._concat_full(
            cseqs.to(self.device) if cseqs is not None and cseqs.numel() else None,
            cshft.to(self.device) if cshft is not None and cshft.numel() else None,
        )
        r_full = self._concat_full(rseqs, rshft)
        concept_data = c_full if c_full is not None else q_full
        pid_data = q_full if c_full is not None else None
        if concept_data is None:
            raise ValueError("HD-AKT requires concept or question sequences.")
        if pid_data is None and self.model.n_pid > 0:
            raise ValueError("HD-AKT requires question ids when n_pid > 0.")
        valid_mask = torch.cat(
            (transition_mask[:, 0:1], transition_mask), dim=1
        )

        output = self.model(
            concept_data.long(),
            r_full.long(),
            pid_data.long() if pid_data is not None else None,
            valid_mask=valid_mask,
            return_details=True,
        )
        aligned_predictions = output["predictions"][:, 1:]
        pred = torch.masked_select(aligned_predictions, score_mask)
        target = torch.masked_select(rshft, score_mask)
        prediction_loss = (
            binary_cross_entropy(pred, target)
            if pred.numel()
            else aligned_predictions.sum() * 0.0
        )
        regularization_loss = output["regularization_loss"]
        loss = (
            prediction_loss
            + regularization_loss
            + self.model.reconstruction_weight * output["reconstruction_loss"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("HD-AKT produced a non-finite loss.")
        return pred, target, regularization_loss, loss

