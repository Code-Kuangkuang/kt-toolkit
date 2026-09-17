import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainers.dkt_trainer import DKTTrainer
from models.multi_concept import pool_concept_predictions


@TRAINER_REGISTRY.register("hd_dkt")
class HDDKTTrainer(DKTTrainer):
    def _forward_batch(self, batch):
        qseqs = batch.get("qseqs")
        cseqs = batch.get("cseqs")
        qshft = batch.get("shft_qseqs")
        cshft = batch.get("shft_cseqs")
        base_seqs = cseqs if cseqs is not None and cseqs.numel() else qseqs
        target_idx = cshft if cshft is not None and cshft.numel() else qshft
        if base_seqs is None or target_idx is None:
            raise ValueError("HD-DKT requires current and shifted concept/question ids.")

        base_seqs = base_seqs.to(self.device).long()
        target_idx = target_idx.to(self.device).long()
        item_data = (
            qseqs.to(self.device).long()
            if qseqs is not None and qseqs.numel()
            else base_seqs
        )
        responses = batch["rseqs"].to(self.device).long()
        shifted_responses = batch["shft_rseqs"].to(self.device).float()
        valid_mask = batch["masks"].to(self.device).bool()
        score_mask = batch["smasks"].to(self.device).bool()

        output = self.model(
            base_seqs,
            responses,
            item_data=item_data,
            valid_mask=valid_mask,
            return_details=True,
        )
        all_predictions = output["predictions"]
        aligned_predictions, target_has_concept = pool_concept_predictions(
            all_predictions, target_idx, all_predictions.size(-1)
        )
        if torch.any(score_mask & ~target_has_concept):
            raise ValueError("HD-DKT found a scored question without a valid concept id.")
        pred = torch.masked_select(aligned_predictions, score_mask)
        target = torch.masked_select(shifted_responses, score_mask)
        prediction_loss = (
            binary_cross_entropy(pred, target)
            if pred.numel()
            else aligned_predictions.sum() * 0.0
        )
        loss = prediction_loss + self.model.reconstruction_weight * output[
            "reconstruction_loss"
        ]
        if not torch.isfinite(loss):
            raise FloatingPointError("HD-DKT produced a non-finite loss.")
        return pred, target, loss
