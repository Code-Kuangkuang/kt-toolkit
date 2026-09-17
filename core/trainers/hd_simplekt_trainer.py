import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainers.simplekt_trainer import SimpleKTTrainer


@TRAINER_REGISTRY.register("hd_simplekt")
class HDSimpleKTTrainer(SimpleKTTrainer):
    def _forward_batch(self, batch):
        def optional_to_device(key):
            value = batch.get(key)
            if value is None or not value.numel():
                return None
            return value.to(self.device).long()

        qseqs = optional_to_device("qseqs")
        qshft = optional_to_device("shft_qseqs")
        cseqs = optional_to_device("cseqs")
        cshft = optional_to_device("shft_cseqs")
        pidseqs = optional_to_device("pidseqs")
        pidshft = optional_to_device("shft_pidseqs")
        rseqs = batch["rseqs"].to(self.device).long()
        rshft = batch["shft_rseqs"].to(self.device).float()
        transition_mask = batch["masks"].to(self.device).bool()
        score_mask = batch["smasks"].to(self.device).bool()

        base_seqs = cseqs if cseqs is not None else qseqs
        base_shft = cshft if cshft is not None else qshft
        if base_seqs is None or base_shft is None:
            raise ValueError("HD-SimpleKT requires concept or question sequences.")
        valid_mask = torch.cat(
            (transition_mask[:, 0:1], transition_mask), dim=1
        )
        output = self.model(
            qseqs=qseqs,
            rseqs=rseqs,
            cseqs=base_seqs,
            qshft=qshft,
            cshft=base_shft,
            rshft=rshft,
            pidseqs=pidseqs,
            pidshft=pidshft,
            valid_mask=valid_mask,
            return_details=True,
        )
        predictions = output["predictions"]
        aligned_predictions = (
            predictions[:, 1:]
            if predictions.size(1) == rshft.size(1) + 1
            else predictions
        )
        if aligned_predictions.shape != rshft.shape:
            raise ValueError("HD-SimpleKT predictions do not align with targets.")
        pred = torch.masked_select(aligned_predictions, score_mask)
        target = torch.masked_select(rshft, score_mask)
        prediction_loss = (
            binary_cross_entropy(pred, target)
            if pred.numel()
            else aligned_predictions.sum() * 0.0
        )
        loss = prediction_loss + self.model.reconstruction_weight * output[
            "reconstruction_loss"
        ]
        if not torch.isfinite(loss):
            raise FloatingPointError("HD-SimpleKT produced a non-finite loss.")
        return pred, target, loss

