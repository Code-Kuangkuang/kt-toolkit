import torch
from torch.nn.functional import binary_cross_entropy

from core.registry import TRAINER_REGISTRY
from core.trainers.lpkt_trainer import LPKTTrainer


@TRAINER_REGISTRY.register("hdkt")
class HDKTTrainer(LPKTTrainer):
    """Batch alignment and objective for causal HD-KT."""

    @staticmethod
    def _full_sequence(current, shifted):
        if current is None or shifted is None:
            return None
        return torch.cat((current[:, 0:1], shifted), dim=1)

    def _validate_indices(self, name, tensor, maximum, allow_negative=False):
        if tensor is None or tensor.numel() == 0:
            return
        valid = tensor >= 0 if allow_negative else torch.ones_like(tensor, dtype=torch.bool)
        selected = tensor[valid]
        if selected.numel() == 0:
            return
        minimum_value = int(selected.min().item())
        maximum_value = int(selected.max().item())
        if minimum_value < 0 or maximum_value > maximum:
            raise ValueError(
                f"HDKT {name} index out of range: min={minimum_value}, "
                f"max={maximum_value}, allowed=[0, {maximum}]."
            )

    def _forward_batch(self, batch):
        qseqs = batch.get("qseqs")
        qshft = batch.get("shft_qseqs")
        cseqs = batch.get("cseqs")
        cshft = batch.get("shft_cseqs")
        if qseqs is None or qshft is None:
            raise ValueError("HDKT requires question sequences.")
        if cseqs is None or cshft is None:
            raise ValueError("HDKT requires concept sequences.")

        rseqs = batch["rseqs"]
        rshft = batch["shft_rseqs"].to(self.device)
        transition_mask = batch["masks"].to(self.device).bool()
        score_mask = batch["smasks"].to(self.device).bool()

        exercise_data = self._full_sequence(qseqs, qshft).long()
        concept_data = self._full_sequence(cseqs, cshft).long()
        responses = self._full_sequence(rseqs, batch["shft_rseqs"]).float()
        valid_mask = torch.cat(
            (transition_mask[:, 0:1], transition_mask), dim=1
        )

        it_data = self._full_sequence(
            batch.get("itseqs"), batch.get("shft_itseqs")
        )
        at_data = self._full_sequence(
            batch.get("utseqs"), batch.get("shft_utseqs")
        )
        if self.model.use_time:
            if it_data is None:
                raise ValueError("HDKT use_time=True requires itseqs in the batch.")
            it_data = self._bucketize_time(
                it_data.long(), self.model.it_embed.num_embeddings - 1
            )
            if at_data is not None:
                at_data = self._bucketize_time(
                    at_data.long(), self.model.at_embed.num_embeddings - 1
                )

        self._validate_indices(
            "question", exercise_data, self.model.e_embed.num_embeddings - 1
        )
        self._validate_indices(
            "concept", concept_data, self.model.num_c - 1, allow_negative=True
        )
        self._validate_indices(
            "interval time",
            it_data,
            self.model.it_embed.num_embeddings - 1,
        )
        self._validate_indices(
            "answer time",
            at_data,
            self.model.at_embed.num_embeddings - 1,
        )

        exercise_data = exercise_data.to(self.device)
        concept_data = concept_data.to(self.device)
        responses = responses.to(self.device)
        if it_data is not None:
            it_data = it_data.to(self.device)
        if at_data is not None:
            at_data = at_data.to(self.device)

        output = self.model(
            exercise_data,
            concept_data,
            responses,
            it_data=it_data,
            at_data=at_data,
            valid_mask=valid_mask,
            return_details=True,
        )
        predictions = output["predictions"][:, 1:]
        pred = torch.masked_select(predictions, score_mask)
        target = torch.masked_select(rshft, score_mask)

        if pred.numel() == 0:
            prediction_loss = predictions.sum() * 0.0
        else:
            if not torch.isfinite(pred).all():
                raise FloatingPointError("HDKT produced non-finite predictions.")
            prediction_loss = binary_cross_entropy(pred, target.float())
        loss = prediction_loss + (
            self.model.reconstruction_weight * output["reconstruction_loss"]
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("HDKT produced a non-finite loss.")
        return pred, target, loss
