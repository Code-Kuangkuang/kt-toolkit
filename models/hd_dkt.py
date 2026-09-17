import torch
import torch.nn as nn

from core.registry import MODEL_REGISTRY
from models.dkt import DKT
from models.hdkt_core import HybridInteractionDenoiser
from models.multi_concept import pool_interaction_embeddings


@MODEL_REGISTRY.register("hd_dkt")
class HDDKT(nn.Module):
    """DKT backbone augmented with causal HD-KT interaction denoising."""

    def __init__(
        self,
        num_c,
        num_q=0,
        emb_size=200,
        dropout=0.1,
        emb_type="qid",
        detector_hidden=None,
        latent_dim=None,
        gumbel_tau=1.0,
        hard_detection=True,
        reconstruction_weight=0.01,
        kl_weight=0.001,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "hd_dkt"
        self.num_c = int(num_c)
        self.reconstruction_weight = float(reconstruction_weight)
        self.backbone = DKT(num_c, emb_size, dropout=dropout, emb_type=emb_type)
        self.denoiser = HybridInteractionDenoiser(
            num_items=max(int(num_q), int(num_c)),
            num_c=num_c,
            embedding_dim=emb_size,
            detector_hidden=detector_hidden,
            latent_dim=latent_dim,
            dropout=dropout,
            gumbel_tau=gumbel_tau,
            hard_detection=hard_detection,
            kl_weight=kl_weight,
        )

    def forward(
        self, concepts, responses, item_data=None, valid_mask=None, return_details=False
    ):
        if concepts.dim() not in (2, 3):
            raise ValueError("HD-DKT concepts must have shape [B,T] or [B,T,K].")
        concepts = concepts.long()
        responses = responses.long().clamp(min=0, max=1)
        items = concepts if item_data is None else item_data.long()
        details = self.denoiser(items, concepts, responses, valid_mask=valid_mask)

        interaction_embedding = pool_interaction_embeddings(
            self.backbone.interaction_emb, concepts, responses, self.num_c
        )
        interaction_embedding = interaction_embedding * details["gate"].unsqueeze(-1)
        hidden, _ = self.backbone.lstm_layer(interaction_embedding)
        hidden = self.backbone.dropout_layer(hidden)
        predictions = torch.sigmoid(self.backbone.out_layer(hidden))
        output = {"predictions": predictions, **details}
        return output if return_details else predictions
