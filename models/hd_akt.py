import torch
import torch.nn as nn

from core.model_inputs import InputSpec
from core.registry import MODEL_REGISTRY
from models.akt import AKT
from models.hdkt_core import HybridInteractionDenoiser
from models.multi_concept import pool_concept_embeddings


@MODEL_REGISTRY.register("hd_akt")
class HDAKT(nn.Module):
    class Inputs(InputSpec):
        """Declares what this model needs; it derives nothing from the data."""

        dataset_mode = "all_in_one"

    """AKT backbone augmented with causal HD-KT interaction denoising."""

    def __init__(
        self,
        num_c,
        num_q=0,
        d_model=256,
        n_blocks=1,
        dropout=0.1,
        d_ff=256,
        kq_same=1,
        final_fc_dim=512,
        num_attn_heads=8,
        separate_qa=False,
        l2=1e-5,
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
        self.model_name = "hd_akt"
        self.reconstruction_weight = float(reconstruction_weight)
        self.backbone = AKT(
            num_c=num_c,
            num_q=num_q,
            d_model=d_model,
            n_blocks=n_blocks,
            dropout=dropout,
            d_ff=d_ff,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            num_attn_heads=num_attn_heads,
            separate_qa=separate_qa,
            l2=l2,
            emb_type=emb_type,
        )
        self.n_pid = self.backbone.n_pid
        self.denoiser = HybridInteractionDenoiser(
            num_items=max(int(num_q), int(num_c)),
            num_c=num_c,
            embedding_dim=d_model,
            detector_hidden=detector_hidden,
            latent_dim=latent_dim,
            dropout=dropout,
            gumbel_tau=gumbel_tau,
            hard_detection=hard_detection,
            kl_weight=kl_weight,
        )

    def forward(
        self,
        concept_data,
        responses,
        pid_data=None,
        valid_mask=None,
        return_details=False,
    ):
        if concept_data.dim() not in (2, 3):
            raise ValueError("HD-AKT concepts must have shape [B,T] or [B,T,K].")
        concept_data = concept_data.long()
        responses = responses.long().clamp(min=0, max=1)
        items = concept_data if pid_data is None else pid_data.long()
        details = self.denoiser(
            items, concept_data, responses, valid_mask=valid_mask
        )

        q_embed_data, qa_embed_data = self.backbone.base_emb(
            concept_data, responses
        )
        pid_embed_data = None
        if self.backbone.n_pid > 0:
            if pid_data is None:
                raise ValueError("HD-AKT requires question ids when n_pid > 0.")
            q_embed_diff = pool_concept_embeddings(
                self.backbone.q_embed_diff,
                concept_data,
                self.backbone.n_question,
            )
            pid_embed_data = self.backbone.difficult_param(pid_data.long())
            q_embed_data = q_embed_data + pid_embed_data * q_embed_diff
            qa_embed_diff = self.backbone.qa_embed_diff(responses)
            if self.backbone.separate_qa:
                qa_embed_data = qa_embed_data + pid_embed_data * qa_embed_diff
            else:
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff + q_embed_diff
                )
            regularization_loss = (pid_embed_data**2).sum() * self.backbone.l2
        else:
            regularization_loss = q_embed_data.sum() * 0.0

        # Only the response-aware history values are denoised.  The question
        # queries remain available at their prediction positions.
        qa_embed_data = qa_embed_data * details["gate"].unsqueeze(-1)
        hidden = self.backbone.model(
            q_embed_data, qa_embed_data, pid_embed_data
        )
        logits = self.backbone.out(
            torch.cat((hidden, q_embed_data), dim=-1)
        ).squeeze(-1)
        output = {
            "predictions": torch.sigmoid(logits),
            "regularization_loss": regularization_loss,
            **details,
        }
        return output if return_details else (
            output["predictions"], regularization_loss
        )
