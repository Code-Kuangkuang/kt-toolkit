import torch
import torch.nn as nn

from core.model_inputs import InputSpec
from core.registry import MODEL_REGISTRY
from models.hdkt_core import HybridInteractionDenoiser
from models.simplekt import SimpleKT
from models.multi_concept import pool_concept_embeddings


@MODEL_REGISTRY.register("hd_simplekt")
class HDSimpleKT(nn.Module):
    class Inputs(InputSpec):
        """Declares what this model needs; it derives nothing from the data."""

        dataset_mode = "all_in_one"

    """SimpleKT backbone augmented with causal HD-KT interaction denoising."""

    def __init__(
        self,
        num_c,
        num_q=0,
        num_pid=None,
        emb_size=256,
        num_blocks=2,
        dropout=0.1,
        d_ff=256,
        num_attn_heads=8,
        seq_len=200,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        separate_qa=False,
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
        self.model_name = "hd_simplekt"
        self.reconstruction_weight = float(reconstruction_weight)
        self.backbone = SimpleKT(
            num_c=num_c,
            num_q=num_q,
            num_pid=num_pid,
            emb_size=emb_size,
            num_blocks=num_blocks,
            dropout=dropout,
            d_ff=d_ff,
            num_attn_heads=num_attn_heads,
            seq_len=seq_len,
            kq_same=kq_same,
            final_fc_dim=final_fc_dim,
            final_fc_dim2=final_fc_dim2,
            separate_qa=separate_qa,
            emb_type=emb_type,
        )
        self.num_pid = self.backbone.num_pid
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
        self,
        qseqs,
        rseqs,
        cseqs,
        qshft,
        cshft,
        rshft,
        pidseqs=None,
        pidshft=None,
        valid_mask=None,
        return_details=False,
        **kwargs,
    ):
        q = qseqs.long() if qseqs is not None else None
        concept = cseqs.long() if cseqs is not None else q
        shifted_concept = cshft.long() if cshft is not None else qshft.long()
        if concept.dim() not in (2, 3) or shifted_concept.dim() != concept.dim():
            raise ValueError(
                "HD-SimpleKT concepts must align as [B,T] or [B,T,K]."
            )
        responses = torch.cat((rseqs[:, 0:1], rshft), dim=1).long()
        concept_data = torch.cat(
            (concept[:, 0:1], shifted_concept), dim=1
        )

        if pidseqs is not None:
            pid = pidseqs.long()
            shifted_pid = pidshft.long() if pidshft is not None else qshft.long()
        else:
            pid = q
            shifted_pid = qshft.long() if qshft is not None else None
        pid_data = (
            torch.cat((pid[:, 0:1], shifted_pid), dim=1)
            if pid is not None and shifted_pid is not None
            else None
        )
        items = concept_data if pid_data is None else pid_data
        details = self.denoiser(
            items, concept_data, responses, valid_mask=valid_mask
        )

        q_embed_data, qa_embed_data = self.backbone.base_emb(
            concept_data, responses
        )
        if self.backbone.num_pid > 0 and self.backbone.emb_type.find("norasch") == -1:
            if pid_data is None:
                raise ValueError("HD-SimpleKT Rasch path requires question ids.")
            q_embed_diff = pool_concept_embeddings(
                self.backbone.q_embed_diff, concept_data, self.backbone.num_c
            )
            pid_embed = self.backbone.difficult_param(pid_data)
            q_embed_data = q_embed_data + pid_embed * q_embed_diff
            if self.backbone.emb_type.find("aktrasch") != -1:
                qa_embed_diff = self.backbone.qa_embed_diff(responses)
                qa_embed_data = qa_embed_data + pid_embed * (
                    qa_embed_diff + q_embed_diff
                )

        qa_embed_data = qa_embed_data * details["gate"].unsqueeze(-1)
        hidden = self.backbone.model(q_embed_data, qa_embed_data)
        logits = self.backbone.out(
            torch.cat((hidden, q_embed_data), dim=-1)
        ).squeeze(-1)
        output = {"predictions": torch.sigmoid(logits), **details}
        return output if return_details else output["predictions"]
