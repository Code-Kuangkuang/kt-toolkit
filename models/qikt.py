import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from core.registry import MODEL_REGISTRY


class QueEmb(nn.Module):
    """Question and Concept Embedding layer for QIKT model.

    Supports different embedding types:
    - qid: question id only
    - iekt: question + concept average embedding
    """
    def __init__(self, num_q, num_c, emb_size, model_name, device='cpu', emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.model_name = model_name
        self.emb_type = emb_type

        if emb_type == "iekt":
            self.que_emb = nn.Embedding(num_q, emb_size)
            self.concept_emb = nn.Parameter(torch.randn(num_c, emb_size), requires_grad=True)
            self.que_c_linear = nn.Linear(2 * emb_size, emb_size)
        elif emb_type.startswith("qid"):
            self.que_emb = nn.Embedding(num_q, emb_size)

    def get_avg_skill_emb(self, c):
        """Get average concept embedding."""
        # Create on same device as concept_emb
        concept_emb_cat = torch.cat(
            [torch.zeros(1, self.emb_size, device=self.concept_emb.device), self.concept_emb], dim=0
        )
        related_concepts = (c + 1).long()
        concept_emb = concept_emb_cat[related_concepts, :]

        # Support both single-concept ids [B, T] and multi-concept ids [B, T, K].
        if related_concepts.dim() == 2:
            return concept_emb

        concept_emb_sum = concept_emb.sum(dim=-2)
        concept_num = (related_concepts != 0).sum(dim=-1, keepdim=True).clamp(min=1)
        concept_avg = concept_emb_sum / concept_num.to(concept_emb_sum.dtype)
        return concept_avg

    def forward(self, q, c, r=None):
        emb_type = self.emb_type
        if emb_type == "iekt":
            emb_c = self.get_avg_skill_emb(c)
            emb_q = self.que_emb(q)
            emb_qc = torch.cat([emb_q, emb_c], dim=-1)
            xemb = self.que_c_linear(emb_qc)
            emb_qca = torch.cat([
                emb_qc.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size * 2)),
                emb_qc.mul(r.unsqueeze(-1).repeat(1, 1, self.emb_size * 2))
            ], dim=-1)
            return xemb, emb_qca, emb_qc, emb_q, emb_c
        elif emb_type.startswith("qid"):
            return self.que_emb(q)


class MLP(nn.Module):
    """MLP classifier decoder."""
    def __init__(self, n_layer, hidden_dim, output_dim, dpo):
        super().__init__()
        self.lins = nn.ModuleList([
            nn.Linear(hidden_dim, hidden_dim)
            for _ in range(n_layer)
        ])
        self.dropout = nn.Dropout(p=dpo)
        self.out = nn.Linear(hidden_dim, output_dim)
        self.act = torch.nn.Sigmoid()

    def forward(self, x):
        for lin in self.lins:
            x = F.relu(lin(x))
        return self.out(self.dropout(x))


def get_outputs(self, emb_qc_shift, h, data, add_name="", model_type='question'):
    """Get model outputs for question or concept prediction."""
    outputs = {}

    if model_type == 'question':
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_question_next = torch.sigmoid(self.out_question_next(h_next))
        y_question_all = torch.sigmoid(self.out_question_all(h))
        outputs["y_question_next" + add_name] = y_question_next.squeeze(-1)
        outputs["y_question_all" + add_name] = (y_question_all * F.one_hot(data['qshft'].long(), self.num_q)).sum(-1)
    else:
        h_next = torch.cat([emb_qc_shift, h], axis=-1)
        y_concept_next = torch.sigmoid(self.out_concept_next(h_next))
        y_concept_all = torch.sigmoid(self.out_concept_all(h))
        outputs["y_concept_next" + add_name] = self.get_avg_fusion_concepts(y_concept_next, data['cshft'])
        outputs["y_concept_all" + add_name] = self.get_avg_fusion_concepts(y_concept_all, data['cshft'])

    return outputs


@MODEL_REGISTRY.register("qikt")
class QIKTNet(nn.Module):
    def __init__(self, num_q, num_c, emb_size, dropout=0.1, emb_type='qid', emb_path="", pretrain_dim=768, device='cpu', mlp_layer_num=1, other_config=None):
        super().__init__()
        if other_config is None:
            other_config = {}

        self.model_name = "qikt"
        self.num_q = num_q
        self.num_c = num_c
        self.emb_size = emb_size
        self.hidden_size = emb_size
        self.mlp_layer_num = mlp_layer_num
        self.device = device
        self.other_config = other_config
        self.output_mode = self.other_config.get('output_mode', 'an')

        self.emb_type = emb_type
        self.que_emb = QueEmb(
            num_q=num_q, num_c=num_c, emb_size=emb_size, emb_type=self.emb_type,
            model_name=self.model_name, device=device, emb_path=emb_path, pretrain_dim=pretrain_dim
        )

        self.que_lstm_layer = nn.LSTM(self.emb_size * 4, self.hidden_size, batch_first=True)
        self.concept_lstm_layer = nn.LSTM(self.emb_size * 2, self.hidden_size, batch_first=True)
        self.dropout_layer = nn.Dropout(dropout)

        self.out_question_next = MLP(self.mlp_layer_num, self.hidden_size * 3, 1, dropout)
        self.out_question_all = MLP(self.mlp_layer_num, self.hidden_size, num_q, dropout)
        self.out_concept_next = MLP(self.mlp_layer_num, self.hidden_size * 3, num_c, dropout)
        self.out_concept_all = MLP(self.mlp_layer_num, self.hidden_size, num_c, dropout)
        self.que_disc = MLP(self.mlp_layer_num, self.hidden_size * 2, 1, dropout)

    def get_avg_fusion_concepts(self, y_concept, cshft):
        """Get fused concept prediction results."""
        if cshft.dim() == 2:
            concept_ids = torch.where(cshft != -1, cshft, 0).long()
            return y_concept.gather(dim=-1, index=concept_ids.unsqueeze(-1)).squeeze(-1)

        max_num_concept = cshft.shape[-1]
        concept_mask = torch.where(cshft.long() == -1, False, True)
        concept_index = F.one_hot(torch.where(cshft != -1, cshft, 0), self.num_c)
        concept_sum = (y_concept.unsqueeze(2).repeat(1, 1, max_num_concept, 1) * concept_index).sum(-1)
        concept_sum = concept_sum * concept_mask
        y_concept = concept_sum.sum(-1) / torch.where(concept_mask.sum(-1) != 0, concept_mask.sum(-1), 1)
        return y_concept

    def forward(self, q, c, r, data=None):
        _, emb_qca, emb_qc, emb_q, emb_c = self.que_emb(q, c, r)

        emb_qc_shift = emb_qc[:, 1:, :]
        emb_qca_current = emb_qca[:, :-1, :]

        # Question model
        que_h = self.dropout_layer(self.que_lstm_layer(emb_qca_current)[0])
        que_outputs = get_outputs(self, emb_qc_shift, que_h, data, add_name="", model_type="question")
        outputs = que_outputs

        # Concept model
        emb_ca = torch.cat([
            emb_c.mul((1 - r).unsqueeze(-1).repeat(1, 1, self.emb_size)),
            emb_c.mul(r.unsqueeze(-1).repeat(1, 1, self.emb_size))
        ], dim=-1)
        emb_ca_current = emb_ca[:, :-1, :]
        concept_h = self.dropout_layer(self.concept_lstm_layer(emb_ca_current)[0])
        concept_outputs = get_outputs(self, emb_qc_shift, concept_h, data, add_name="", model_type="concept")
        outputs['y_concept_all'] = concept_outputs['y_concept_all']
        outputs['y_concept_next'] = concept_outputs['y_concept_next']

        return outputs
