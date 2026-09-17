import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings


class Dim:
    batch = 0
    seq = 1
    feature = 2


@MODEL_REGISTRY.register("simplekt")
class SimpleKT(nn.Module):
    def __init__(
        self,
        num_c,
        num_q,
        num_pid=None,
        emb_size=None,
        num_blocks=None,
        dropout=0.2,
        d_ff=256,
        num_layers=2,
        num_attn_heads=8,
        seq_len=200,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        separate_qa=False,
        emb_type="qid",
        **kwargs
    ):
        super().__init__()
        if emb_size is None:
            emb_size = kwargs.pop("d_model", 128)
        if num_blocks is None:
            num_blocks = kwargs.pop("n_blocks", 2)
        if num_pid is None:
            num_pid = num_q

        self.model_name = "simplekt"
        self.num_c = num_c
        self.num_q = num_q
        self.num_pid = num_pid
        self.emb_size = emb_size
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.emb_type = emb_type

        embed_l = emb_size

        # Problem ID embedding (difficulty)
        if self.num_pid > 0:
            if emb_type.find("scalar") != -1:
                self.difficult_param = nn.Embedding(self.num_pid + 1, 1)
            else:
                self.difficult_param = nn.Embedding(self.num_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.num_c + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_c + 1, embed_l)

        # Question embedding
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.num_c, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.num_c + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        # Transformer architecture
        self.model = SimpleKTArchitecture(
            num_c=num_c,
            num_blocks=num_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=emb_size,
            d_feature=emb_size // num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            seq_len=seq_len,
        )

        self.out = nn.Sequential(
            nn.Linear(emb_size + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_pid + 1 and self.num_pid > 0:
                torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        q_embed_data = pool_concept_embeddings(
            self.q_embed, q_data, self.num_c
        )
        if self.separate_qa:
            qa_embed_data = pool_interaction_embeddings(
                self.qa_embed, q_data, target, self.num_c
            )
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

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
        return_features=False,
        **kwargs,
    ):
        q = qseqs.long() if qseqs is not None else None
        c = cseqs.long() if cseqs is not None else q
        if c is None:
            raise ValueError("SimpleKT requires concept sequences or question sequences.")
        r = rseqs.long()
        qshft = qshft.long() if qshft is not None else None
        cshft = cshft.long() if cshft is not None else qshft
        if cshft is None:
            raise ValueError("SimpleKT requires shifted concept or question sequences.")
        rshft = rshft.long()

        # Match pykt: concepts drive base embeddings, questions drive problem difficulty.
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)

        # Base embeddings
        if self.emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        # Add problem difficulty
        if self.num_pid > 0 and self.emb_type.find("norasch") == -1:
            if pidseqs is not None:
                pid = pidseqs.long()
                next_pid = pidshft.long() if pidshft is not None else qshft
            else:
                pid = q
                next_pid = qshft
            if pid is None or next_pid is None:
                raise ValueError(
                    "SimpleKT Rasch difficulty requires qseqs/shft_qseqs "
                    "or pidseqs/shft_pidseqs. Set num_pid=0 for concept-only data."
                )
            pid_data = torch.cat((pid[:, 0:1], next_pid), dim=1)
            if self.emb_type.find("aktrasch") == -1:
                q_embed_diff_data = pool_concept_embeddings(
                    self.q_embed_diff, q_data, self.num_c
                )
                pid_embed_data = self.difficult_param(pid_data)
                q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            else:
                q_embed_diff_data = pool_concept_embeddings(
                    self.q_embed_diff, q_data, self.num_c
                )
                pid_embed_data = self.difficult_param(pid_data)
                q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )

        # Pass through transformer
        d_output = self.model(q_embed_data, qa_embed_data)

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        output = self.out(concat_q).squeeze(-1)

        preds = torch.sigmoid(output)
        if return_features:
            return {
                "preds": preds,
                "logits": output,
                "hidden": d_output,
                "question_embed": q_embed_data,
            }
        return preds


class SimpleKTArchitecture(nn.Module):
    def __init__(
        self,
        num_c,
        num_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        seq_len,
    ):
        super().__init__()
        self.d_model = d_model

        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(num_blocks)
            ]
        )
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        seqlen = q_embed_data.size(1)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        # Encoder
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        if mask == 0:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            nn.init.xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            nn.init.constant_(self.k_linear.bias, 0.0)
            nn.init.constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                nn.init.constant_(self.q_linear.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(scores.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]
