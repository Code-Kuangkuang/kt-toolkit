import math
from enum import IntEnum

import numpy as np
import torch
from torch import nn
from torch.nn.init import xavier_uniform_
import torch.nn.functional as F

from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings


class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


@MODEL_REGISTRY.register("hqaf")
@MODEL_REGISTRY.register("hqaf_kt")
class HQAFKT(nn.Module):
    class Inputs(InputSpec):
        """Difficulty, response-time and question-type attributes.

        The difficulty half rides DIMKT's `difficulty_maps` dataset argument.
        That aliasing was implicit in the runner; it is spelled out here.
        """

        dataset_mode = "all_in_one"
        requires_question_ids = True
        needs_num_pid = True

        @classmethod
        def prepare(cls, ctx):
            from datasets.feature_utils import compute_hqaf_feature_maps

            diff_level = int(ctx.model_cfg.get(
                "diff_level", ctx.model_cfg.get("difficult_levels", 50)
            ))
            num_time_bins = int(ctx.model_cfg.get("num_time_bins", 20))
            num_type = int(ctx.dataset_cfg.get(
                "num_type", ctx.model_cfg.get("num_type", 16)
            ))
            maps = compute_hqaf_feature_maps(
                ctx.dataset_cfg["dpath"],
                ctx.resolve_file(ctx.quelevel_key("train_valid_file"), "train_valid_file"),
                diff_level=diff_level,
                num_time_bins=num_time_bins,
                folds=ctx.train_folds(),
            )
            if not maps.get("has_usetimes", False):
                print("Warning: HQAF source data has no 'usetimes' column; using default time bucket 0.")
            if not maps.get("has_type", False):
                print("Warning: HQAF source data has no 'type' column; using default question type 0.")

            inputs = super().prepare(ctx)  # supplies num_pid
            inputs.model_kwargs["num_type"] = num_type
            inputs.model_cfg_updates.update({
                "diff_level": diff_level,
                "num_time_bins": num_time_bins,
                "num_type": num_type,
            })
            inputs.dataset_kwargs.update({
                "include_hqaf_attrs": True,
                "hqaf_feature_maps": maps,
                # HQAF has no difficulty channel of its own; it borrows the one
                # DIMKT established.
                "difficulty_maps": {
                    "skills": maps.get("skills", {}),
                    "questions": maps.get("questions", {}),
                },
            })
            inputs.feature_fit_scope = "train_folds"
            return inputs

    def __init__(
        self,
        n_question=None,
        n_pid=None,
        d_model=256,
        n_blocks=4,
        dropout=0.2,
        d_ff=512,
        kq_same=1,
        final_fc_dim=512,
        num_attn_heads=4,
        separate_qa=False,
        l2=1e-5,
        emb_type="qid",
        emb_path="",
        pretrain_dim=768,
        num_c=None,
        num_q=None,
        num_type=16,
        diff_level=50,
        num_time_bins=20,
        **kwargs,
    ):
        super().__init__()
        if n_question is None:
            if num_c is None:
                raise ValueError("HQAFKT requires n_question or num_c.")
            n_question = num_c
        if n_pid is None:
            if num_q is None:
                raise ValueError("HQAFKT requires n_pid or num_q.")
            n_pid = num_q

        self.model_name = "hqaf"
        self.n_question = int(n_question)
        self.n_pid = int(n_pid)
        self.dropout = dropout
        self.kq_same = kq_same
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.num_type = int(num_type or 0)
        self.diff_level = int(diff_level)
        self.num_time_bins = int(num_time_bins)

        embed_l = int(d_model)
        type_size = max(self.num_type + 1, 17)
        diff_size = max(self.diff_level + 2, 52)
        time_size = max(self.num_time_bins + 1, 21)

        self.q_embed = nn.Embedding(self.n_pid + 1, embed_l)
        self.qa_embed = nn.Embedding(2, embed_l)
        self.c_embed = nn.Embedding(self.n_question + 1, embed_l)
        self.c_embed_difficult = nn.Embedding(diff_size, embed_l)
        self.utT_embedding = nn.Embedding(time_size, embed_l)
        self.utT_embedding2 = nn.Embedding(time_size, embed_l)
        self.pt_embedding = nn.Embedding(type_size, embed_l)

        for emb in (
            self.c_embed_difficult,
            self.pt_embedding,
            self.utT_embedding,
            self.utT_embedding2,
            self.q_embed,
            self.qa_embed,
            self.c_embed,
        ):
            nn.init.xavier_uniform_(emb.weight)

        self.model = HQAFArchitecture(
            n_question=self.n_question,
            n_blocks=int(n_blocks),
            n_heads=int(num_attn_heads),
            dropout=dropout,
            d_model=embed_l * 2,
            d_feature=(embed_l * 2) / int(num_attn_heads),
            d_ff=int(d_ff),
            kq_same=self.kq_same,
            model_type=self.model_type,
            emb_type=self.emb_type,
        )

        self.out = nn.Sequential(
            nn.Linear(embed_l * 4, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
        )

        self.simFFN = nn.Sequential(
            nn.Linear(embed_l, 256),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )
        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.dim() > 0 and p.size(0) == self.n_pid + 1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.0)

    def compute_differences(self, avg_embeddings, embeddings):
        return avg_embeddings.unsqueeze(1) - embeddings.unsqueeze(2)

    def compute_similarities(self, avg_embeddings, embeddings):
        diff = self.compute_differences(avg_embeddings, embeddings)
        batch_size, seq_len, _, embed_dim = diff.shape
        diff = diff.reshape(batch_size * seq_len * seq_len, embed_dim)
        similarities = self.simFFN(diff)
        return similarities.reshape(batch_size, seq_len, seq_len)

    def _clamp_inputs(self, q_data, pid_data, cd, qu, cutT, cpT):
        q_data = q_data.long().clamp(min=0, max=self.n_question)
        pid_data = pid_data.long().clamp(min=0, max=self.n_pid)
        cd = cd.long().clamp(min=0, max=self.c_embed_difficult.num_embeddings - 1)
        qu = qu.long().clamp(min=0, max=self.utT_embedding2.num_embeddings - 1)
        cutT = cutT.long().clamp(min=0, max=self.utT_embedding.num_embeddings - 1)
        cpT = cpT.long().clamp(min=0, max=self.pt_embedding.num_embeddings - 1)
        return q_data, pid_data, cd, qu, cutT, cpT

    def forward(
        self,
        q_data,
        target,
        pid_data=None,
        qtest=False,
        cd=None,
        qd=None,
        qu=None,
        cutT=None,
        cpT=None,
        sdshft=None,
        sm=None,
        return_attn=False,
    ):
        if pid_data is None:
            raise ValueError("HQAFKT requires question id sequence as pid_data.")
        if cd is None or qu is None or cutT is None or cpT is None:
            raise ValueError("HQAFKT requires cd, qu, cutT, and cpT attribute sequences.")

        target = target.long().clamp(min=0, max=1)
        q_data, pid_data, cd, qu, cutT, cpT = self._clamp_inputs(
            q_data, pid_data, cd, qu, cutT, cpT
        )

        # q_data holds concepts and pid_data holds question ids, so only the
        # concept side is pooled.  Identity on [B,T]; on [B,T,K] mean-pools
        # the question's KCs with -1 padding masked.
        q_embed_data = self.q_embed(pid_data) + pool_concept_embeddings(
            self.c_embed, q_data, self.n_question
        )
        qa_embed_data = q_embed_data + self.qa_embed(target)

        utT_embeddings = self.utT_embedding(cutT)
        pt_embeddings = self.pt_embedding(cpT)
        avg_utT_embeddings = self.utT_embedding2(qu)
        utT_similarities = self.compute_similarities(avg_utT_embeddings, utT_embeddings)

        c_diff = self.c_embed_difficult(cd)
        c_diff_add_pt_embeddings = c_diff + pt_embeddings
        q_embed_data = torch.cat([q_embed_data, c_diff_add_pt_embeddings], dim=-1)
        qa_embed_data = torch.cat([qa_embed_data, c_diff_add_pt_embeddings], dim=-1)

        d_output, attn_weights = self.model(
            q_embed_data,
            qa_embed_data,
            utT_similarities,
            c_diff,
            return_attn=return_attn,
        )

        concat_q = torch.cat([d_output, q_embed_data], dim=-1)
        preds = torch.sigmoid(self.out(concat_q).squeeze(-1))

        if return_attn:
            return preds, attn_weights
        if qtest:
            return preds, concat_q
        return preds, None


class HQAFArchitecture(nn.Module):
    def __init__(
        self,
        n_question,
        n_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        model_type,
        emb_type,
    ):
        super().__init__()
        self.d_model = d_model
        self.model_type = model_type
        self.blocks_1 = nn.ModuleList(
            [
                HQAFTransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                    emb_type=emb_type,
                )
                for _ in range(n_blocks)
            ]
        )
        self.blocks_2 = nn.ModuleList(
            [
                HQAFTransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                    emb_type=emb_type,
                )
                for _ in range(n_blocks * 2)
            ]
        )

        self.qc1 = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model),
            nn.Tanh(),
        )
        self.qc2 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model),
            nn.Tanh(),
        )
        self.qc3 = nn.Sequential(
            nn.Linear(d_model, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, d_model),
            nn.Tanh(),
        )
        self.sc1 = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.sc2 = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())
        self.sc3 = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, q_embed_data, qa_embed_data, utT_similarities, c_diff, return_attn=False):
        y = qa_embed_data
        x = q_embed_data
        for block in self.blocks_1:
            y, _ = block(mask=1, query=y, key=y, values=y, utT_similarities=utT_similarities)

        init_x = x
        collected_attentions = []
        flag_first = True
        for i, block in enumerate(self.blocks_2):
            if flag_first:
                x, attn = block(
                    mask=1,
                    query=x,
                    key=x,
                    values=x,
                    utT_similarities=utT_similarities,
                    apply_pos=False,
                )
                if return_attn:
                    collected_attentions.append(attn)
                flag_first = False
            else:
                x, _ = block(
                    mask=0,
                    query=x,
                    key=x,
                    values=y,
                    utT_similarities=utT_similarities,
                    apply_pos=True,
                )
                flag_first = True

                if i == 1:
                    sdf = self.qc1(x - init_x)
                    sc = self.sc1(x - init_x)
                    x = sc * x + sdf * (1 - sc)
                elif i == 3:
                    sdf = self.qc2(c_diff)
                    sc = self.sc2(x - init_x)
                    x = sc * x + sdf * (1 - sc)
                elif i == 5:
                    sdf = self.qc3(x - init_x)
                    sc = self.sc3(x - init_x)
                    x = sc * x + sdf * (1 - sc)
        return x, collected_attentions


class HQAFTransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same, emb_type):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = HQAFMultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, emb_type=emb_type
        )
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, utT_similarities=None, apply_pos=True):
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)

        query2, attn_score = self.masked_attn_head(
            query,
            key,
            values,
            mask=src_mask,
            zero_pad=(mask == 0),
            utT_similarities=utT_similarities,
        )

        query = self.layer_norm1(query + self.dropout1(query2))
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = self.layer_norm2(query + self.dropout2(query2))
        return query, attn_score


class HQAFMultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True, emb_type="qid"):
        super().__init__()
        self.d_model = d_model
        self.emb_type = emb_type
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)
        self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
        torch.nn.init.xavier_uniform_(self.gammas)
        self._reset_parameters()

    def _reset_parameters(self):
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

    def forward(self, q, k, v, mask, zero_pad, utT_similarities=None):
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

        scores, output_attn = hqaf_attention(
            q,
            k,
            v,
            self.d_k,
            mask,
            self.dropout,
            zero_pad,
            utT_similarities,
            self.gammas,
        )
        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output, output_attn


def hqaf_attention(q, k, v, d_k, mask, dropout, zero_pad, utT_similarities, gamma=None):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)
    attn_device = q.device

    if utT_similarities is None:
        utT_similarities = torch.ones(bs, seqlen, seqlen, device=attn_device)
    utT_similarities_expanded = utT_similarities.to(attn_device).unsqueeze(1).expand(-1, head, -1, -1)
    scores = scores * utT_similarities_expanded

    x1 = torch.arange(seqlen, device=attn_device).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()

    with torch.no_grad():
        scores_ = scores.masked_fill(mask == 0, -1e32)
        scores_ = F.softmax(scores_, dim=-1)
        scores_ = scores_ * mask.float().to(attn_device)
        distcum_scores = torch.cumsum(scores_, dim=-1)
        disttotal_scores = torch.sum(scores_, dim=-1, keepdim=True)
        position_effect = torch.abs(x1 - x2)[None, None, :, :].float().to(attn_device)
        dist_scores = torch.clamp((disttotal_scores - distcum_scores) * position_effect, min=0.0)
        dist_scores = dist_scores.sqrt().detach()

    gamma = -1.0 * nn.Softplus()(gamma).unsqueeze(0)
    total_effect = torch.clamp(torch.clamp((dist_scores * gamma).exp(), min=1e-5), max=1e5)
    scores = scores * total_effect

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    attn_weights = scores.clone()

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen, device=attn_device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)

    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output, attn_weights
