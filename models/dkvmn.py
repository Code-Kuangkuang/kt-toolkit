import torch

from torch.nn import Module, Parameter, Embedding, Linear, Dropout
from torch.nn.init import kaiming_normal_

from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings

@MODEL_REGISTRY.register("dkvmn")
class DKVMN(Module):
    r""" DKVMN模型初始化
    Args:
        num_c: 概念数量
        dim_s: 嵌入向量(知识状态)维度, 默认200
        size_m: 记忆网络大小, 默认50
        dropout: Dropout概率
        emb_type: 题目嵌入类型，默认为"qid"
        emb_path: 预训练题目嵌入路径，默认为空
        pretrain_dim: 预训练题目嵌入维度, 默认为768
    """
    def __init__(self, num_c, dim_s, size_m, dropout=0.2, emb_type='qid', emb_path="", pretrain_dim=768):
        super().__init__()
        self.model_name = "dkvmn"
        self.num_c = num_c
        self.dim_s = dim_s
        self.size_m = size_m
        self.emb_type = emb_type
        if emb_type.startswith("qid"):
            self.k_emb_layer = Embedding(self.num_c, self.dim_s)
            # 键记忆矩阵：存储知识概念的静态表示
            self.Mk = Parameter(torch.Tensor(self.size_m, self.dim_s))
            # 值记忆矩阵的初始值，存储知识概念的动态表示
            self.Mv0 = Parameter(torch.Tensor(self.size_m, self.dim_s))

        kaiming_normal_(self.Mk)
        kaiming_normal_(self.Mv0)

        self.v_emb_layer = Embedding(self.num_c * 2, self.dim_s)

        self.f_layer = Linear(self.dim_s * 2, self.dim_s)
        self.dropout_layer = Dropout(dropout)
        self.p_layer = Linear(self.dim_s, 1)

        self.e_layer = Linear(self.dim_s, self.dim_s)
        self.a_layer = Linear(self.dim_s, self.dim_s)

    def forward(self, q, r, qtest=False):
        emb_type = self.emb_type
        batch_size = q.shape[0]
        if emb_type == "qid":
            # q is [B,T] under one_by_one and [B,T,K] under all_in_one. Both
            # helpers are the identity on [B,T], so the concept-level path is
            # untouched; on [B,T,K] they mean-pool the valid KCs and mask the
            # -1 padding, which is what pykt's QueEmb.get_avg_skill_emb does.
            # Taking only KC 1 instead would silently drop up to max_concepts-1
            # of them, which pykt never does.
            k = pool_concept_embeddings(self.k_emb_layer, q, self.num_c)
            v = pool_interaction_embeddings(self.v_emb_layer, q, r, self.num_c)

        Mvt = self.Mv0.unsqueeze(0).repeat(batch_size, 1, 1)

        Mv = [Mvt]

        w = torch.softmax(torch.matmul(k, self.Mk.T), dim=-1)

        # Write Process
        # erase 向量 和 add 向量
        e = torch.sigmoid(self.e_layer(v))
        a = torch.tanh(self.a_layer(v))

        for et, at, wt in zip(
            e.permute(1, 0, 2), a.permute(1, 0, 2), w.permute(1, 0, 2)
        ):
            Mvt = Mvt * (1 - (wt.unsqueeze(-1) * et.unsqueeze(1))) + \
                (wt.unsqueeze(-1) * at.unsqueeze(1))
            Mv.append(Mvt)

        Mv = torch.stack(Mv, dim=1)

        # Read Process
        f = torch.tanh(
            self.f_layer(
                torch.cat(
                    [
                        (w.unsqueeze(-1) * Mv[:, :-1]).sum(-2),
                        k
                    ],
                    dim=-1
                )
            )
        )
        p = self.p_layer(self.dropout_layer(f))

        p = torch.sigmoid(p)
        # print(f"p: {p.shape}")
        p = p.squeeze(-1)
        if not qtest:
            return p
        else:
            return p, f
