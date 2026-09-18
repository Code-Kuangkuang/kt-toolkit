"""DenoiseKT port for KT-Toolkit.

    Denoised Attention and Question-Augmented Representations for Knowledge
    Tracing.

Source: pykt-team/pykt-toolkit, pykt/models/denoisekt.py (fetched 2026-09-18).

Two deliberate departures from upstream:

1. Upstream's `DenoiseKT(QueBaseModel)` wrapper is dropped. It is pykt's own
   training harness; only `DenoiseKTNet` -- the actual model -- is kept, and
   core/trainers/denoisekt_trainer.py drives it.

2. The question-question adjacency arrives as a constructor argument instead of
   `torch.load`ing a Google-Drive-only `questions_concepts.pt`. It is rebuilt
   from the dataset Q-matrix; models/kc_graph_utils.py documents the shape and
   normalisation, and which parts of that are inferred rather than stated.

The upstream module-level `device = "cuda" if available` is gone for the same
reason as in models/mockt.py: it pinned freshly built masks to CUDA regardless
of where the model lived.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd

from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
from enum import IntEnum
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from core.model_inputs import InputSpec
from core.registry import MODEL_REGISTRY
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy

import os

class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2


class DenoiseKTNet(nn.Module):
    def __init__(self, num_c, num_q, 
            d_model, n_blocks, dropout, dropout1, bf, d_ff=256, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, 
            separate_qa=False, l2=1e-5, emb_type="qid", dpath = "", emb_path="", 
            pretrain_dim=768,device='cpu',matrix=None,other_config={}):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "denoisekt"
        print(f"model_name: {self.model_name}, emb_type: {emb_type}")
        self.num_c = num_c
        self.dropout = dropout
        self.dropout1 = dropout1
        self.kq_same = kq_same
        self.num_q = num_q
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        self.emb_size = d_model
        self.device = device
        self.bf = bf
        self.num_attn_heads = num_attn_heads
        embed_l = d_model
        if self.num_q > 0:
            if emb_type.find("scalar") != -1:
                # print(f"question_difficulty is scalar")
                self.difficult_param = nn.Embedding(self.num_q+1, 1)
            else:
                self.difficult_param = nn.Embedding(self.num_q+1, embed_l) 
            

        # Upstream loaded a Google-Drive-only `questions_concepts.pt` here. The
        # runner now supplies the equivalent matrix via the model's InputSpec;
        # see models/kc_graph_utils.py for what it is and how it is inferred.
        if matrix is None:
            raise ValueError(
                "DenoiseKT needs its question-question adjacency; it is built by "
                "DenoiseKT.Inputs.prepare, so constructing the net directly means "
                "passing `matrix=` yourself."
            )
        # A buffer, not a plain attribute: `model.to(device)` then moves it once
        # instead of `GCN.forward` re-copying ~134 MB of sparse indices on every
        # batch. `persistent=False` keeps it out of the checkpoint, since it is
        # rebuilt from qmatrix.npz anyway.
        self.register_buffer("matrix", matrix, persistent=False)
        

        if emb_type.startswith("qid"):
            self.ans_embed = Embedding(2, embed_l)

        self.skill_embed = nn.Parameter(torch.rand(self.num_c, self.emb_size))
        
        nn.init.xavier_uniform_(self.skill_embed)
        self.pro_embed = nn.Parameter(torch.ones((self.num_q, self.emb_size))) 
        nn.init.xavier_uniform_(self.pro_embed)

        self.gcn = GCN(self.emb_size, self.emb_size, dropout1)


        self.model = Architecture(n_question=num_q, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        self.reset()

    def get_avg_skill_emb(self,c,emb):
        # add zero for padding
        concept_emb_cat = torch.cat(
            # Was `.to(self.device)`, a constructor argument defaulting to
            # 'cpu'; the model is moved to the run's device afterwards, so that
            # default desynchronises as soon as nothing passes `device=`. Take
            # it from the parameter this row is concatenated onto instead.
            [torch.zeros(1, self.emb_size, device=emb.device, dtype=emb.dtype),
            emb], dim=0)
        # shift c

        related_concepts = (c+1).long()
        #[batch_size, seq_len, emb_dim]
        concept_emb_sum = concept_emb_cat[related_concepts, :].sum(
            axis=-2)

        #[batch_size, seq_len,1]
        concept_num = torch.where(related_concepts != 0, 1, 0).sum(
            axis=-1).unsqueeze(-1)
        concept_num = torch.where(concept_num == 0, 1, concept_num)
        concept_avg = (concept_emb_sum / concept_num)
        return concept_avg

    def boost_focus(self,concept):
        
        batch_size, rows, cols = concept.size()

        
        cl = concept.unsqueeze(2)  
        cr = concept.unsqueeze(1)  
        resultl = cl.repeat(1, 1, rows, 1)  
        resultr = cr.repeat(1, rows, 1, 1)    
        result = torch.all(resultl == resultr, dim=-1)

        
        diag_mask = torch.eye(rows, rows, dtype=torch.bool).unsqueeze(0)
        diag_mask = diag_mask.repeat(batch_size, 1, 1)
        
        result[diag_mask] = False

        
        row_indices = torch.arange(rows).unsqueeze(1)  # shape: (rows, 1)
        col_indices = torch.arange(rows).unsqueeze(0)  # shape: (1, cols)

        
        index = (row_indices - col_indices).repeat(batch_size, 1, 1).to(concept.device)
        bf = torch.abs(result * index)
        # djw[djw == 0] = 1024
        return bf.to(concept)

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_q+1 and self.num_q > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        q_embed_data = self.q_embed(q_data)  # BS, seqlen,  d_model# c_ct
        if self.separate_qa:
            qa_data = q_data + self.num_c * target
            qa_embed_data = self.qa_embed(qa_data)
        else:
            # BS, seqlen, d_model # c_ct+ g_rt =e_(ct,rt)
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def forward(self, cq, cc, cr, perb=None):

        emb_type = self.emb_type

        if emb_type == "qid":
            contrast_loss = 0
            
            q_embed = self.gcn(self.pro_embed, self.matrix)

            q_embed_data = F.embedding(cq, q_embed)
            ans_embed_data = self.ans_embed(cr)
            qa_embed_data = q_embed_data + ans_embed_data
             
            q_embed_diff_data = self.get_avg_skill_emb(cc,self.skill_embed)
            # q_embed_diff_data = self.get_avg_skill_emb_ablation(cc,self.skill_embed)

            pid_embed_data = self.difficult_param(cq)  # uq 
            q_embed_data = q_embed_data + pid_embed_data * \
                q_embed_diff_data  # uq *d_ct + c_ct # question encoder

            boost_focus = (self.bf ** self.boost_focus(cc))
            boost_focus[boost_focus == 1] = 0
            boost_focus = boost_focus.unsqueeze(1) # shape: bs,1,seqlen,seqlen
            boost_focus = boost_focus.repeat(1, self.num_attn_heads, 1, 1) # shape: bs,head,seqlen,seqlen

        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch"]:

            d_output = self.model(q_embed_data, qa_embed_data, boost_focus)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output).squeeze(-1)
        return preds[:,1:], contrast_loss
        


class GCN(nn.Module):  
    def __init__(self, in_dim, out_dim, p):
        super(GCN, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)

        self.b = nn.Parameter(torch.rand((out_dim)))
        nn.init.zeros_(self.b)

        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)

        # print(adj.shape)
        # print(x.shape)
        # os._exit(1)

        x = torch.sparse.mm(adj.float().to(x.device), x)
        x = x + self.b
        return x



class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'denoisekt'}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data ,boost_focus):
        # target shape  bs, seqlen
        seqlen, batch_size = q_embed_data.size(1), q_embed_data.size(0)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        seqlen, batch_size = y.size(1), y.size(0)
        x = q_pos_embed

        # encoder
        
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, boost_focus=boost_focus, apply_pos=True)
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, boost_focus, apply_pos=True ):
        """
        Input:
            block : object of type BasicBlock(nn.Module). It contains masked_attn_head objects which is of type MultiHeadAttention(nn.Module).
            mask : 0 means, it can peek only past values. 1 means, block can peek only current and pas values
            query : Query. In transformer paper it is the input for both encoder and decoder
            key : Keys. In transformer paper it is the input for both encoder and decoder
            Values. In transformer paper it is the input for encoder and  encoded output for decoder (in masked attention part)

        Output:
            query: Input gets changed over the layer and returned.

        """

        seqlen, batch_size = query.size(1), query.size(0)
        nopeek_mask = np.triu(
            np.ones((1, 1, seqlen, seqlen)), k=mask).astype('uint8')
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        if mask == 0:  # If 0, zero-padding is needed.
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=True, boost_focus=boost_focus)
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False, boost_focus=boost_focus)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
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
        xavier_uniform_(self.k_linear.weight)
        xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            xavier_uniform_(self.q_linear.weight)

        if self.proj_bias:
            constant_(self.k_linear.bias, 0.)
            constant_(self.v_linear.bias, 0.)
            if self.kq_same is False:
                constant_(self.q_linear.bias, 0.)
            constant_(self.out_proj.bias, 0.)

    def forward(self, q, k, v, mask, zero_pad ,boost_focus):

        bs = q.size(0)

        # perform linear operation and split into h heads

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        # transpose to get dimensions bs * h * sl * d_model

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        # calculate attention using function we will define next
        scores = attention(q, k, v, self.d_k,
                           mask, self.dropout, zero_pad, boost_focus)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, boost_focus):#
    """
    This is called by Multi-head atention object to find the values.
    """
    
    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    
    scores = scores * (1 + boost_focus)
    
    
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(scores.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) 
    # print(f"after zero pad scores: {scores}")
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    # import sys
    # sys.exit()
    return output


class LearnablePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=True)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        # Compute the positional encodings once in log space.
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, :x.size(Dim.seq), :]  # ( 1,seq,  Feature)


@MODEL_REGISTRY.register("denoisekt")
class DenoiseKT(DenoiseKTNet):
    """Registry adapter: renames the constructor arguments and supplies the graph."""

    class Inputs(InputSpec):
        """Needs question ids and a question-question graph built from qmatrix."""

        dataset_mode = "all_in_one"
        requires_question_ids = True

        @classmethod
        def prepare(cls, ctx):
            from core.model_inputs import ModelInputs
            from models.kc_graph_utils import build_question_graph

            graph = build_question_graph(
                ctx.dataset_cfg["dpath"], int(ctx.dataset_cfg["num_q"])
            )
            return ModelInputs(
                model_kwargs={"matrix": graph, "num_q": int(ctx.dataset_cfg["num_q"])},
                # The Q-matrix covers test questions too, so the graph is built
                # already knowing what the test split contains. Same call as
                # dgekt's question metadata; see datasets/init_dataset.py.
                run_config_extras={"graph_scope": "train_valid_test"},
                graph_scope="train_valid_test",
            )

    def __init__(
        self,
        num_c,
        num_q,
        emb_size=None,
        num_blocks=None,
        d_model=None,
        n_blocks=None,
        dropout=0.1,
        dropout1=0.1,
        bf=0.5,  # pykt sweeps 0.01-0.99; >1 flips bf**distance from decay to growth
        matrix=None,
        **kwargs,
    ):
        for key in ("num_pid", "dpath", "num_at", "num_it", "seed"):
            kwargs.pop(key, None)
        if emb_size is None:
            emb_size = d_model if d_model is not None else 256
        if num_blocks is None:
            num_blocks = n_blocks if n_blocks is not None else 2
        super().__init__(
            num_c=num_c,
            num_q=num_q,
            d_model=emb_size,
            n_blocks=num_blocks,
            dropout=dropout,
            dropout1=dropout1,
            bf=bf,
            matrix=matrix,
            **kwargs,
        )
