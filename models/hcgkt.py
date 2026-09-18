"""HCGKT port for KT-Toolkit.

    Hierarchical Contrastive Graph Knowledge Tracing with Multi-level Feature
    Learning.

Source: pykt-team/pykt-toolkit, pykt/models/hcgkt.py (fetched 2026-09-18).

Three artefacts upstream loads from Google Drive now arrive through the
InputSpec: the question-question adjacency and the question-concept map are
rebuilt from the dataset Q-matrix, and the concept-text BGE embeddings are a
downloaded file under utils/kc_embedding/. models/kc_graph_utils.py documents
what each one is, and records that the embeddings' concept indexing was checked
against this repo's keyid2idx.json rather than assumed.

Training is not a plain forward: HCGKT wants the FLAG-style adversarial loop in
core/trainers/hcgkt_trainer.py. Without it `perb` stays None, and SFM_CL's
contrastive branch -- half of what the paper proposes -- never runs.
"""

import torch
from torch import nn
from torch.nn.init import xavier_uniform_
from torch.nn.init import constant_
import math
import torch.nn.functional as F
from enum import IntEnum
import numpy as np
from .utils import transformer_FFN, ut_mask, pos_encode, get_clones
from core.model_inputs import InputSpec
from core.registry import MODEL_REGISTRY
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings
from torch.nn import Module, Embedding, LSTM, Linear, Dropout, LayerNorm, TransformerEncoder, TransformerEncoderLayer, \
        MultiLabelMarginLoss, MultiLabelSoftMarginLoss, CrossEntropyLoss, BCELoss, MultiheadAttention
from torch.nn.functional import one_hot, cross_entropy, multilabel_margin_loss, binary_cross_entropy
from .sfm_cl import SFM_CL



class Dim(IntEnum):
    batch = 0
    seq = 1
    feature = 2

class HCGKT(nn.Module):
    def __init__(self, n_question, n_pid, 
            d_model, n_blocks, dropout, d_ff=256, 
            loss1=0.5, loss2=0.5, loss3=0.5, start=50, num_layers=2, nheads=4, seq_len=200, 
            kq_same=1, final_fc_dim=512, final_fc_dim2=256, num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid", emb_path="", pretrain_dim=768, matrix=None,
            concept_map=None, concept_embedding=None, K=3, dropout_sfm=0.4,
            step_size=None, step_m=None, grad_clip=None, mm=None):
        super().__init__()
        """
        Input:
            d_model: dimension of attention block
            final_fc_dim: dimension of final fully connected net before prediction
            num_attn_heads: number of heads in multi-headed attention
            d_ff : dimension for fully conntected net inside the basic block
            kq_same: if key query same, kq_same=1, else = 0
        """
        self.model_name = "hcgkt"
        self.n_question = n_question
        self.dropout = dropout
        self.dropout_sfm = dropout_sfm
        self.kq_same = kq_same
        self.n_pid = n_pid
        self.l2 = l2
        self.model_type = self.model_name
        self.separate_qa = separate_qa
        self.emb_type = emb_type
        embed_l = d_model


        # Upstream picked a dataset by substring-matching an abbreviation inside
        # emb_type ('as09', 'al05', ...) and torch.load()ed a Google-Drive-only
        # `ques_skill_gcn_adj.pt` from a hardcoded relative path. The runner now
        # supplies all three artefacts through this model's InputSpec; see
        # models/kc_graph_utils.py.
        if matrix is None or concept_map is None or concept_embedding is None:
            raise ValueError(
                "HCGKT needs its question graph, question-concept map and "
                "concept-text embeddings; HCGKT.Inputs.prepare builds them, so "
                "constructing the net directly means passing them yourself."
            )

        self.K = K  
        self.w_K = nn.Parameter(torch.zeros(num_attn_heads))
        self.w_0 = nn.Parameter(torch.tensor(0.1))  


        if self.n_pid > 0:
            if emb_type.find("scalar") != -1:
                self.difficult_param = nn.Embedding(self.n_pid+1, 1) 
            else:
                self.difficult_param = nn.Embedding(self.n_pid+1, embed_l) 
            self.q_embed_diff = nn.Embedding(self.n_question+1, embed_l) 
            self.qa_embed_diff = nn.Embedding(2 * self.n_question + 1, embed_l) 
        
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.n_question, embed_l)
            if self.separate_qa: 
                    self.qa_embed = nn.Embedding(2*self.n_question+1, embed_l)
            else: # false default
                self.qa_embed = nn.Embedding(2, embed_l)
        self.model = Architecture(n_question=n_question, n_blocks=n_blocks, n_heads=num_attn_heads, dropout=dropout,
                                    d_model=d_model, d_feature=d_model / num_attn_heads, d_ff=d_ff,  kq_same=self.kq_same, model_type=self.model_type, seq_len=seq_len, K=self.K,
                                    w_k=self.w_K, w_0=self.w_0)

        self.out = nn.Sequential(
            nn.Linear(d_model + embed_l,
                      final_fc_dim), nn.ReLU(), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2), nn.ReLU(
            ), nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1)
        )

        self.register_buffer('matrix', matrix, persistent=False)
        positive_matrix = self.matrix
        pro_max = self.n_pid
        d = d_model
        p = self.dropout_sfm 

        self.sfm_cl = SFM_CL(self.n_question, pro_max, d, p, concept_map, concept_embedding)

        self.reset()

    def reset(self):
        for p in self.parameters():
            if p.ndimension() > 0 and p.size(0) == self.n_pid+1 and self.n_pid > 0:
                torch.nn.init.constant_(p, 0.)

    def base_emb(self, q_data, target):
        # Identity on [B,T]; on [B,T,K] mean-pools the question's KCs with -1
        # padding masked, as models/robustkt.py and models/stablekt.py do. HCGKT
        # shares their AKT base_emb, so it has to share their concept protocol
        # too or a row against them is not comparable.
        q_embed_data = pool_concept_embeddings(self.q_embed, q_data, self.n_question)
        if self.separate_qa:
            qa_embed_data = pool_interaction_embeddings(
                self.qa_embed, q_data, target, self.n_question
            )
        else:
            qa_embed_data = self.qa_embed(target)+q_embed_data
        return q_embed_data, qa_embed_data

    def get_attn_pad_mask(self, sm):
        batch_size, l = sm.size()
        pad_attn_mask = sm.data.eq(0).unsqueeze(1)
        pad_attn_mask = pad_attn_mask.expand(batch_size, l, l)
        return pad_attn_mask.repeat(self.nhead, 1, 1)

    def forward(self, dcur, qtest=False, train=False, perb=None):
        q, c, r = dcur["qseqs"].long(), dcur["cseqs"].long(), dcur["rseqs"].long()
        qshft, cshft, rshft = dcur["shft_qseqs"].long(), dcur["shft_cseqs"].long(), dcur["shft_rseqs"].long()
        pid_data = torch.cat((q[:,0:1], qshft), dim=1)
        q_data = torch.cat((c[:,0:1], cshft), dim=1)
        target = torch.cat((r[:,0:1], rshft), dim=1)

        emb_type = self.emb_type
        
        last_pro = q
        last_ans = r
        last_skill = c
        next_pro = qshft
        next_skill = cshft
        perb = perb

        xemb, next_xemb, contrast_loss = self.sfm_cl(
            last_pro, last_ans, next_pro, self.matrix, perb
        )
        
        xemb = xemb
        next_xemb = next_xemb
        contrast_loss = contrast_loss

        all_que_emb = torch.cat((xemb[:, 0:1], next_xemb), dim=1)


        if emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)
        if self.n_pid > 0 and emb_type.find("norasch") == -1: 
            if emb_type.find("aktrasch") == -1:   
                q_embed_diff_data = pool_concept_embeddings(self.q_embed_diff, q_data, self.n_question)  
                pid_embed_data = all_que_emb
                q_embed_data = q_embed_data + pid_embed_data 

            else:
                q_embed_diff_data = pool_concept_embeddings(self.q_embed_diff, q_data, self.n_question)  
                pid_embed_data = all_que_emb
                q_embed_data = q_embed_data + pid_embed_data * \
                    q_embed_diff_data  

                qa_embed_diff_data = self.qa_embed_diff(
                    target)  
                qa_embed_data = qa_embed_data + pid_embed_data * \
                        (qa_embed_diff_data+q_embed_diff_data)  

        y2, y3 = 0, 0
        if emb_type in ["qid", "qidaktrasch", "qid_scalar", "qid_norasch",
                        "qid_as09", "qid_ni34", "qid_al05", "qid_bd06", "qid_py", "qid_sta11"]:
            d_output = self.model(q_embed_data, qa_embed_data)

            concat_q = torch.cat([d_output, q_embed_data], dim=-1)
            output = self.out(concat_q).squeeze(-1)
            m = nn.Sigmoid()
            preds = m(output)

        if train:
            return preds, y2, y3, contrast_loss
        else:
            if qtest:
                return preds, concat_q
            else:
                return preds

class Architecture(nn.Module):
    def __init__(self, n_question,  n_blocks, d_model, d_feature,
                 d_ff, n_heads, dropout, kq_same, model_type, seq_len, K, w_k, w_0):
        super().__init__()
        """
            n_block : number of stacked blocks in the attention
            d_model : dimension of attention input/output
            d_feature : dimension of input in each of the multi-head attention part.
            n_head : number of heads. n_heads*d_feature = d_model
        """
        self.d_model = d_model
        self.model_type = model_type

        if model_type in {'simplekt', "hcgkt"}:
            self.blocks_2 = nn.ModuleList([
                TransformerLayer(d_model=d_model, d_feature=d_model // n_heads,
                                 d_ff=d_ff, dropout=dropout, n_heads=n_heads, kq_same=kq_same, K=K, w_k=w_k, w_0=w_0)
                for _ in range(n_blocks)
            ])
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
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
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True) 
        return x

class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature,
                 d_ff, n_heads, dropout,  kq_same, K, w_k, w_0):
        super().__init__()
        """
            This is a Basic Block of Transformer paper. It containts one Multi-head attention object. Followed by layer norm and postion wise feedforward net and dropout layer.
        """
        kq_same = kq_same == 1
        # Multi-Head Attention Block
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same, K=K, w_k=w_k, w_0=w_0)

        # Two layer norm layer and two droput layer
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
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
                query, key, values, mask=src_mask, zero_pad=True) 
        else:
            # Calls block.masked_attn_head.forward() method
            query2 = self.masked_attn_head(
                query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2)) 
        query = self.layer_norm1(query) # layer norm
        if apply_pos:
            query2 = self.linear2(self.dropout( # FFN
                self.activation(self.linear1(query))))
            query = query + self.dropout2((query2)) 
            query = self.layer_norm2(query) # lay norm
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, K, w_k, w_0, bias=True):
        super().__init__()
        """
        It has projection layer for getting keys, queries and values. Followed by attention and a connected layer.
        """
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.K = K
        self.w_k = w_k
        self.w_0 = w_0

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

    def forward(self, q, k, v, mask, zero_pad):

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
                           mask, self.dropout, zero_pad, self.K, self.w_k, self.w_0)

        # concatenate heads and put through final linear layer
        concat = scores.transpose(1, 2).contiguous()\
            .view(bs, -1, self.d_model)

        output = self.out_proj(concat)

        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad, K, w_K, w_0):
    """
    This is called by Multi-head atention object to find the values.
    """

    scores = torch.matmul(q, k.transpose(-2, -1)) / \
        math.sqrt(d_k)  # BS, 8, seqlen, seqlen
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)  # BS,8,seqlen,seqlen

    attention_probs = scores
    attention_probs2 = torch.matmul(attention_probs, attention_probs)

    I = torch.eye(attention_probs2.shape[2], device=scores.device)[None,None,:,:]
    attention_probsK = attention_probs + (K-1) * (attention_probs2-attention_probs)

    w_K_expanded = w_K[None,:,None,None]
    attention_probs = w_0 * I + attention_probs + w_K_expanded * attention_probsK

    attention_probs.masked_fill_(mask == 0, 0.0)

    scores = attention_probs

    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen, device=scores.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2) 

    scores = dropout(scores)
    output = torch.matmul(scores, v)
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






@MODEL_REGISTRY.register("hcgkt")
class HCGKTModel(HCGKT):
    """Registry adapter: renames the constructor arguments and supplies the artefacts."""

    class Inputs(InputSpec):
        """Needs question ids, a question graph, a concept map and concept texts."""

        dataset_mode = "all_in_one"
        requires_question_ids = True

        @classmethod
        def prepare(cls, ctx):
            import json
            import os

            import torch as _torch

            from core.model_inputs import ModelInputs
            from models.kc_graph_utils import (
                build_question_concept_map,
                build_question_graph,
                load_kc_text_embeddings,
            )

            dpath = ctx.dataset_cfg["dpath"]
            num_q = int(ctx.dataset_cfg["num_q"])
            num_c = int(ctx.dataset_cfg["num_c"])
            with open(os.path.join(dpath, "keyid2idx.json"), encoding="utf-8") as handle:
                max_concepts = int(json.load(handle)["max_concepts"])

            return ModelInputs(
                model_kwargs={
                    "matrix": build_question_graph(dpath, num_q),
                    "concept_map": _torch.tensor(
                        build_question_concept_map(dpath, num_q, max_concepts)
                    ),
                    "concept_embedding": _torch.tensor(
                        load_kc_text_embeddings(ctx.dataset_name, num_c, ctx.root_dir)
                    ),
                    "num_pid": num_q,
                },
                # The Q-matrix covers test questions, so the graph and the map
                # are built already knowing what the test split holds -- the same
                # call as dgekt's question metadata.
                run_config_extras={"graph_scope": "train_valid_test"},
                graph_scope="train_valid_test",
            )

    def __init__(
        self,
        num_c,
        num_q,
        num_pid=None,
        emb_size=None,
        num_blocks=None,
        d_model=None,
        n_blocks=None,
        dropout=0.1,
        **kwargs,
    ):
        for key in ("dpath", "num_at", "num_it", "seed", "device"):
            kwargs.pop(key, None)
        if emb_size is None:
            emb_size = d_model if d_model is not None else 256
        if num_blocks is None:
            num_blocks = n_blocks if n_blocks is not None else 2
        if num_pid is None:
            num_pid = num_q
        super().__init__(
            n_question=num_c,
            n_pid=num_pid,
            d_model=emb_size,
            n_blocks=num_blocks,
            dropout=dropout,
            **kwargs,
        )
