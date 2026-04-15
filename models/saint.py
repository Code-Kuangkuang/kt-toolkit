import torch
import torch.nn as nn
from torch.nn import Module, Embedding, Linear, Dropout
import copy
import pandas as pd
from torch.nn import Sequential, ReLU

from core.registry import MODEL_REGISTRY

device = "cpu" if not torch.cuda.is_available() else "cuda"


class transformer_FFN(Module):
    def __init__(self, emb_size, dropout) -> None:
        super().__init__()
        self.emb_size = emb_size
        self.dropout = dropout
        self.FFN = Sequential(
            Linear(self.emb_size, self.emb_size),
            ReLU(),
            Dropout(self.dropout),
            Linear(self.emb_size, self.emb_size),
        )

    def forward(self, in_fea):
        return self.FFN(in_fea)


def ut_mask(seq_len):
    """Upper Triangular Mask"""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(device)


def pos_encode(seq_len):
    """Position Encoding"""
    return torch.arange(seq_len).unsqueeze(0).to(device)


def get_clones(module, N):
    """Cloning nn modules"""
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class Encoder_block(nn.Module):
    def __init__(self, dim_model, heads_en, total_ex, total_cat, seq_len, dropout, emb_path="", pretrain_dim=768):
        super().__init__()
        self.seq_len = seq_len
        self.emb_path = emb_path
        self.total_cat = total_cat
        self.total_ex = total_ex

        if total_ex > 0:
            if emb_path == "":
                self.embd_ex = nn.Embedding(total_ex, embedding_dim=dim_model)
            else:
                embs = pd.read_pickle(emb_path)
                self.exercise_embed = Embedding.from_pretrained(embs)
                self.linear = Linear(pretrain_dim, dim_model)

        if total_cat > 0:
            self.emb_cat = nn.Embedding(total_cat, embedding_dim=dim_model)

        self.multi_en = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_en, dropout=dropout)
        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.dropout1 = Dropout(dropout)

        self.ffn_en = transformer_FFN(dim_model, dropout)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.dropout2 = Dropout(dropout)

    def forward(self, in_ex, in_cat, in_pos, first_block=True):
        if first_block:
            embs = []
            if self.total_ex > 0:
                if self.emb_path == "":
                    in_ex = self.embd_ex(in_ex)
                else:
                    in_ex = self.linear(self.exercise_embed(in_ex))
                embs.append(in_ex)
            if self.total_cat > 0:
                in_cat = self.emb_cat(in_cat)
                embs.append(in_cat)
            out = embs[0]
            for i in range(1, len(embs)):
                out += embs[i]
            out = out + in_pos
        else:
            out = in_ex

        out = out.permute(1, 0, 2)
        n, _, _ = out.shape
        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_en(out, out, out, attn_mask=ut_mask(seq_len=n))
        out = self.dropout1(out)
        out = out + skip_out

        out = out.permute(1, 0, 2)
        out = self.layer_norm2(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout2(out)
        out = out + skip_out

        return out


class Decoder_block(nn.Module):
    def __init__(self, dim_model, total_res, heads_de, seq_len, dropout):
        super().__init__()
        self.seq_len = seq_len
        self.embd_res = nn.Embedding(total_res + 1, embedding_dim=dim_model)

        self.multi_de1 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de, dropout=dropout)
        self.multi_de2 = nn.MultiheadAttention(embed_dim=dim_model, num_heads=heads_de, dropout=dropout)
        self.ffn_en = transformer_FFN(dim_model, dropout)

        self.layer_norm1 = nn.LayerNorm(dim_model)
        self.layer_norm2 = nn.LayerNorm(dim_model)
        self.layer_norm3 = nn.LayerNorm(dim_model)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

    def forward(self, in_res, in_pos, en_out, first_block=True):
        if first_block:
            in_in = self.embd_res(in_res)
            out = in_in + in_pos
        else:
            out = in_res

        out = out.permute(1, 0, 2)
        n, _, _ = out.shape

        out = self.layer_norm1(out)
        skip_out = out
        out, attn_wt = self.multi_de1(out, out, out, attn_mask=ut_mask(seq_len=n))
        out = self.dropout1(out)
        out = skip_out + out

        en_out = en_out.permute(1, 0, 2)
        en_out = self.layer_norm2(en_out)
        skip_out = out
        out, attn_wt = self.multi_de2(out, en_out, en_out, attn_mask=ut_mask(seq_len=n))
        out = self.dropout2(out)
        out = out + skip_out

        out = out.permute(1, 0, 2)
        out = self.layer_norm3(out)
        skip_out = out
        out = self.ffn_en(out)
        out = self.dropout3(out)
        out = out + skip_out

        return out


@MODEL_REGISTRY.register("saint")
class SAINT(nn.Module):
    def __init__(
        self,
        num_q,
        num_c,
        seq_len=200,
        emb_size=256,
        num_attn_heads=8,
        dropout=0.2,
        n_blocks=2,
        emb_type='qid',
        emb_path="",
        pretrain_dim=768,
        **kwargs
    ):
        super().__init__()
        self.model_name = "saint"
        self.num_q = num_q
        self.num_c = num_c
        self.seq_len = seq_len
        self.emb_type = emb_type
        self.num_en = n_blocks
        self.num_de = n_blocks

        self.embd_pos = nn.Embedding(seq_len, embedding_dim=emb_size)

        if emb_type.startswith("qid"):
            self.encoder = get_clones(
                Encoder_block(emb_size, num_attn_heads, num_q, num_c, seq_len, dropout),
                self.num_en
            )

        self.decoder = get_clones(Decoder_block(emb_size, 2, num_attn_heads, seq_len, dropout), self.num_de)

        self.dropout = Dropout(dropout)
        self.out = nn.Linear(in_features=emb_size, out_features=1)

    def forward(self, data, return_details=False):
        qseqs = data["qseqs"]
        cseqs = data.get("cseqs")
        rseqs = data["rseqs"]
        qshft = data["shft_qseqs"]
        cshft = data.get("shft_cseqs")
        rshft = data["shft_rseqs"]
        sm = data["smasks"]

        in_ex = qshft
        in_cat = cshft if cshft is not None else qshft
        # Response ids are stored as float for BCE targets in dataset; embeddings require integer indices.
        in_res = rshft.long()

        if self.num_q > 0:
            in_pos = pos_encode(in_ex.shape[1])
        else:
            in_pos = pos_encode(in_cat.shape[1])
        in_pos = self.embd_pos(in_pos)

        first_block = True
        for i in range(self.num_en):
            if i >= 1:
                first_block = False
            if self.emb_type == "qid":
                in_ex = self.encoder[i](in_ex, in_cat, in_pos, first_block=first_block)
            in_cat = in_ex

        start_token = torch.tensor([[2]], dtype=in_res.dtype, device=in_res.device).repeat(in_res.shape[0], 1)
        # Teacher forcing input: [START, r_1, ..., r_{t-1}] so decoder length stays aligned with encoder/targets.
        in_res = torch.cat((start_token, in_res[:, :-1]), dim=-1)

        first_block = True
        for i in range(self.num_de):
            if i >= 1:
                first_block = False
            in_res = self.decoder[i](in_res, in_pos, en_out=in_ex, first_block=first_block)

        res = self.out(self.dropout(in_res))
        res = torch.sigmoid(res).squeeze(-1)

        if return_details:
            return res, sm
        return res
