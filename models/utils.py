import copy

import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, ReLU, Sequential


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class transformer_FFN(nn.Module):
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
    """Upper triangular mask."""
    return torch.triu(torch.ones(seq_len, seq_len), diagonal=1).to(dtype=torch.bool).to(device)


def lt_mask(seq_len):
    """Lower triangular mask."""
    return torch.tril(torch.ones(seq_len, seq_len), diagonal=-1).to(dtype=torch.bool).to(device)


def pos_encode(seq_len):
    """Position encoding indices."""
    return torch.arange(seq_len).unsqueeze(0).to(device)


def get_clones(module, n):
    """Cloning nn modules."""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])
