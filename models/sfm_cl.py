"""SFM_CL, the contrastive graph front-end HCGKT is built on.

Source: pykt-team/pykt-toolkit, pykt/models/SFM_CL_model.py (fetched
2026-09-18), kept as its own module as upstream has it.

Two changes: the two Google-Drive-only artefacts become constructor arguments
(models/kc_graph_utils.py builds or loads them), and the module-level
`device = "cuda" if available` with its scattered `.to(device)` calls is gone --
submodules follow their parent under `model.to(...)`, and the global only pinned
tensors to whatever was visible at import time.

`BGRL.forward` returns a zero contrastive loss when `perb` is None, so the
contrastive half of this model exists only inside the FLAG-style adversarial
loop in core/trainers/hcgkt_trainer.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy



class GCNConv(nn.Module):  
    def __init__(self, in_dim, out_dim, p):
        super(GCNConv, self).__init__()

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
        x = torch.sparse.mm(adj.float(), x)
        x = x + self.b
        return x


class MLP_Predictor(nn.Module):  
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP_Predictor, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size, bias=True),
            nn.BatchNorm1d(hidden_size),
            nn.PReLU(),
            nn.Linear(hidden_size, output_size, bias=True)
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.reset_parameters()

    def forward(self, x):
        return self.net(x)


def loss_fn(x, y):  
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


def get_kc_embedding(last_kc, kc_emb, padding_idx=-1):
    batch_size, seq_len, max_concepts = last_kc.shape
    dim = kc_emb.size(1)
    device = last_kc.device
    
    mask = (last_kc != padding_idx)  # [batch_size, seq_len, max_concepts]
    
    last_kc = last_kc.clamp(min=0)
    
    flat_kc = last_kc.view(-1)  # [batch_size * seq_len * max_concepts]
    flat_emb = F.embedding(flat_kc, kc_emb)  # [batch_size * seq_len * max_concepts, dim]

    emb = flat_emb.view(batch_size, seq_len, max_concepts, dim)  # [batch_size, seq_len, max_concepts, dim]
    
    mask = mask.unsqueeze(-1).expand(-1, -1, -1, dim)  # [batch_size, seq_len, max_concepts, dim]
    
    masked_emb = emb * mask.float()
    
    concept_counts = mask.sum(dim=2, keepdim=True)  # [batch_size, seq_len, 1, dim]
    
    concept_counts = concept_counts.clamp(min=1.0)
    
    pooled_emb = masked_emb.sum(dim=2) / concept_counts.squeeze(2)  # [batch_size, seq_len, dim]

    pooled_emb = pooled_emb.squeeze(0)  # [seq_len, dim]

    return pooled_emb  

class BGRL(nn.Module):  
    def __init__(self, d, p):
        super(BGRL, self).__init__()

        self.online_encoder = GCNConv(d, d, p)  

        self.decoder = GCNConv(d, d, p)

        self.predictor = MLP_Predictor(d, d, d)

        self.target_encoder = copy.deepcopy(self.online_encoder)

        self.fc1 = nn.Linear(d, d)
        self.fc2 = nn.Linear(d, d)

        for param in self.target_encoder.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def update_target_network(self, mm):
        for param_q, param_k in zip(self.online_encoder.parameters(), self.target_encoder.parameters()):
            param_k.data.mul_(mm).add_(param_q.data, alpha=1. - mm)


    def forward(self, x, adj, perb=None):
        if perb is None:  
            return (x + self.online_encoder(x, adj)), torch.zeros((), device=x.device)

        x1, adj1 = x, copy.deepcopy(adj)
        x2, adj2 = (x + perb), copy.deepcopy(adj)

        embed = (x2 + self.online_encoder(x2, adj2))   

        online_x = self.online_encoder(x1, adj1)
        online_y = self.online_encoder(x2, adj2)

        with torch.no_grad():
            target_y = self.target_encoder(x1, adj1).detach()
            target_x = self.target_encoder(x2, adj2).detach()

        online_x = self.predictor(online_x)
        online_y = self.predictor(online_y)

        loss = (loss_fn(online_x, target_x) + loss_fn(online_y, target_y)).mean()

        return embed, loss



class SFM_CL(nn.Module):
    def __init__(self, skill_max, pro_max, d, p, concept_map, concept_embedding):
        super(SFM_CL, self).__init__()

        self.d_model = d
        self.pro_max = pro_max

        self.gcl = BGRL(d, p)

        self.gcn = GCNConv(d, d, p)

        self.pro_embed = nn.Parameter(torch.ones((pro_max, d)))  
        self.ans_embed = nn.Embedding(2, d)  
        self.change = nn.Linear(concept_embedding.size(1), d)

        # Upstream read `question_concept_map.npy` and
        # `kc_embeddings_<dataset>_bge.npy` from two Google-Drive folders here.
        # The map is rebuilt from the Q-matrix; the embeddings are a downloaded
        # artefact under utils/kc_embedding/. models/kc_graph_utils.py covers
        # both, including the index-alignment check.
        concept_map = concept_map.unsqueeze(0)  # [1, pro_max, max_concept]
        pro_embedding_kc = get_kc_embedding(concept_map, concept_embedding)
        pro_embedding_kc = self.change(pro_embedding_kc)  # [pro_max, d]
        self.pro_embed = nn.Parameter(pro_embedding_kc * self.pro_embed)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
                

    def forward(self, last_pro, last_ans, next_pro, matrix, perb=None):

        pro_embed, contrast_loss = self.gcl(self.pro_embed, matrix, perb)
        contrast_loss = 0.1 * contrast_loss  

        last_pro_embed = F.embedding(last_pro, pro_embed)  # [80,199,128]
        next_pro_embed = F.embedding(next_pro, pro_embed)  # [80,199,128]

        ans_embed = self.ans_embed(last_ans)

        X = last_pro_embed
        
        return X, next_pro_embed, contrast_loss

