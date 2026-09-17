# coding: utf-8
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MLP(nn.Module):
    """Two-layer fully-connected ReLU net with batch norm."""

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0., bias=True):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=bias)
        self.fc2 = nn.Linear(hidden_dim, output_dim, bias=bias)
        self.norm = nn.BatchNorm1d(output_dim)
        self.dropout = dropout
        self.output_dim = output_dim
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def batch_norm(self, inputs):
        if inputs.numel() == self.output_dim or inputs.numel() == 0:
            return inputs
        if len(inputs.size()) == 3:
            x = inputs.view(inputs.size(0) * inputs.size(1), -1)
            x = self.norm(x)
            return x.view(inputs.size(0), inputs.size(1), -1)
        else:
            return self.norm(inputs)

    def forward(self, inputs):
        x = F.relu(self.fc1(inputs))
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        return self.batch_norm(x)


class EraseAddGate(nn.Module):
    """Erase & Add Gate module"""

    def __init__(self, feature_dim, num_c, bias=True):
        super(EraseAddGate, self).__init__()
        self.weight = nn.Parameter(torch.rand(num_c))
        self.reset_parameters()
        self.erase = nn.Linear(feature_dim, feature_dim, bias=bias)
        self.add = nn.Linear(feature_dim, feature_dim, bias=bias)

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(0))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x):
        erase_gate = torch.sigmoid(self.erase(x))
        tmp_x = x - self.weight.unsqueeze(dim=1) * erase_gate * x
        add_feat = torch.tanh(self.add(x))
        res = tmp_x + self.weight.unsqueeze(dim=1) * add_feat
        return res


@MODEL_REGISTRY.register("gkt")
class GKT(nn.Module):
    """Graph-based Knowledge Tracing Modeling Student Proficiency Using Graph Neural Network

    Args:
        num_c: total num of unique questions/concepts
        hidden_dim: hidden dimension for MLP
        emb_size: embedding dimension
        graph_type: graph type, dense or transition
        dropout: dropout probability
        emb_type: embedding type
    """

    class Inputs(InputSpec):
        """GKT needs a concept-transition graph, which the loaders do not build.

        Cached next to the data; `get_gkt_graph` writes the file on a miss.

        The cache name used to be `gkt_graph_<type>.npz`, which names the graph
        type and nothing else. With `graph_type: transition` -- the configured
        default -- the matrix is counted from the sequence files, so regenerating
        a dataset (changing `keep_scaffolding`, say, or re-running run_clean.py)
        leaves a graph built from the old data sitting in the same path, and it
        is reused without a word. The name now carries a fingerprint of the
        source files, so a stale graph misses instead of lying.
        """

        @classmethod
        def _source_fingerprint(cls, dpath, filenames):
            """Short digest of the files the graph is counted from.

            Path, size and mtime -- enough to notice a regenerated dataset
            without reading gigabytes to hash the contents.
            """
            import hashlib
            import os

            parts = []
            for name in filenames:
                if not name:
                    continue
                path = os.path.join(dpath, name)
                try:
                    stat = os.stat(path)
                    parts.append(f"{name}:{stat.st_size}:{int(stat.st_mtime)}")
                except OSError:
                    parts.append(f"{name}:missing")
            return hashlib.sha256("|".join(parts).encode()).hexdigest()[:12]

        @classmethod
        def prepare(cls, ctx):
            import os

            import numpy as np
            import torch as _torch

            from models.gkt_utils import get_gkt_graph

            graph_type = ctx.model_cfg.get("graph_type", "dense")
            train_file = ctx.dataset_cfg.get(
                "train_valid_original_file", ctx.dataset_cfg.get("train_valid_file")
            )
            test_file = ctx.dataset_cfg.get(
                "test_original_file", ctx.dataset_cfg.get("test_file")
            )
            dpath = ctx.dataset_cfg["dpath"]

            # A dense graph is all ones and reads no data, so it needs no
            # fingerprint; a transition graph is counted from the files.
            if graph_type == "dense":
                graph_file = "gkt_graph_dense.npz"
            else:
                digest = cls._source_fingerprint(dpath, [train_file, test_file])
                graph_file = f"gkt_graph_{graph_type}_{digest}.npz"

            graph_path = os.path.join(dpath, graph_file)
            if os.path.exists(graph_path):
                graph = np.load(graph_path, allow_pickle=True)["matrix"]
            else:
                graph = get_gkt_graph(
                    ctx.dataset_cfg["num_c"],
                    dpath,
                    train_file,
                    test_file,
                    graph_type=graph_type,
                    tofile=graph_file,
                )
            tensor = graph.float() if _torch.is_tensor(graph) else _torch.tensor(graph).float()
            return ModelInputs(
                model_kwargs={"graph": tensor},
                # A dense graph is all ones and reads nothing; a transition graph
                # counts from the train and test sequence files, with no fold
                # filter, so the run is transductive. Recorded in the protocol
                # block rather than changed: pyKT passes both files too, and
                # deviating would cost comparability. No responses are read.
                run_config_extras={
                    "graph_scope": "none" if graph_type == "dense" else "train_valid_test"
                },
            )

    def __init__(self, num_c, hidden_dim, emb_size, graph_type="dense", graph=None, dropout=0.5, emb_type="qid", emb_path="", bias=True, **kwargs):
        super(GKT, self).__init__()
        self.model_name = "gkt"
        self.num_c = num_c
        self.hidden_dim = hidden_dim
        self.emb_size = emb_size
        self.res_len = 2
        self.graph_type = graph_type
        self.emb_type = emb_type
        self.bias = bias

        # Initialize graph
        if graph is not None:
            self.graph = nn.Parameter(graph)
            self.graph.requires_grad = False
        elif graph_type == "dense":
            self.graph = nn.Parameter(torch.ones(num_c, num_c))
            self.graph.requires_grad = False
        else:
            self.graph = nn.Parameter(torch.eye(num_c))
            self.graph.requires_grad = False

        # One-hot features
        one_hot_feat = torch.eye(self.res_len * self.num_c)
        self.register_buffer("one_hot_feat", one_hot_feat)
        one_hot_q = torch.eye(self.num_c)
        zero_padding = torch.zeros(1, self.num_c)
        one_hot_q = torch.cat((one_hot_q, zero_padding), dim=0)
        self.register_buffer("one_hot_q", one_hot_q)

        if emb_type.startswith("qid"):
            self.interaction_emb = nn.Embedding(self.res_len * num_c, emb_size)
            self.emb_c = nn.Embedding(num_c + 1, emb_size, padding_idx=num_c)

        # f_self function
        mlp_input_dim = hidden_dim + emb_size
        self.f_self = MLP(mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias)

        # f_neighbor functions
        self.f_neighbor_list = nn.ModuleList()
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))
        self.f_neighbor_list.append(MLP(2 * mlp_input_dim, hidden_dim, hidden_dim, dropout=dropout, bias=bias))

        # Erase & Add Gate
        self.erase_add_gate = EraseAddGate(hidden_dim, num_c)
        # Gate Recurrent Unit (per-concept update)
        # We keep a state per concept: [B, num_c, hidden]. GRUCell updates each concept independently.
        self.gru = nn.GRUCell(hidden_dim, hidden_dim, bias=bias)
        # prediction layer
        self.predict = nn.Linear(hidden_dim, 1, bias=bias)

    def _aggregate(self, xt, qt, ht, batch_size):
        qt_mask = torch.ne(qt, -1)
        x_idx_mat = torch.arange(self.res_len * self.num_c, device=xt.device)
        x_embedding = self.interaction_emb(x_idx_mat)
        masked_feat = F.embedding(xt[qt_mask], self.one_hot_feat)
        res_embedding = masked_feat.mm(x_embedding)
        mask_num = res_embedding.shape[0]

        concept_idx_mat = self.num_c * torch.ones((batch_size, self.num_c), device=qt.device).long()
        concept_idx_mat[qt_mask, :] = torch.arange(self.num_c, device=qt.device)
        concept_embedding = self.emb_c(concept_idx_mat)

        index_tuple = (torch.arange(mask_num, device=qt.device), qt[qt_mask].long())
        concept_embedding[qt_mask] = concept_embedding[qt_mask].index_put(index_tuple, res_embedding)
        tmp_ht = torch.cat((ht, concept_embedding), dim=-1)
        return tmp_ht

    def _agg_neighbors(self, tmp_ht, qt):
        qt_mask = torch.ne(qt, -1)
        masked_qt = qt[qt_mask]
        masked_tmp_ht = tmp_ht[qt_mask]
        mask_num = masked_tmp_ht.shape[0]

        self_index_tuple = (torch.arange(mask_num, device=qt.device), masked_qt.long())
        self_ht = masked_tmp_ht[self_index_tuple]
        self_features = self.f_self(self_ht)
        expanded_self_ht = self_ht.unsqueeze(dim=1).repeat(1, self.num_c, 1)
        neigh_ht = torch.cat((expanded_self_ht, masked_tmp_ht), dim=-1)

        adj = self.graph[masked_qt.long(), :].unsqueeze(dim=-1)
        reverse_adj = self.graph[:, masked_qt.long()].transpose(0, 1).unsqueeze(dim=-1)
        neigh_features = adj * self.f_neighbor_list[0](neigh_ht) + reverse_adj * self.f_neighbor_list[1](neigh_ht)

        m_next = tmp_ht[:, :, :self.hidden_dim]
        m_next[qt_mask] = neigh_features
        m_next[qt_mask] = m_next[qt_mask].index_put(self_index_tuple, self_features)
        return m_next

    def _update(self, tmp_ht, ht, qt):
        qt_mask = torch.ne(qt, -1)
        mask_num = qt_mask.nonzero().shape[0]
        m_next = self._agg_neighbors(tmp_ht, qt)
        m_next[qt_mask] = self.erase_add_gate(m_next[qt_mask])

        h_next = m_next
        if mask_num > 0:
            x = m_next[qt_mask].reshape(mask_num * self.num_c, self.hidden_dim)
            h0 = ht[qt_mask].reshape(mask_num * self.num_c, self.hidden_dim)
            res = self.gru(x, h0).reshape(mask_num, self.num_c, self.hidden_dim)
            h_next[qt_mask] = res
        return h_next

    def _predict(self, h_next, qt):
        qt_mask = torch.ne(qt, -1)
        y = self.predict(h_next).squeeze(dim=-1)
        y[qt_mask] = torch.sigmoid(y[qt_mask])
        return y

    def _get_next_pred(self, yt, q_next):
        next_qt = torch.where(q_next != -1, q_next, self.num_c * torch.ones_like(q_next, device=yt.device))
        one_hot_qt = F.embedding(next_qt.long(), self.one_hot_q)
        pred = (yt * one_hot_qt).sum(dim=1)
        return pred

    def forward(self, q, r, **kwargs):
        """Forward pass

        Args:
            q: question indices [batch_size, seq_len]
            r: response values [batch_size, seq_len]

        Returns:
            pred_res: predicted probabilities [batch_size, seq_len-1]
        """
        features = q * 2 + r
        questions = q

        batch_size, seq_len = features.shape
        ht = Variable(torch.zeros((batch_size, self.num_c, self.hidden_dim), device=q.device))

        pred_list = []
        for i in range(seq_len):
            xt = features[:, i]
            qt = questions[:, i]
            qt_mask = torch.ne(qt, -1)
            tmp_ht = self._aggregate(xt, qt, ht, batch_size)
            h_next = self._update(tmp_ht, ht, qt)
            ht[qt_mask] = h_next[qt_mask]
            yt = self._predict(h_next, qt)
            if i < seq_len - 1:
                pred = self._get_next_pred(yt, questions[:, i + 1])
                pred_list.append(pred)
        pred_res = torch.stack(pred_list, dim=1)
        return pred_res
