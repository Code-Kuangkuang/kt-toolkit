import math

import torch
from torch import nn
from torch.nn import functional as F

from core.registry import MODEL_REGISTRY


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features, graph):
        support = features @ self.weight
        output = torch.sparse.mm(graph, support) if graph.is_sparse else graph @ support
        return output if self.bias is None else output + self.bias


class HypergraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(in_features, out_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.weight.size(1))
        nn.init.uniform_(self.weight, -bound, bound)
        if self.bias is not None:
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, features, incidence):
        support = features @ self.weight
        if self.bias is not None:
            support = support + self.bias
        if incidence.is_sparse:
            incidence_t = incidence.transpose(0, 1).coalesce()
            edge_features = torch.sparse.mm(incidence_t, support)
            return torch.sparse.mm(incidence, edge_features)
        return incidence @ (incidence.transpose(0, 1) @ support)


class TwoLayerGraphEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.first = GraphConvolution(in_features, hidden_features)
        self.second = GraphConvolution(hidden_features, out_features)
        self.dropout = float(dropout)

    def forward(self, features, graph):
        hidden = F.relu(self.first(features, graph))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return F.relu(self.second(hidden, graph))


class TwoLayerHypergraphEncoder(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.first = HypergraphConvolution(in_features, hidden_features)
        self.second = HypergraphConvolution(hidden_features, out_features)
        self.dropout = float(dropout)

    def forward(self, features, incidence):
        hidden = F.relu(self.first(features, incidence))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return F.relu(self.second(hidden, incidence))


@MODEL_REGISTRY.register("dgekt")
class DGEKT(nn.Module):
    """Dual Graph Ensemble Knowledge Tracing.

    The two response-conditioned question graphs encode concept association
    and directed interaction transitions. Their sequence representations are
    supervised independently and fused into an online ensemble teacher.
    """

    def __init__(
        self,
        num_q,
        num_c,
        emb_size,
        hidden_dim=128,
        num_layers=1,
        dropout=0.0,
        hypergraph=None,
        transition_out=None,
        transition_in=None,
        emb_type="qid",
    ):
        super().__init__()
        self.model_name = "dgekt"
        self.num_q = int(num_q)
        self.num_c = int(num_c)
        self.emb_size = int(emb_size)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.emb_type = emb_type

        if self.num_q <= 0 or self.num_c <= 0:
            raise ValueError("DGEKT requires positive num_q and num_c.")
        if self.emb_size % 2 != 0:
            raise ValueError(f"DGEKT emb_size must be even, got {self.emb_size}.")
        if hypergraph is None or transition_out is None or transition_in is None:
            raise ValueError("DGEKT requires hypergraph, transition_out, and transition_in.")

        expected_node_shape = (2 * self.num_q, 2 * self.num_q)
        expected_hypergraph_shape = (2 * self.num_q, 2 * self.num_c)
        if tuple(hypergraph.shape) != expected_hypergraph_shape:
            raise ValueError(
                "DGEKT hypergraph incidence shape must be "
                f"{expected_hypergraph_shape}, got {tuple(hypergraph.shape)}."
            )
        self.register_buffer(
            "hypergraph", hypergraph.coalesce() if hypergraph.is_sparse else hypergraph
        )
        for name, graph in (
            ("transition_out", transition_out),
            ("transition_in", transition_in),
        ):
            if tuple(graph.shape) != expected_node_shape:
                raise ValueError(
                    f"DGEKT {name} shape must be {expected_node_shape}, got {tuple(graph.shape)}."
                )
            self.register_buffer(name, graph.coalesce() if graph.is_sparse else graph)

        self.interaction_emb = nn.Embedding(2 * self.num_q, self.emb_size)
        self.hyper_encoder = TwoLayerHypergraphEncoder(
            self.emb_size, self.emb_size, self.emb_size, dropout=dropout
        )
        directed_dim = self.emb_size // 2
        self.out_encoder = TwoLayerGraphEncoder(
            self.emb_size, self.emb_size, directed_dim, dropout=dropout
        )
        self.in_encoder = TwoLayerGraphEncoder(
            self.emb_size, self.emb_size, directed_dim, dropout=dropout
        )

        self.hyper_gru = nn.GRU(
            self.emb_size, self.hidden_dim, self.num_layers, batch_first=True
        )
        self.transition_gru = nn.GRU(
            self.emb_size, self.hidden_dim, self.num_layers, batch_first=True
        )
        self.concept_head = nn.Linear(self.hidden_dim, self.num_q)
        self.transition_head = nn.Linear(self.hidden_dim, self.num_q)
        self.ensemble_head = nn.Linear(2 * self.hidden_dim, self.num_q)
        self.hyper_gate = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.transition_gate = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, q, r):
        q = q.long()
        r = r.long()
        if q.numel() and (int(q.min()) < 0 or int(q.max()) >= self.num_q):
            raise ValueError(
                f"DGEKT question ids must be in [0, {self.num_q - 1}], "
                f"got [{int(q.min())}, {int(q.max())}]."
            )
        if r.numel() and (int(r.min()) < 0 or int(r.max()) > 1):
            raise ValueError("DGEKT responses must be binary (0/1).")

        node_features = self.interaction_emb.weight
        hyper_features = self.hyper_encoder(node_features, self.hypergraph)
        transition_out = self.out_encoder(node_features, self.transition_out)
        transition_in = self.in_encoder(node_features, self.transition_in)
        transition_features = torch.cat((transition_in, transition_out), dim=-1)

        interaction_ids = q + self.num_q * (1 - r)
        hyper_inputs = F.embedding(interaction_ids, hyper_features)
        transition_inputs = F.embedding(interaction_ids, transition_features)

        hyper_state, _ = self.hyper_gru(hyper_inputs)
        transition_state, _ = self.transition_gru(transition_inputs)
        concept_logits = self.concept_head(hyper_state)
        transition_logits = self.transition_head(transition_state)

        gate = torch.sigmoid(
            self.hyper_gate(hyper_state) + self.transition_gate(transition_state)
        )
        gated_transition = gate * transition_state
        gated_hyper = (1.0 - gate) * hyper_state
        ensemble_logits = self.ensemble_head(
            torch.cat((gated_transition, gated_hyper), dim=-1)
        )
        return {
            "concept_logits": concept_logits,
            "transition_logits": transition_logits,
            "ensemble_logits": ensemble_logits,
        }
