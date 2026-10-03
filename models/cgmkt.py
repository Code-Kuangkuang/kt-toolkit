"""CGMKT port for KT-Toolkit.

    Cognition-Driven Dual-Graph Fusion with Group-Level Mastery for Knowledge
    Tracing. Bai, Huang, Zheng, Hou, Guo, Liu. Information Fusion 104800, 2026.
    doi:10.1016/j.inffus.2026.104800

Source: pykt-team/pykt-toolkit, `pykt/models/cgmkt.py` plus `QueGraphLearing.py`
and `SkillGraphLearning.py` (fetched 2026-09-22 from `main` @ 77c3e90, merged
there from `youh_dev` as PR #305).

Correctness repair (2026-10-03): sbm_fit explicitly filters the current fold's
training rows; excluding the test file alone did not exclude validation rows.

The model: two GCN branches (a question branch and a KC branch) feed one GRU,
and a per-student `[k]` mastery vector over SBM knowledge groups gates the
interaction before encoding (Eq. 8, "differentiation") and spreads across groups
after the response is seen (Eq. 7/11, "assimilation"). Causality is sound in
upstream and is preserved here: the gate at step t reads `m_{t-1}`, the mastery
update reads `r_t` only after the GRU step, and `p_{t+1}` is formed from `h_t`,
`m_t` and the next question's identity alone.

Changes relative to upstream, each with its reason:

1. Artefacts are injected, not loaded. Upstream's `__init__` reads four files
   from a hard-coded `../data/<dataset>` resolved from `emb_type`, which only
   works when the process runs from `examples/`. `Inputs.prepare` supplies them
   as constructor arguments here, per core/model_inputs.py.

2. The paper/code disagreements are selectable rather than silent. See
   models/cgmkt_graphs.py for the evidence; `question_graph_source`,
   `kc_graph_source` and `group_source` choose a variant and land in
   run_config.json. Defaults reproduce the released code, so an unqualified
   `cgmkt` row is comparable to the published numbers.

3. No silent fallback. Upstream tries sixteen filenames for the membership
   matrix and falls through to spectral clustering when all miss, recording the
   miss only in a local string. Since the released artefacts contain no
   membership file at all, every upstream run takes that fallback without
   saying so. Here the source is named up front.

4. Graphs, memberships and the transition matrix are buffers. They were plain
   attributes carrying tensors placed on a module-level
   `device = cuda if available`, so `model.to(device)` could not move them and a
   CPU run touched the GPU. Same defect and same fix as models/mtkt.py.

5. Dead parameters removed. `Concepts_Embedding.bert_projection`,
   `Questions_Embedding.change` and the local `kc_id_embed` are applied once at
   construction and never appear in `forward`; upstream leaves them registered,
   so they collect zero gradients and inflate the parameter count. They are
   local here, which leaves the computed embedding tables bit-identical -- the
   xavier loop that re-initialised them ran after they had already been used --
   while dropping the two registered `Linear(1024, emb_size)` layers, 524,800
   parameters at emb_size=256. `Questions_Embedding.pro_id_embed` is built and
   never read at all; upstream leaves it local, so it costs nothing there, but
   it is gone here rather than left as a puzzle for the next reader.

6. `torch.ones(...)` multiplies dropped. `self.kc_embed * bert_tensor` and
   `pro_embedding_kc * self.pro_embed` multiply by a freshly-created tensor of
   ones, so both are identities. Written out directly.

7. `emb_type` no longer selects the dataset. Upstream parses tokens like
   "qid_as09" out of it to find the data directory, so the field carried two
   meanings at once; the dataset comes from `RunContext` here and `emb_type`
   stays the plain "qid" the rest of the registry uses.

Tensor flow (all_in_one, multi-concept):

    qseqs / shft_qseqs     [B, T]        question ids, -1 padded
    cseqs / shft_cseqs     [B, T, K]     concept ids, -1 padded
    rseqs                  [B, T]        float 0/1
    x_t                    [B, T, d]     question branch + KC branch + response
    q_cur / q_next         [B, T, k]     group membership of this/next question
    m_t                    [B, k]        mastery, clipped to +-mastery_bound
    y                      [B, T]        aligned with shft_rseqs and smasks
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from core.model_inputs import InputSpec
from core.registry import MODEL_REGISTRY

# No module-level CUDA default; see models/mockt.py for why.


class GCNConv(nn.Module):
    """Upstream's GCN layer: dropout, dense projection, sparse propagate, bias."""

    def __init__(self, in_dim, out_dim, p):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w = nn.Parameter(torch.rand((in_dim, out_dim)))
        nn.init.xavier_uniform_(self.w)
        self.b = nn.Parameter(torch.zeros(out_dim))
        self.dropout = nn.Dropout(p=p)

    def forward(self, x, adj):
        x = self.dropout(x)
        x = torch.matmul(x, self.w)
        x = torch.sparse.mm(adj.float(), x) if adj.is_sparse else torch.matmul(adj.float(), x)
        return x + self.b


class GCN_Graph_Learning(nn.Module):
    """`num_gcn_layers` of GCNConv with ReLU between and one residual at the end."""

    def __init__(self, d, p, num_gcn_layers=1):
        super().__init__()
        self.gcn_layers = nn.ModuleList([GCNConv(d, d, p) for _ in range(num_gcn_layers)])

    def forward(self, x, adj):
        embed = x
        for layer in self.gcn_layers:
            embed = F.relu(layer(embed, adj))
        return x + embed


def get_kc_embedding(concept_ids, concept_embedding, padding_idx=-1):
    """Masked mean of the concept rows of each `[B, T]` position.

    Equivalent to `multi_concept.pool_concept_embeddings`; upstream's own version
    is kept so a future diff against pykt stays readable. `-1` padding is clamped
    to 0 before the lookup and then zeroed by the mask, so no padded slot
    contributes and no index goes out of range.
    """
    mask = concept_ids != padding_idx                       # [B, T, K]
    safe = concept_ids.clamp(min=0)
    emb = F.embedding(safe, concept_embedding)              # [B, T, K, d]
    emb = emb * mask.unsqueeze(-1).to(emb.dtype)
    counts = mask.sum(dim=2, keepdim=True).clamp(min=1).to(emb.dtype)
    return emb.sum(dim=2) / counts                          # [B, T, d]


class Questions_Embedding(nn.Module):
    """Question branch: a GCN over the question graph, plus a response embedding.

    The question table is initialised from the mean BGE vector of each question's
    concepts, projected once. Two questions with the same concept set therefore
    start identical and only diverge through training -- true upstream as well,
    since the question-id embedding it builds for this purpose is never read.
    """

    def __init__(self, d, p, num_gcn_layers, concept_map, concept_embedding,
                 tie_to_concepts=True, num_q=None):
        super().__init__()
        self.d_model = d
        self.tie_to_concepts = bool(tie_to_concepts)
        self.ans_embed = nn.Embedding(2, d)
        nn.init.xavier_uniform_(self.ans_embed.weight)

        if not self.tie_to_concepts:
            # The paper's own "w/o question-question graph" arm: a free table per
            # question, no concept tying at init and no graph hop at run time.
            # This is the control the -0.0259 in Table 4 is measured against, so
            # it has to exist here for that number to be checkable on this
            # repository's protocol rather than taken on trust.
            self.gcl = None
            self.pro_embed = nn.Parameter(torch.empty(int(num_q), d))
            nn.init.xavier_uniform_(self.pro_embed)
            return

        self.gcl = GCN_Graph_Learning(d, p, num_gcn_layers)
        # Applied once, so it is local: registering it would add parameters that
        # never receive gradient. See change 5 in the module docstring.
        change = nn.Linear(concept_embedding.shape[1], d)
        with torch.no_grad():
            pooled = get_kc_embedding(
                concept_map.unsqueeze(0).long(), concept_embedding.float()
            ).squeeze(0)                                    # [num_q, emb_dim]
            self.pro_embed = nn.Parameter(change(pooled))   # [num_q, d]

    def forward(self, last_pro, last_ans, next_pro, matrix):
        pro_embed = self.gcl(self.pro_embed, matrix) if self.gcl is not None else self.pro_embed
        last = F.embedding(last_pro, pro_embed)             # [B, T, d]
        nxt = F.embedding(next_pro, pro_embed)
        return last + self.ans_embed(last_ans), nxt


class Concepts_Embedding(nn.Module):
    """KC branch: a GCN over the KC graph, plus the same response embedding."""

    def __init__(self, d, p, num_gcn_layers, concept_embedding):
        super().__init__()
        self.d_model = d
        self.gcl = GCN_Graph_Learning(d, p, num_gcn_layers)
        self.ans_embed = nn.Embedding(2, d)
        nn.init.xavier_uniform_(self.ans_embed.weight)

        num_c, emb_dim = concept_embedding.shape
        bert_projection = nn.Linear(emb_dim, d)             # local, see change 5
        kc_id_embed = nn.Embedding(num_c, emb_dim)          # local, upstream too
        with torch.no_grad():
            self.kc_embed = nn.Parameter(
                bert_projection(concept_embedding.float() + kc_id_embed.weight)
            )                                               # [num_c, d]

    def forward(self, last_ans, last_skill, next_skill, matrix):
        kc_embed = self.gcl(self.kc_embed, matrix)          # [num_c, d]
        last = get_kc_embedding(last_skill, kc_embed)       # [B, T, d]
        nxt = get_kc_embedding(next_skill, kc_embed)
        return last + self.ans_embed(last_ans), nxt


class CGMKT(nn.Module):
    """Dual-graph fusion with a group-level mastery state.

    `question_graph`, `kc_graph`, `membership` and `group_transition` are
    supplied by `Inputs.prepare`; which variant of each was built is recorded in
    `run_config.json` rather than being implied by the model name.
    """

    def __init__(
        self,
        num_c,
        num_q,
        emb_size=256,
        dropout=0.1,
        emb_type="qid",
        dropout_qk=0.2,
        dropout_kk=0.2,
        num_clusters=9,
        num_gcn_layers=1,
        mastery_update_hidden=64,
        mastery_step=0.1,
        modulation_type="gate",
        spread_type="sbm",
        spread_rate_init=0.1,
        spread_rate_max=1.0,
        mastery_bound=5.0,
        use_mastery=True,
        question_graph_source="incidence",
        question_graph=None,
        kc_graph=None,
        membership=None,
        group_transition=None,
        concept_map=None,
        concept_embedding=None,
        **kwargs,
    ):
        super().__init__()
        for key in ("device", "dpath", "num_at", "num_it", "seed", "num_pid",
                    "final_fc_dim2", "kc_graph_source",
                    "group_source", "graph_seed", "kc_embedding_source"):
            kwargs.pop(key, None)

        missing = [
            name
            for name, value in (
                ("question_graph", question_graph),
                ("kc_graph", kc_graph),
                ("membership", membership),
                ("group_transition", group_transition),
                ("concept_map", concept_map),
                ("concept_embedding", concept_embedding),
            )
            if value is None
        ]
        if missing:
            raise ValueError(
                f"CGMKT needs {missing}; CGMKT.Inputs.prepare builds them, so "
                f"constructing the model directly means passing them."
            )

        self.model_name = "cgmkt"
        self.num_c = num_c
        self.num_q = num_q
        self.emb_size = emb_size
        self.emb_type = emb_type
        self.dropout = dropout
        self.num_clusters = int(num_clusters)
        self.mastery_step = mastery_step
        self.mastery_bound = mastery_bound
        self.modulation_type = modulation_type
        self.spread_type = spread_type
        self.spread_rate_max = spread_rate_max
        # The paper's "w/o mastery state" arm: modulation, propagation, updating
        # and readout all bypassed, leaving the graph-fused sequence encoded by
        # the GRU alone. Needed as a switch because with the question branch
        # worth only -0.0014 here, the mastery module and the KC branch are the
        # only places the rest of the margin can be.
        self.use_mastery = bool(use_mastery)

        membership = torch.as_tensor(membership, dtype=torch.float32)
        if membership.shape != (num_c, self.num_clusters):
            raise ValueError(
                f"CGMKT membership must be [num_c={num_c}, k={self.num_clusters}], "
                f"got {tuple(membership.shape)}."
            )
        # Buffers, so model.to(device) moves them (change 4).
        self.register_buffer("Q_hat", membership)
        self.register_buffer(
            "group_transition",
            torch.as_tensor(group_transition, dtype=torch.float32),
        )
        self.register_buffer("question_graph", question_graph.coalesce()
                             if question_graph.is_sparse else question_graph.float())
        self.register_buffer("kc_graph", kc_graph.coalesce()
                             if kc_graph.is_sparse else kc_graph.float())

        if spread_rate_max <= 0:
            raise ValueError("CGMKT's spread_rate_max must be positive.")
        ratio = min(max(spread_rate_init / spread_rate_max, 1e-6), 1.0 - 1e-6)
        self.raw_spread_rate = nn.Parameter(torch.logit(torch.tensor(float(ratio))))

        d = emb_size
        self.qe = Questions_Embedding(
            d, dropout_qk, num_gcn_layers,
            torch.as_tensor(concept_map), torch.as_tensor(concept_embedding),
            tie_to_concepts=(question_graph_source != "none"), num_q=num_q,
        )
        self.ce = Concepts_Embedding(
            d, dropout_kk, num_gcn_layers, torch.as_tensor(concept_embedding)
        )

        if modulation_type == "gate":
            self.modulator = nn.Sequential(nn.Linear(self.num_clusters + d, d), nn.Sigmoid())
        elif modulation_type == "film":
            self.modulator_gamma = nn.Linear(self.num_clusters, d)
            self.modulator_beta = nn.Linear(self.num_clusters, d)
        elif modulation_type in ("scalar", "none"):
            self.modulator = None
        else:
            raise ValueError(f"Unknown CGMKT modulation_type: {modulation_type!r}.")

        self.gru = nn.GRU(d, d, batch_first=True)
        self.update_mlp = nn.Sequential(
            nn.Linear(d + 1 + self.num_clusters, mastery_update_hidden),
            nn.ReLU(),
            nn.Linear(mastery_update_hidden, self.num_clusters),
        )
        self.out_layer = nn.Sequential(
            nn.Linear(2 * d + (self.num_clusters if self.use_mastery else 0), d),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(d, 1),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        def init_linear(module):
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=nn.init.calculate_gain("relu"))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        # Zeroed so the gate starts at sigmoid(0)=0.5 and the factor of 2 in
        # `_modulate` leaves the signal magnitude untouched at step 0 (Eq. 8).
        if self.modulation_type == "gate":
            nn.init.zeros_(self.modulator[0].weight)
            nn.init.zeros_(self.modulator[0].bias)
        elif self.modulation_type == "film":
            for layer in (self.modulator_gamma, self.modulator_beta):
                nn.init.zeros_(layer.weight)
                nn.init.zeros_(layer.bias)
        self.update_mlp.apply(init_linear)
        self.out_layer.apply(init_linear)

    def _concept_group_membership(self, concept_ids):
        """`[B, T, k]`: mean SBM membership over a position's concepts (Eq. 6)."""
        if concept_ids.dim() == 2:
            concept_ids = concept_ids.unsqueeze(-1)
        valid = concept_ids >= 0
        memberships = F.embedding(concept_ids.clamp(min=0), self.Q_hat)
        memberships = memberships * valid.unsqueeze(-1).to(memberships.dtype)
        counts = valid.sum(dim=2, keepdim=True).clamp(min=1).to(memberships.dtype)
        return memberships.sum(dim=2) / counts

    def _modulate(self, x_t, m_prev, q_cur):
        """Eq. 8. Reads `m_prev`, never `m_t`: the gate cannot see r_t."""
        if self.modulation_type == "none":
            return x_t
        m_q = m_prev * q_cur
        if self.modulation_type == "gate":
            return x_t * (2.0 * self.modulator(torch.cat([m_q, x_t], dim=-1)))
        if self.modulation_type == "film":
            return (1.0 + torch.tanh(self.modulator_gamma(m_q))) * x_t + self.modulator_beta(m_q)
        return (1.0 + torch.tanh(m_q.sum(dim=-1, keepdim=True))) * x_t  # scalar

    def _spread_mask(self, q_cur):
        """Eq. 7. `spread_type="hard"` is the no-cross-group-propagation ablation."""
        if self.spread_type == "hard":
            return q_cur
        beta = self.spread_rate_max * torch.sigmoid(self.raw_spread_rate)
        spread = q_cur + beta * torch.matmul(q_cur, self.group_transition)
        return spread / (spread.sum(dim=-1, keepdim=True) + 1e-8)

    def _update_mastery(self, m_prev, h_t, r_t, spread):
        """Eq. 10-11, applied only after the GRU has consumed step t."""
        context = m_prev * spread
        delta = torch.tanh(
            self.update_mlp(torch.cat([h_t, r_t.unsqueeze(-1).float(), context], dim=-1))
        )
        updated = m_prev + self.mastery_step * spread * delta
        return updated.clamp(-self.mastery_bound, self.mastery_bound)

    def forward(self, last_pro, last_ans, last_skill, next_pro, next_skill):
        x_pro, next_pro_emb = self.qe(last_pro, last_ans.long(), next_pro, self.question_graph)
        x_kc, next_kc_emb = self.ce(
            last_ans.long(), last_skill.long(), next_skill.long(), self.kc_graph
        )
        x_t = x_pro + x_kc                                  # [B, T, d]
        next_x = next_pro_emb + next_kc_emb

        q_cur = self._concept_group_membership(last_skill.long())    # [B, T, k]
        q_next = self._concept_group_membership(next_skill.long())
        valid = (last_skill >= 0).any(dim=-1) if last_skill.dim() == 3 else last_skill >= 0

        batch, seq_len, dim = x_t.shape
        if not self.use_mastery:
            # One GRU pass over the unmodulated sequence; no per-step Python
            # loop, and nothing reads r_t beyond the response embedding already
            # inside x_t.
            h_t, _ = self.gru(x_t)
            logits = self.out_layer(torch.cat([h_t, next_x], dim=-1)).squeeze(-1)
            return torch.sigmoid(logits), logits.new_zeros(())

        m_prev = x_t.new_zeros(batch, self.num_clusters)
        h_gru = x_t.new_zeros(1, batch, dim)
        h_seq, m_seq = [], []

        for t in range(seq_len):
            spread = self._spread_mask(q_cur[:, t, :])
            x_mod = self._modulate(x_t[:, t, :], m_prev, q_cur[:, t, :])
            h_out, h_gru = self.gru(x_mod.unsqueeze(1), h_gru)
            h_step = h_out.squeeze(1)
            # r_t enters only here, after the GRU step that produced h_step.
            m_new = self._update_mastery(m_prev, h_step, last_ans[:, t], spread)
            m_new = torch.where(valid[:, t].unsqueeze(-1), m_new, m_prev)
            h_seq.append(h_step)
            m_seq.append(m_new)
            m_prev = m_new

        h_t = torch.stack(h_seq, dim=1)                     # [B, T, d]
        readout = torch.sigmoid(torch.stack(m_seq, dim=1)) * q_next
        logits = self.out_layer(torch.cat([h_t, next_x, readout], dim=-1)).squeeze(-1)
        return torch.sigmoid(logits), logits.new_zeros(())


@MODEL_REGISTRY.register("cgmkt")
class CGMKTModel(CGMKT):
    """Registry adapter: builds the four artefacts and names which variant of each."""

    class Inputs(InputSpec):
        """Question ids, two graphs, a membership matrix and concept texts.

        Fit scope depends on `kc_graph_source`: every variant but `"sbm_fit"`
        reads only static metadata (the Q-matrix, the concept texts), which
        covers the test split, while `"sbm_fit"` counts its observed graph from
        the training folds only. `prepare` reports whichever applies so the two
        are never averaged into one baseline row.
        """
        supports_multi_concept = True

        dataset_mode = "all_in_one"
        requires_question_ids = True

        @classmethod
        def prepare(cls, ctx):
            import json
            import os

            from core.model_inputs import ModelInputs
            from models import cgmkt_graphs as cg
            from models.kc_graph_utils import (
                build_question_concept_map,
                build_question_graph,
                load_kc_text_embeddings,
            )

            dpath = ctx.dataset_cfg["dpath"]
            num_q = int(ctx.dataset_cfg["num_q"])
            num_c = int(ctx.dataset_cfg["num_c"])
            num_clusters = int(ctx.model_cfg.get("num_clusters", 9))
            graph_seed = int(ctx.model_cfg.get("graph_seed", 3407))
            q_source = ctx.model_cfg.get("question_graph_source", "incidence")
            kc_source = ctx.model_cfg.get("kc_graph_source", "sbm_random")
            group_source = ctx.model_cfg.get("group_source", "spectral")

            with open(os.path.join(dpath, "keyid2idx.json"), encoding="utf-8") as handle:
                max_concepts = int(json.load(handle)["max_concepts"])
            concept_embedding = load_kc_text_embeddings(ctx.dataset_name, num_c, ctx.root_dir)

            # Separates the two things the question branch introduces at once.
            # Tying every question to its concepts shares statistical strength
            # (assist2009 averages 13 responses per question against 2,820 per
            # concept); the BGE vectors additionally carry which concepts are
            # semantically alike. The paper's ablation removes both together, so
            # it cannot say which one earns the -0.0259.
            #
            # "random" keeps the tying and every graph exactly as they are -- the
            # KC graph is built from the real vectors first -- and swaps only the
            # values the two embedding tables are initialised from. A run that
            # holds up under it was never using the semantics.
            emb_source = ctx.model_cfg.get("kc_embedding_source", "bge")
            init_embedding = concept_embedding

            # ---- question branch
            if q_source == "incidence":
                question_graph = cg.incidence_question_graph(dpath, num_q, num_c)
            elif q_source == "cooccur":
                question_graph = build_question_graph(dpath, num_q)
            elif q_source == "none":
                # Free per-question table, no tying and no hop. The graph still
                # has to be a valid tensor because it is a buffer; it is never
                # read, since Questions_Embedding skips the GCN entirely.
                question_graph = torch.zeros(1, 1)
            else:
                raise ValueError(
                    f"CGMKT question_graph_source must be 'incidence', 'cooccur' "
                    f"or 'none'; got {q_source!r}."
                )

            # ---- KC branch
            fit_scope = "train_valid_test"
            fitted_membership = None
            if kc_source == "sbm_random":
                kc_dense = cg.sbm_random_graph(concept_embedding, num_clusters, graph_seed)
            elif kc_source == "cooccur":
                kc_dense = cg.build_kc_cooccurrence(dpath, num_c)
            elif kc_source == "random":
                reference = cg.sbm_random_graph(concept_embedding, num_clusters, graph_seed)
                kc_dense = cg.random_graph(num_c, float((reference > 0).mean()), graph_seed)
            elif kc_source == "sbm_fit":
                observed = cg.build_kc_transition(
                    num_c, dpath, ctx.resolve_file(
                        ctx.quelevel_key("train_valid_file"), "train_valid_file"
                    ), folds=ctx.train_folds(),
                )
                fitted_membership, _, kc_dense = cg.fit_sbm(
                    observed, num_clusters, graph_seed
                )
                fit_scope = "train_folds"
            else:
                raise ValueError(
                    f"CGMKT kc_graph_source must be one of 'sbm_random', 'sbm_fit', "
                    f"'cooccur', 'random'; got {kc_source!r}."
                )

            # ---- groups
            if group_source == "spectral":
                membership = cg.spectral_membership(kc_dense, num_clusters, graph_seed)
            elif group_source == "random":
                membership = cg.random_membership(num_c, num_clusters, graph_seed)
            elif group_source == "sbm_fit":
                if fitted_membership is None:
                    raise ValueError(
                        "CGMKT group_source='sbm_fit' needs kc_graph_source='sbm_fit'; "
                        "the memberships come out of the same fit."
                    )
                membership = fitted_membership
            else:
                raise ValueError(
                    f"CGMKT group_source must be 'spectral', 'sbm_fit' or 'random'; "
                    f"got {group_source!r}."
                )

            block = cg.block_from_membership(kc_dense, membership)
            transition = cg.group_transition(block)

            if emb_source == "random":
                rng = np.random.default_rng(graph_seed)
                init_embedding = rng.standard_normal(
                    concept_embedding.shape
                ).astype(concept_embedding.dtype)
                # Matched in scale so the downstream Linear sees the same
                # magnitudes and the comparison is about structure, not norms.
                init_embedding *= float(np.linalg.norm(concept_embedding, axis=1).mean())
                init_embedding /= np.linalg.norm(init_embedding, axis=1, keepdims=True)
            elif emb_source != "bge":
                raise ValueError(
                    f"CGMKT kc_embedding_source must be 'bge' or 'random', "
                    f"got {emb_source!r}."
                )

            variants = {
                "question_graph_source": q_source,
                "kc_graph_source": kc_source,
                "group_source": group_source,
                "kc_embedding_source": emb_source,
                "graph_seed": graph_seed,
            }
            inputs = ModelInputs(
                model_kwargs={
                    "question_graph": question_graph,
                    "kc_graph": torch.as_tensor(
                        cg.row_normalise(kc_dense), dtype=torch.float32
                    ),
                    "membership": membership,
                    "group_transition": transition,
                    "concept_map": build_question_concept_map(dpath, num_q, max_concepts),
                    "concept_embedding": init_embedding,
                    "question_graph_source": q_source,
                },
                run_config_extras={
                    **variants, "graph_scope": fit_scope,
                    "model_correctness_revision": "2026-10-03",
                    **({"graph_fit_folds": ctx.train_folds()} if kc_source == "sbm_fit" else {}),
                },
                graph_scope=fit_scope,
            )
            inputs.model_cfg_updates = dict(variants)
            return inputs
