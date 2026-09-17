# coding: utf-8
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY


def generate_qmatrix(dpath, num_q, num_c, gamma=0.0):
    """Generate Q-matrix from raw data.

    Args:
        dpath: data path
        num_q: number of questions
        num_c: number of concepts
        gamma: default value for Q-matrix

    Returns:
        q_matrix: numpy array of shape (num_q+1, num_c+1)
    """
    qmatrix_path = os.path.join(dpath, "qmatrix.npz")

    if os.path.exists(qmatrix_path):
        q_matrix = np.load(qmatrix_path)['matrix']
        return q_matrix

    # Try to generate from train_valid.csv
    train_file = os.path.join(dpath, "train_valid.csv")
    test_file = os.path.join(dpath, "test.csv")

    if not os.path.exists(train_file):
        print(f"Warning: {train_file} not found, using empty Q-matrix")
        return np.zeros((num_q + 1, num_c + 1)) + gamma

    try:
        df_train = pd.read_csv(train_file)
        df_test = pd.read_csv(test_file) if os.path.exists(test_file) else pd.DataFrame()
        df = pd.concat([df_train, df_test])

        problem2skill = {}
        for _, row in df.iterrows():
            if "concepts" not in row or "questions" not in row:
                continue
            cids = [int(_) for _ in row["concepts"].split(",")]
            qids = [int(_) for _ in row["questions"].split(",")]
            for q, c in zip(qids, cids):
                if q not in problem2skill:
                    problem2skill[q] = []
                if c not in problem2skill[q]:
                    problem2skill[q].append(c)

        q_matrix = np.zeros((num_q + 1, num_c + 1)) + gamma
        for p, cs in problem2skill.items():
            if p < num_q + 1:
                for c in cs:
                    if c < num_c + 1:
                        q_matrix[p][c] = 1

        np.savez(qmatrix_path, matrix=q_matrix)
        return q_matrix
    except Exception as e:
        print(f"Warning: Failed to generate qmatrix: {e}")
        return np.zeros((num_q + 1, num_c + 1)) + gamma


@MODEL_REGISTRY.register("lpkt")
class LPKT(nn.Module):
    class Inputs(InputSpec):
        """Builds the answer-time and interval-time vocabularies.

        LPKT bins a raw duration into an index and embeds it, so `num_at` and
        `num_it` are table heights. `always_train_folds` is the difference
        between the two models: HDKT always fits on the current fold's training
        rows, while LPKT does so only under all_in_one -- `folds=None` in
        one_by_one reads every split, which is the historical pyKT behaviour and
        is recorded as such rather than silently changed.
        """

        dataset_mode = "all_in_one"
        requires_question_ids = True
        always_train_folds = False

        @classmethod
        def prepare(cls, ctx):
            from datasets.lpkt_utils import generate_time2idx

            fold_scoped = cls.always_train_folds or ctx.dataset_mode == "all_in_one"
            folds = ctx.train_folds() if fold_scoped else None
            at2idx, it2idx = generate_time2idx(ctx.dataset_cfg, folds=folds)

            sizes = {"num_at": len(at2idx) + 1, "num_it": len(it2idx) + 1}
            dataset_updates = dict(sizes)
            if fold_scoped:
                dataset_updates["time_index_scope"] = "train_folds_only"
                dataset_updates["time_index_folds"] = folds

            inputs = ModelInputs(
                model_cfg_updates=dict(sizes),
                dataset_cfg_updates=dataset_updates,
                dataset_kwargs={"time_idx_maps": {"at2idx": at2idx, "it2idx": it2idx}},
                feature_fit_scope="train_folds" if fold_scoped else "train_valid_test",
            )
            cls.extend(inputs, ctx)
            return inputs

        @classmethod
        def extend(cls, inputs, ctx):
            """Hook for the subclass-specific part. Nothing shared to add."""

        @classmethod
        def extend(cls, inputs, ctx):
            # Decides whether the initial knowledge state is a learned parameter
            # or re-drawn per forward; see the note at the xavier_uniform_ call.
            inputs.model_kwargs["use_runtime_concepts"] = (
                ctx.dataset_mode == "all_in_one"
            )

    """Linear Pedagogical Knowledge Tracing.

    Args:
        num_at: number of unique timestamps
        num_it: number of unique interaction timestamps
        num_q: number of questions/exercises
        num_c: number of concepts
        d_a: answer embedding dimension
        d_e: exercise embedding dimension
        d_k: knowledge embedding dimension
        gamma: q-matrix default value
        dropout: dropout probability
        q_matrix: question-concept matrix
        emb_type: embedding type
        use_time: whether to use time information
    """

    def __init__(
        self,
        num_at=100,
        num_it=100,
        num_q=100,
        num_c=10,
        d_a=64,
        d_e=64,
        d_k=64,
        gamma=0.03,
        dropout=0.2,
        q_matrix=None,
        emb_type="qid",
        use_time=True,
        dpath="",
        use_runtime_concepts=False,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "lpkt"
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.n_question = num_q
        self.num_c = int(num_c)
        self.gamma = float(gamma)
        self.use_runtime_concepts = bool(use_runtime_concepts)

        # Q-matrix: load from file or generate
        if q_matrix is not None:
            matrix = torch.tensor(q_matrix, dtype=torch.float) if not torch.is_tensor(q_matrix) else q_matrix.float()
        elif self.use_runtime_concepts:
            # all_in_one supplies the complete KC set per question at runtime,
            # so no validation/test-derived Q-matrix is needed.
            matrix = torch.zeros(num_q + 1, num_c + 1, dtype=torch.float)
        else:
            matrix = torch.tensor(
                generate_qmatrix(dpath, num_q, num_c, gamma=0.0),
                dtype=torch.float,
            )
        matrix[matrix == 0] = gamma
        self.register_buffer("q_matrix", matrix)

        self.emb_type = emb_type
        self.use_time = use_time
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Embeddings
        self.at_embed = nn.Embedding(num_at + 10, d_k)
        self.it_embed = nn.Embedding(num_it + 10, d_k)
        self.e_embed = nn.Embedding(num_q + 10, d_e)

        # Linear layers
        self.linear_0 = nn.Linear(d_a + d_e, d_k)
        self.linear_1 = nn.Linear(d_a + d_e + d_k, d_k)
        self.linear_2 = nn.Linear(4 * d_k, d_k)
        self.linear_3 = nn.Linear(4 * d_k, d_k)
        self.linear_4 = nn.Linear(3 * d_k, d_k)
        self.linear_5 = nn.Linear(d_e + d_k, d_k)
        self.linear_6 = nn.Linear(3 * d_k, d_k)
        self.linear_7 = nn.Linear(3 * d_k, d_k)
        self.linear_8 = nn.Linear(2 * d_k, d_k)

        self.tanh = nn.Tanh()
        self.sig = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        if self.use_runtime_concepts:
            self.initial_knowledge = nn.Parameter(torch.empty(num_c + 1, d_k))
        else:
            self.register_parameter("initial_knowledge", None)

        # Initialize weights
        self._init_weights()
        if self.initial_knowledge is not None:
            nn.init.xavier_uniform_(self.initial_knowledge)

    def _init_weights(self):
        for m in [self.at_embed, self.it_embed, self.e_embed,
                  self.linear_0, self.linear_1, self.linear_2, self.linear_3,
                  self.linear_4, self.linear_5, self.linear_6, self.linear_7, self.linear_8]:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def _runtime_concept_weights(self, concept_data, valid_mask=None):
        if concept_data is None:
            raise ValueError(
                "LPKT all_in_one mode requires runtime concept sequences."
            )
        if concept_data.dim() == 2:
            concept_data = concept_data.unsqueeze(-1)
        if concept_data.dim() != 3:
            raise ValueError("LPKT concepts must have shape [B,T] or [B,T,K].")
        concept_slots = (concept_data >= 0) & (concept_data < self.num_c)
        if torch.any(concept_data >= self.num_c):
            raise ValueError("LPKT encountered an out-of-range concept id.")
        safe = concept_data.clamp(min=0, max=self.num_c - 1)
        weights = torch.full(
            (*concept_data.shape[:2], self.num_c + 1),
            self.gamma,
            dtype=self.q_matrix.dtype,
            device=concept_data.device,
        )
        one_hot = torch.nn.functional.one_hot(
            safe, num_classes=self.num_c + 1
        ).to(weights.dtype)
        present = (one_hot * concept_slots.unsqueeze(-1)).amax(dim=2)
        weights = torch.where(present.bool(), torch.ones_like(weights), weights)
        if valid_mask is not None:
            weights = weights * valid_mask.unsqueeze(-1).to(weights.dtype)
        return weights

    def forward(
        self,
        e_data,
        a_data,
        it_data=None,
        at_data=None,
        concept_data=None,
        valid_mask=None,
        qtest=False,
    ):
        """Forward pass.

        Args:
            e_data: exercise/question sequences [batch_size, seq_len]
            a_data: answer sequences [batch_size, seq_len]
            it_data: interaction time sequences [batch_size, seq_len]
            at_data: time since start sequences [batch_size, seq_len]
            qtest: whether to return hidden states

        Returns:
            predictions: predicted probabilities [batch_size, seq_len]
        """
        emb_type = self.emb_type
        batch_size, seq_len = e_data.size(0), e_data.size(1)

        # Exercise embeddings
        e_embed_data = self.e_embed(e_data)

        # Time embeddings
        at_embed_data = None
        it_embed_data = None
        if self.use_time and it_data is None:
            raise ValueError("LPKT requires it_data when use_time=True.")
        if self.use_time and at_data is not None:
            at_embed_data = self.at_embed(at_data)
        if self.use_time and it_data is not None:
            it_embed_data = self.it_embed(it_data)

        # Answer embeddings
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        a_data = a_data.to(e_embed_data.dtype)

        q_matrix = self.q_matrix
        n_skills = q_matrix.size(1)
        runtime_weights = None
        if self.use_runtime_concepts:
            runtime_weights = self._runtime_concept_weights(
                concept_data, valid_mask=valid_mask
            )

        # Initialize knowledge state
        if self.initial_knowledge is None:
            # Historical pyKT-compatible one_by_one behavior.
            h_pre = nn.init.xavier_uniform_(
                torch.zeros(n_skills, self.d_k, device=e_data.device)
            ).repeat(batch_size, 1, 1)
        else:
            h_pre = self.initial_knowledge.unsqueeze(0).expand(
                batch_size, -1, -1
            )
        h_tilde_pre = None

        # Learning computation
        if emb_type == "qid":
            if self.use_time and at_data is not None:
                all_learning = self.linear_1(
                    torch.cat((e_embed_data, at_embed_data, a_data), 2)
                )
            else:
                all_learning = self.linear_0(
                    torch.cat((e_embed_data, a_data), 2)
                )

        learning_pre = torch.zeros(batch_size, self.d_k).to(e_data.device)
        pred = torch.zeros(batch_size, seq_len).to(e_data.device)
        hidden_state = torch.zeros(batch_size, seq_len, self.d_k).to(e_data.device)

        for t in range(0, seq_len - 1):
            e = e_data[:, t]
            q_e = (
                runtime_weights[:, t].unsqueeze(1)
                if runtime_weights is not None
                else q_matrix[e].view(batch_size, 1, -1)
            )

            if self.use_time:
                it = it_embed_data[:, t]
                # Learning Module
                if h_tilde_pre is None:
                    c_pre = torch.unsqueeze(torch.sum(torch.squeeze(q_e, dim=1), 1), -1)
                    h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k) / c_pre

                learning = all_learning[:, t]
                learning_gain = self.linear_2(
                    torch.cat((learning_pre, it, learning, h_tilde_pre), 1)
                )
                learning_gain = self.tanh(learning_gain)
                gamma_l = self.linear_3(
                    torch.cat((learning_pre, it, learning, h_tilde_pre), 1)
                )
            else:
                # Learning Module without time
                if h_tilde_pre is None:
                    c_pre = torch.unsqueeze(torch.sum(torch.squeeze(q_e, dim=1), 1), -1)
                    h_tilde_pre = q_e.bmm(h_pre).view(batch_size, self.d_k) / c_pre

                learning = all_learning[:, t]
                learning_gain = self.linear_6(
                    torch.cat((learning_pre, learning, h_tilde_pre), 1)
                )
                learning_gain = self.tanh(learning_gain)
                gamma_l = self.linear_7(
                    torch.cat((learning_pre, learning, h_tilde_pre), 1)
                )

            gamma_l = self.sig(gamma_l)
            LG = gamma_l * ((learning_gain + 1) / 2)
            LG_tilde = self.dropout(q_e.transpose(1, 2).bmm(LG.view(batch_size, 1, -1)))

            # Forgetting Module
            n_skill = LG_tilde.size(1)
            if self.use_time:
                gamma_f = self.sig(self.linear_4(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill).view(batch_size, -1, self.d_k),
                    it.repeat(1, n_skill).view(batch_size, -1, self.d_k)
                ), 2)))
            else:
                gamma_f = self.sig(self.linear_8(torch.cat((
                    h_pre,
                    LG.repeat(1, n_skill).view(batch_size, -1, self.d_k)
                ), 2)))

            h = LG_tilde + gamma_f * h_pre

            # Predicting Module
            e_next = e_data[:, t + 1]
            next_weights = (
                runtime_weights[:, t + 1].unsqueeze(1)
                if runtime_weights is not None
                else q_matrix[e_next].view(batch_size, 1, -1)
            )
            c_tilde = torch.unsqueeze(
                torch.sum(torch.squeeze(next_weights, dim=1), 1),
                -1
            )
            h_tilde = next_weights.bmm(h).view(batch_size, self.d_k) / c_tilde.clamp_min(1e-12)

            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1))).sum(1) / self.d_k
            pred[:, t + 1] = y
            hidden_state[:, t + 1, :] = h_tilde

            # Prepare for next prediction
            learning_pre = learning
            h_pre = h
            h_tilde_pre = h_tilde

        if not qtest:
            return pred
        else:
            return pred, hidden_state[:, :-1, :], e_embed_data
