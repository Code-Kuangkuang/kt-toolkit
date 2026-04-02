# coding: utf-8
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from core.registry import MODEL_REGISTRY


def generate_qmatrix(dpath, num_q, num_c, gamma=0.03):
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
        **kwargs,
    ):
        super().__init__()
        self.model_name = "lpkt"
        self.d_k = d_k
        self.d_a = d_a
        self.d_e = d_e
        self.n_question = num_q

        # Q-matrix: load from file or generate
        if q_matrix is not None:
            self.q_matrix = torch.tensor(q_matrix, dtype=torch.float) if not torch.is_tensor(q_matrix) else q_matrix.float()
            self.q_matrix[self.q_matrix == 0] = gamma
        else:
            from .lpkt import generate_qmatrix
            self.q_matrix = generate_qmatrix(dpath, num_q, num_c, gamma)
            self.q_matrix = torch.tensor(self.q_matrix, dtype=torch.float)

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

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        for m in [self.at_embed, self.it_embed, self.e_embed,
                  self.linear_0, self.linear_1, self.linear_2, self.linear_3,
                  self.linear_4, self.linear_5, self.linear_6, self.linear_7, self.linear_8]:
            if isinstance(m, nn.Embedding):
                nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, e_data, a_data, it_data=None, at_data=None, qtest=False):
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
        if self.use_time and at_data is not None:
            at_embed_data = self.at_embed(at_data)
        if self.use_time and it_data is not None:
            it_embed_data = self.it_embed(it_data)

        # Answer embeddings
        a_data = a_data.view(-1, 1).repeat(1, self.d_a).view(batch_size, -1, self.d_a)
        a_data = a_data.to(e_embed_data.dtype)

        q_matrix = self.q_matrix.to(e_data.device)
        n_skills = q_matrix.size(1)

        # Initialize knowledge state
        h_pre = nn.init.xavier_uniform_(
            torch.zeros(n_skills, self.d_k)
        ).repeat(batch_size, 1, 1).to(e_data.device)
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
            q_e = q_matrix[e].view(batch_size, 1, -1)

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
            c_tilde = torch.unsqueeze(
                torch.sum(torch.squeeze(q_matrix[e_next].view(batch_size, 1, -1), dim=1), 1),
                -1
            )
            h_tilde = q_matrix[e_next].view(batch_size, 1, -1).bmm(h).view(batch_size, self.d_k) / c_tilde

            y = self.sig(self.linear_5(torch.cat((e_embed_data[:, t + 1], h_tilde), 1)).sum(1) / self.d_k)
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
