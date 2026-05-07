import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODEL_REGISTRY


class HistoricalBallAttention(nn.Module):
    """Attention over historical knowledge-state centers."""

    def __init__(self, d_h, n_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(d_h)

    def forward(self, mu_h_current, mu_h_history):
        if mu_h_history.size(1) == 0:
            return mu_h_current

        query = mu_h_current.unsqueeze(1)
        attn_out, _ = self.attn(query=query, key=mu_h_history, value=mu_h_history)
        enhanced = self.norm(query + attn_out)
        return enhanced.squeeze(1)


@MODEL_REGISTRY.register("gbkt_svm_aux")
class GBKTSVMAux(nn.Module):
    """GBKT with SVM-style auxiliary heads.

    The main prediction path is kept identical to GBKT: validation/test scores come
    from the BCE-trained p_hat. SVM heads are only used by the trainer as small
    auxiliary losses, so they do not replace the ranking signal used for AUC.
    """

    def __init__(
        self,
        num_q,
        num_c,
        emb_size=64,
        d_h=128,
        d_g=64,
        d_p=64,
        dropout=0.2,
        init_radius=0.5,
        eps=1e-6,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "gbkt_svm_aux"
        self.num_q = int(num_q)
        self.num_c = int(num_c)
        self.emb_size = int(emb_size)
        self.d_h = int(d_h)
        self.d_g = int(d_g)
        self.d_p = int(d_p)
        self.eps = float(eps)

        self.question_emb = nn.Embedding(self.num_q + 1, self.emb_size, padding_idx=self.num_q)
        self.concept_emb = nn.Embedding(self.num_c + 1, self.emb_size, padding_idx=self.num_c)
        self.response_emb = nn.Embedding(2, 2 * self.emb_size)

        self.W_mu_d = nn.Linear(2 * self.emb_size, self.d_g)
        self.W_r_d = nn.Linear(2 * self.emb_size, self.d_g)
        self.q_repr_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_size, 2 * self.emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.emb_size, 2 * self.emb_size),
        )

        self.gru_mu = nn.GRUCell(2 * self.emb_size + self.d_h, self.d_h)
        self.W_gamma = nn.Linear(2 * self.d_h + 2 * self.emb_size, self.d_h)
        self.concept_key = nn.Linear(self.emb_size, self.d_h)
        self.concept_gate = nn.Linear(2 * self.d_h, self.d_h)
        self.mu_h0 = nn.Parameter(torch.zeros(self.d_h))
        init_radius = max(float(init_radius), 1e-6)
        init_val = math.log(math.exp(init_radius) - 1.0)
        self.r_h0 = nn.Parameter(torch.full((self.d_h,), init_val))
        self.state_dropout = nn.Dropout(dropout)

        self.ball_attn = HistoricalBallAttention(self.d_h, n_heads=4, dropout=dropout)

        self.W_proj_h = nn.Linear(self.d_h, self.d_p)
        self.W_proj_rh = nn.Linear(self.d_h, self.d_p)
        self.W_proj_d = nn.Linear(self.d_g, self.d_p)
        self.W_proj_rd = nn.Linear(self.d_g, self.d_p)

        self.W_theta = nn.Linear(self.d_p, 1)
        self.W_b = nn.Linear(self.d_p, 1)
        self.W_a = nn.Linear(self.d_p, 1)
        self.W_conf = nn.Linear(self.d_p, 1)
        self.W_shortcut = nn.Linear(4 * self.d_p, 1)

        self.svm_shortcut = nn.Linear(4 * self.d_p, 1)
        self.svm_theta = nn.Linear(self.d_p, 1)
        self.svm_conf = nn.Linear(self.d_p, 1)

        self.W_pl = nn.Sequential(
            nn.Linear(3 * self.d_p + 1, self.d_p),
            nn.ReLU(),
            nn.Linear(self.d_p, self.d_p),
        )
        self.W_mu_delta = nn.Linear(4 * self.emb_size + self.d_h, self.d_h)
        self.W_pl2h = nn.Linear(self.d_p, self.d_h)
        self.W_r_delta = nn.Linear(self.d_p, self.d_h)

    def _sanitize_question_ids(self, q):
        q = q.long()
        pad_q = torch.full_like(q, self.num_q)
        return torch.where((q >= 0) & (q < self.num_q), q, pad_q)

    def _sanitize_concept_ids(self, c):
        c = c.long()
        pad_c = torch.full_like(c, self.num_c)
        return torch.where((c >= 0) & (c < self.num_c), c, pad_c)

    @staticmethod
    def _response_to_index(r):
        return (r > 0.5).long().clamp(min=0, max=1)

    def get_avg_concept_emb(self, c):
        cidx = self._sanitize_concept_ids(c)
        cemb = self.concept_emb(cidx)
        if cidx.dim() <= 2:
            return cemb

        cmask = (cidx != self.num_c).float().unsqueeze(-1)
        csum = (cemb * cmask).sum(dim=-2)
        cnum = cmask.sum(dim=-2).clamp(min=1.0)
        return csum / cnum

    def get_question_repr(self, q, c):
        qidx = self._sanitize_question_ids(q)
        qemb = self.question_emb(qidx)
        cemb = self.get_avg_concept_emb(c)
        shallow = torch.cat([qemb + cemb, qemb * cemb], dim=-1)
        return shallow + self.q_repr_mlp(shallow)

    def question_difficulty_ball(self, q_repr):
        mu_d = self.W_mu_d(q_repr)
        r_d = F.softplus(self.W_r_d(q_repr))
        return mu_d, r_d

    def _project_balls(self, mu_h, r_h, mu_d, r_d):
        mu_h_p = self.W_proj_h(mu_h)
        r_h_p = F.softplus(self.W_proj_rh(r_h))
        mu_d_p = self.W_proj_d(mu_d)
        r_d_p = F.softplus(self.W_proj_rd(r_d))
        return mu_h_p, r_h_p, mu_d_p, r_d_p

    def ball_to_ball_predict(self, mu_h, r_h, mu_d, r_d):
        mu_h_p, r_h_p, mu_d_p, r_d_p = self._project_balls(mu_h, r_h, mu_d, r_d)

        center_diff = mu_h_p - mu_d_p
        radius_sum = r_h_p + r_d_p
        effective_diff = center_diff / (radius_sum + self.eps)

        theta = self.W_theta(effective_diff).squeeze(-1)
        b = self.W_b(center_diff).squeeze(-1)
        a = F.softplus(self.W_a(-radius_sum)).squeeze(-1)
        confidence = torch.sigmoid(self.W_conf(-radius_sum).squeeze(-1))

        logit_irt = a * (theta - b)
        shortcut_input = torch.cat([mu_h_p, mu_d_p, center_diff, mu_h_p * mu_d_p], dim=-1)
        shortcut = self.W_shortcut(shortcut_input).squeeze(-1)
        logit = logit_irt + 0.1 * shortcut
        p_hat = torch.sigmoid(logit)

        return {
            "p_hat": p_hat,
            "theta": theta,
            "b": b,
            "a": a,
            "confidence": confidence,
            "mu_h_p": mu_h_p,
            "mu_d_p": mu_d_p,
            "r_h_p": r_h_p,
            "r_d_p": r_d_p,
            "radius_sum": radius_sum,
            "feature_svm_shortcut": shortcut_input,
            "feature_svm_theta": effective_diff,
            "feature_svm_conf": -radius_sum,
        }

    def knowledge_acquisition_ball(self, mu_h, r_h, mu_d, r_d, q_repr, r_t):
        mu_h_p, r_h_p, mu_d_p, r_d_p = self._project_balls(mu_h, r_h, mu_d, r_d)

        center_diff = mu_h_p - mu_d_p
        radius_sum = r_h_p + r_d_p
        effective_diff = center_diff / (radius_sum + self.eps)

        r_idx = self._response_to_index(r_t)
        sign = r_idx.float().mul(2.0).sub(1.0).unsqueeze(-1)
        pl_input = torch.cat(
            [
                sign * center_diff,
                effective_diff,
                radius_sum,
                sign,
            ],
            dim=-1,
        )
        plausibility = torch.sigmoid(self.W_pl(pl_input))

        e_r = self.response_emb(r_idx)
        acq_input = torch.cat([q_repr, e_r, mu_h], dim=-1)
        mu_delta_raw = self.W_mu_delta(acq_input)
        pl_gate = torch.sigmoid(self.W_pl2h(plausibility))
        mu_delta = pl_gate * mu_delta_raw

        r_delta = F.softplus(self.W_r_delta(1.0 - plausibility))
        return mu_delta, r_delta, plausibility

    def update_state(self, mu_h, r_h, q_repr, r_t, mu_delta, r_delta, c_emb_t):
        r_idx = self._response_to_index(r_t)
        e_r = self.response_emb(r_idx)

        x_t = torch.cat([q_repr + e_r, mu_delta], dim=-1)
        mu_h_gru = self.gru_mu(x_t, mu_h)
        mu_h_gru = self.state_dropout(mu_h_gru)

        c_key = torch.sigmoid(self.concept_key(c_emb_t))
        gate_input = torch.cat([mu_h_gru - mu_h, c_key * mu_h], dim=-1)
        c_gate = torch.sigmoid(self.concept_gate(gate_input))
        update_mask = c_key * c_gate
        mu_h_next = (1.0 - update_mask) * mu_h + update_mask * mu_h_gru

        gamma_input = torch.cat([mu_h, r_h, q_repr + e_r], dim=-1)
        gamma = torch.sigmoid(self.W_gamma(gamma_input))
        r_h_next = (1.0 - gamma * c_key) * r_h + (gamma * c_key) * r_delta
        return mu_h_next, r_h_next

    def forward(self, q, c, r):
        if q is None or c is None:
            raise ValueError("GBKTSVMAux requires both question and concept sequences.")

        q = q.long()
        c = c.long()
        r = r.float()

        if q.dim() != 2:
            raise ValueError(f"GBKTSVMAux expects q shape [B, T], but got {tuple(q.shape)}")

        batch_size, seq_len = q.shape
        device = q.device
        if seq_len < 2:
            empty = torch.zeros(batch_size, 0, device=device)
            return {
                "y": empty,
                "theta": empty,
                "confidence": empty,
                "r_h_mean": empty,
                "r_d_mean": empty,
                "feature_svm_shortcut": torch.zeros(batch_size, 0, 4 * self.d_p, device=device),
                "feature_svm_theta": torch.zeros(batch_size, 0, self.d_p, device=device),
                "feature_svm_conf": torch.zeros(batch_size, 0, self.d_p, device=device),
            }

        q_repr_all = self.get_question_repr(q, c)
        mu_d_all, r_d_all = self.question_difficulty_ball(q_repr_all)

        mu_h = self.mu_h0.unsqueeze(0).expand(batch_size, -1)
        r_h = F.softplus(self.r_h0).unsqueeze(0).expand(batch_size, -1)

        p_list, theta_list, conf_list = [], [], []
        r_h_list, r_d_list = [], []
        svm_shortcut_list, svm_theta_list, svm_conf_list = [], [], []
        mu_h_history_list = []

        for t in range(seq_len - 1):
            q_t = q[:, t]
            r_t = r[:, t]

            q_repr_t = q_repr_all[:, t, :]
            mu_d_t, r_d_t = mu_d_all[:, t, :], r_d_all[:, t, :]

            if c.dim() == 2:
                c_emb_t = self.get_avg_concept_emb(c[:, t])
            else:
                c_emb_t = self.get_avg_concept_emb(c[:, t:t + 1, :]).squeeze(1)

            mu_delta, r_delta, _ = self.knowledge_acquisition_ball(
                mu_h, r_h, mu_d_t, r_d_t, q_repr_t, r_t
            )
            mu_h_next, r_h_next = self.update_state(
                mu_h, r_h, q_repr_t, r_t, mu_delta, r_delta, c_emb_t
            )

            if mu_h_history_list:
                mu_h_history = torch.stack(mu_h_history_list, dim=1)
                mu_h_next = self.ball_attn(mu_h_next, mu_h_history)

            valid_cur = (q_t >= 0).float().unsqueeze(-1)
            mu_h = valid_cur * mu_h_next + (1.0 - valid_cur) * mu_h
            r_h = valid_cur * r_h_next + (1.0 - valid_cur) * r_h
            mu_h_history_list.append(mu_h.detach())

            mu_d_next, r_d_next = mu_d_all[:, t + 1, :], r_d_all[:, t + 1, :]
            pred = self.ball_to_ball_predict(mu_h, r_h, mu_d_next, r_d_next)

            p_list.append(pred["p_hat"])
            theta_list.append(pred["theta"])
            conf_list.append(pred["confidence"])
            r_h_list.append(r_h.mean(dim=-1))
            r_d_list.append(r_d_next.mean(dim=-1))
            svm_shortcut_list.append(pred["feature_svm_shortcut"])
            svm_theta_list.append(pred["feature_svm_theta"])
            svm_conf_list.append(pred["feature_svm_conf"])

        return {
            "y": torch.stack(p_list, dim=1),
            "theta": torch.stack(theta_list, dim=1),
            "confidence": torch.stack(conf_list, dim=1),
            "r_h_mean": torch.stack(r_h_list, dim=1),
            "r_d_mean": torch.stack(r_d_list, dim=1),
            "feature_svm_shortcut": torch.stack(svm_shortcut_list, dim=1),
            "feature_svm_theta": torch.stack(svm_theta_list, dim=1),
            "feature_svm_conf": torch.stack(svm_conf_list, dim=1),
        }
