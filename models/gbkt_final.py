import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODEL_REGISTRY


def _softplus_inverse(value):
    value = max(float(value), 1e-8)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


@MODEL_REGISTRY.register("gbkt_final")
class GBKTFinal(nn.Module):
    """Coverage-radius GBKT final model.

    This final version keeps a clean three-module story:
    1. Represent each exercise as a requirement ball.
    2. Update the student knowledge coverage ball from response-conditioned geometry.
    3. Predict by coverage-aware relative geometry with an IRT-style decoder.

    Dimension convention:
    - B: batch size
    - T: sequence length
    - K: number of concepts attached to one question
    - E: embedding size
    - H: student knowledge center dimension
    - G: exercise requirement center dimension
    - P: shared prediction-space dimension
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
        self.model_name = "gbkt_final"
        self.num_q = int(num_q)
        self.num_c = int(num_c)
        self.emb_size = int(emb_size)
        self.d_h = int(d_h)
        self.d_g = int(d_g)
        self.d_p = int(d_p)
        self.eps = float(eps)

        # Module 1: exercise requirement ball representation.
        self.question_emb = nn.Embedding(self.num_q + 1, self.emb_size, padding_idx=self.num_q)
        self.concept_emb = nn.Embedding(self.num_c + 1, self.emb_size, padding_idx=self.num_c)
        self.response_emb = nn.Embedding(2, 2 * self.emb_size)
        self.q_repr_mlp = nn.Sequential(
            nn.Linear(2 * self.emb_size, 2 * self.emb_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.emb_size, 2 * self.emb_size),
        )
        self.W_mu_q = nn.Linear(2 * self.emb_size, self.d_g)
        self.W_r_q = nn.Linear(2 * self.emb_size, 1)

        # Module 2: response-conditioned student coverage ball update.
        self.mu_h0 = nn.Parameter(torch.zeros(self.d_h))
        self.rho_h0 = nn.Parameter(torch.tensor([_softplus_inverse(init_radius)], dtype=torch.float32))
        self.W_proj_h = nn.Linear(self.d_h, self.d_p)
        self.W_proj_q = nn.Linear(self.d_g, self.d_p)

        coverage_dim = 6
        self.W_pl = nn.Sequential(
            nn.Linear(self.d_p + coverage_dim + 1, self.d_p),
            nn.ReLU(),
            nn.Linear(self.d_p, self.d_p),
        )
        self.W_mu_delta = nn.Linear(4 * self.emb_size + self.d_h, self.d_h)
        self.W_pl2h = nn.Linear(self.d_p, self.d_h)
        self.gru_mu = nn.GRUCell(2 * self.emb_size + self.d_h, self.d_h)
        self.gru_radius = nn.GRUCell(2 * self.emb_size + coverage_dim, 1)
        self.concept_key = nn.Linear(self.emb_size, self.d_h)
        self.concept_gate = nn.Linear(2 * self.d_h, self.d_h)
        self.state_dropout = nn.Dropout(dropout)

        # Module 3: coverage-aware relative geometric IRT-style prediction.
        pred_dim = self.d_p + coverage_dim
        self.W_theta = nn.Linear(pred_dim, 1)
        self.W_b = nn.Linear(pred_dim, 1)
        self.W_a = nn.Linear(pred_dim, 1)
        self.W_conf = nn.Linear(pred_dim, 1)
        self.W_shortcut = nn.Linear(4 * self.d_p + coverage_dim, 1)

    def _sanitize_question_ids(self, q):
        qidx = q.long()
        invalid = (qidx < 0) | (qidx >= self.num_q)
        return torch.where(invalid, torch.full_like(qidx, self.num_q), qidx)

    def _sanitize_concept_ids(self, c):
        cidx = c.long()
        invalid = (cidx < 0) | (cidx >= self.num_c)
        return torch.where(invalid, torch.full_like(cidx, self.num_c), cidx)

    @staticmethod
    def _response_to_index(r):
        return r.float().ge(0.5).long().clamp(min=0, max=1)

    def _radius(self, rho):
        return F.softplus(rho).clamp(min=self.eps, max=50.0)

    def get_avg_concept_emb(self, c):
        """Return concept embedding.

        Inputs:
        - c: [B], [B, T], [B, K], or [B, T, K]

        Outputs:
        - single-concept input: [B, E] or [B, T, E]
        - multi-concept input: [B, E] or [B, T, E], masked average over K
        """
        cidx = self._sanitize_concept_ids(c)
        cemb = self.concept_emb(cidx)
        if cidx.dim() <= 2:
            return cemb

        mask = (cidx != self.num_c).float().unsqueeze(-1)
        summed = (cemb * mask).sum(dim=-2)
        denom = mask.sum(dim=-2).clamp(min=1.0)
        return summed / denom

    def get_question_repr(self, q, c):
        """Build question-concept representation.

        Inputs:
        - q: [B] or [B, T]
        - c: [B], [B, T], [B, K], or [B, T, K]

        Output:
        - q_repr: [B, 2E] or [B, T, 2E]
        """
        qemb = self.question_emb(self._sanitize_question_ids(q))
        cemb = self.get_avg_concept_emb(c)
        shallow = torch.cat([qemb + cemb, qemb * cemb], dim=-1)
        return shallow + self.q_repr_mlp(shallow)

    def exercise_requirement_ball(self, q_repr):
        """Map question representation to an exercise requirement ball.

        Input:
        - q_repr: [B, 2E] or [B, T, 2E]

        Outputs:
        - mu_q: [B, G] or [B, T, G]
        - r_q: [B, 1] or [B, T, 1]
        """
        mu_q = self.W_mu_q(q_repr)
        r_q = F.softplus(self.W_r_q(q_repr)).clamp(min=self.eps, max=50.0)
        return mu_q, r_q

    def initial_student_ball(self, batch_size, device):
        """Return initial student ball: mu_h [B, H], rho_h [B, 1]."""
        mu_h = self.mu_h0.unsqueeze(0).expand(batch_size, -1)
        rho_h = self.rho_h0.unsqueeze(0).expand(batch_size, -1).to(device=device)
        return mu_h, rho_h

    def _project_centers(self, mu_h, mu_q):
        """Project student and exercise centers to shared prediction space."""
        return self.W_proj_h(mu_h), self.W_proj_q(mu_q)

    def _coverage_features(self, mu_h, rho_h, mu_q, r_q):
        """Compute relative center and scalar coverage features.

        Outputs:
        - z_h/z_q/center_diff: [B, P]
        - coverage_features: [B, 6]
        - scalar fields: [B]
        """
        r_h = self._radius(rho_h)
        z_h, z_q = self._project_centers(mu_h, mu_q)
        center_diff = z_h - z_q
        center_dist = torch.linalg.norm(center_diff, dim=-1, keepdim=True)
        coverage_margin = r_h - center_dist - r_q
        overlap_margin = r_h + r_q - center_dist
        log_radius_ratio = torch.log((r_h + self.eps) / (r_q + self.eps))
        coverage_features = torch.cat(
            [coverage_margin, overlap_margin, log_radius_ratio, center_dist, r_h, r_q],
            dim=-1,
        )
        return {
            "z_h": z_h,
            "z_q": z_q,
            "center_diff": center_diff,
            "coverage_features": coverage_features,
            "r_h": r_h.squeeze(-1),
            "r_q": r_q.squeeze(-1),
            "center_dist": center_dist.squeeze(-1),
            "coverage_margin": coverage_margin.squeeze(-1),
            "overlap_margin": overlap_margin.squeeze(-1),
            "log_radius_ratio": log_radius_ratio.squeeze(-1),
        }

    def coverage_irt_predict(self, mu_h, rho_h, mu_q, r_q):
        """Predict correctness by coverage-aware relative geometry.

        Inputs:
        - mu_h/rho_h: [B, H], [B, 1]
        - mu_q/r_q: [B, G], [B, 1]

        Outputs:
        - p_hat/theta/b/a/confidence: [B]
        - coverage diagnostics: [B]
        """
        geo = self._coverage_features(mu_h, rho_h, mu_q, r_q)
        z_h = geo["z_h"]
        z_q = geo["z_q"]
        center_diff = geo["center_diff"]
        coverage_features = geo["coverage_features"]
        pred_features = torch.cat([center_diff, coverage_features], dim=-1)

        theta = self.W_theta(pred_features).squeeze(-1)
        b = self.W_b(pred_features).squeeze(-1)
        a = F.softplus(self.W_a(pred_features)).squeeze(-1)
        confidence = torch.sigmoid(self.W_conf(pred_features).squeeze(-1))

        shortcut_input = torch.cat([z_h, z_q, center_diff, z_h * z_q, coverage_features], dim=-1)
        shortcut = self.W_shortcut(shortcut_input).squeeze(-1)
        logit = a * (theta - b) + 0.1 * shortcut
        p_hat = torch.sigmoid(logit)

        geo.update(
            {
                "p_hat": p_hat,
                "theta": theta,
                "b": b,
                "a": a,
                "confidence": confidence,
            }
        )
        return geo

    def knowledge_acquisition_ball(self, mu_h, rho_h, mu_q, r_q, q_repr, r_t):
        """Generate response-conditioned update evidence.

        Inputs:
        - mu_h/rho_h: [B, H], [B, 1]
        - mu_q/r_q: [B, G], [B, 1]
        - q_repr: [B, 2E]
        - r_t: [B]

        Outputs:
        - delta_h: [B, H]
        - plausibility: [B, P]
        - coverage_features: [B, 6]
        """
        geo = self._coverage_features(mu_h, rho_h, mu_q, r_q)
        center_diff = geo["center_diff"]
        coverage_features = geo["coverage_features"]

        r_idx = self._response_to_index(r_t)
        sign = r_idx.float().mul(2.0).sub(1.0).unsqueeze(-1)
        pl_input = torch.cat([sign * center_diff, coverage_features, sign], dim=-1)
        plausibility = torch.sigmoid(self.W_pl(pl_input))

        e_r = self.response_emb(r_idx)
        acq_input = torch.cat([q_repr, e_r, mu_h], dim=-1)
        delta_raw = self.W_mu_delta(acq_input)
        pl_gate = torch.sigmoid(self.W_pl2h(plausibility))
        delta_h = pl_gate * delta_raw
        return delta_h, plausibility, coverage_features

    def update_state(self, mu_h, rho_h, q_repr, r_t, delta_h, coverage_features, c_emb_t):
        """Update student coverage ball.

        Inputs:
        - mu_h/rho_h: [B, H], [B, 1]
        - q_repr: [B, 2E]
        - r_t: [B]
        - delta_h: [B, H]
        - coverage_features: [B, 6]
        - c_emb_t: [B, E]

        Outputs:
        - mu_h_next/rho_h_next: [B, H], [B, 1]
        """
        r_idx = self._response_to_index(r_t)
        e_r = self.response_emb(r_idx)
        update_input = q_repr + e_r

        mu_gru = self.state_dropout(self.gru_mu(torch.cat([update_input, delta_h], dim=-1), mu_h))
        c_key = torch.sigmoid(self.concept_key(c_emb_t))
        gate_input = torch.cat([mu_gru - mu_h, c_key * mu_h], dim=-1)
        c_gate = torch.sigmoid(self.concept_gate(gate_input))
        update_mask = c_key * c_gate
        mu_h_next = (1.0 - update_mask) * mu_h + update_mask * mu_gru

        rho_input = torch.cat([update_input, coverage_features], dim=-1)
        rho_h_next = self.gru_radius(rho_input, rho_h).clamp(min=-8.0, max=8.0)
        return mu_h_next, rho_h_next

    def forward(self, q, c, r, it=None, ut=None):
        """Run sequence prediction.

        Inputs:
        - q: [B, T]
        - c: [B, T] or [B, T, K]
        - r: [B, T]
        - it/ut: accepted for trainer compatibility, unused in coverage-radius model

        Outputs:
        - y/y_ball/theta/confidence/r_h_mean/r_d_mean: [B, T-1]
        - center_dist/coverage_margin/overlap_margin/log_radius_ratio: [B, T-1]
        """
        if q is None or c is None:
            raise ValueError("GBKTFinal requires both question and concept sequences.")

        q = q.long()
        c = c.long()
        r = r.float()
        if q.dim() != 2:
            raise ValueError(f"GBKTFinal expects q shape [B, T], got {tuple(q.shape)}.")

        batch_size, seq_len = q.shape
        device = q.device
        if seq_len < 2:
            empty = torch.zeros(batch_size, 0, device=device)
            return {
                "y": empty,
                "y_ball": empty,
                "theta": empty,
                "confidence": empty,
                "r_h_mean": empty,
                "r_d_mean": empty,
                "center_dist": empty,
                "coverage_margin": empty,
                "overlap_margin": empty,
                "log_radius_ratio": empty,
            }

        q_repr_all = self.get_question_repr(q, c)
        mu_q_all, r_q_all = self.exercise_requirement_ball(q_repr_all)
        mu_h, rho_h = self.initial_student_ball(batch_size, device)

        outputs = {
            "y": [],
            "y_ball": [],
            "theta": [],
            "confidence": [],
            "r_h_mean": [],
            "r_d_mean": [],
            "center_dist": [],
            "coverage_margin": [],
            "overlap_margin": [],
            "log_radius_ratio": [],
        }

        for t in range(seq_len - 1):
            q_t = q[:, t]
            r_t = r[:, t]
            q_repr_t = q_repr_all[:, t, :]
            mu_q_t = mu_q_all[:, t, :]
            r_q_t = r_q_all[:, t, :]

            if c.dim() == 2:
                c_emb_t = self.get_avg_concept_emb(c[:, t])
            else:
                c_emb_t = self.get_avg_concept_emb(c[:, t:t + 1, :]).squeeze(1)

            delta_h, _, coverage_features = self.knowledge_acquisition_ball(
                mu_h,
                rho_h,
                mu_q_t,
                r_q_t,
                q_repr_t,
                r_t,
            )
            mu_next, rho_next = self.update_state(
                mu_h,
                rho_h,
                q_repr_t,
                r_t,
                delta_h,
                coverage_features,
                c_emb_t,
            )

            valid_cur = (q_t >= 0).float().unsqueeze(-1)
            mu_h = valid_cur * mu_next + (1.0 - valid_cur) * mu_h
            rho_h = valid_cur * rho_next + (1.0 - valid_cur) * rho_h

            pred = self.coverage_irt_predict(mu_h, rho_h, mu_q_all[:, t + 1, :], r_q_all[:, t + 1, :])
            outputs["y"].append(pred["p_hat"])
            outputs["y_ball"].append(pred["p_hat"])
            outputs["theta"].append(pred["theta"])
            outputs["confidence"].append(pred["confidence"])
            outputs["r_h_mean"].append(pred["r_h"])
            outputs["r_d_mean"].append(pred["r_q"])
            outputs["center_dist"].append(pred["center_dist"])
            outputs["coverage_margin"].append(pred["coverage_margin"])
            outputs["overlap_margin"].append(pred["overlap_margin"])
            outputs["log_radius_ratio"].append(pred["log_radius_ratio"])

        return {key: torch.stack(value, dim=1) for key, value in outputs.items()}
