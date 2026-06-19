import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODEL_REGISTRY


@MODEL_REGISTRY.register("gbktv5")
@MODEL_REGISTRY.register("cgbkt")
class CGBKT(nn.Module):
    """Coverage-aware Geometric Knowledge Tracing.

    This model is a lightweight redesign of GBKT for the coverage story:
    - centers model latent student/question positions;
    - scalar radii model student knowledge coverage and question requirement scope;
    - coverage-aware BBP predicts whether the student ball covers the question ball.

    It intentionally excludes history attention, time dynamics, and contextual readout
    so that it runs much faster than GBKTV4 and isolates the coverage mechanism.
    """

    def __init__(
        self,
        num_q,
        num_c,
        emb_size=64,
        d_h=128,
        d_p=64,
        dropout=0.2,
        init_radius=0.5,
        eps=1e-6,
        coverage_gate_init=-3.0,
        max_coverage_weight=1.0,
        **kwargs,
    ):
        super().__init__()
        self.model_name = "cgbkt"
        self.num_q = int(num_q)
        self.num_c = int(num_c)
        self.emb_size = int(emb_size)
        self.d_h = int(d_h)
        self.d_p = int(d_p)
        self.eps = float(eps)
        self.max_coverage_weight = float(max_coverage_weight)

        self.question_emb = nn.Embedding(self.num_q + 1, self.emb_size, padding_idx=self.num_q)
        self.concept_emb = nn.Embedding(self.num_c + 1, self.emb_size, padding_idx=self.num_c)
        self.response_emb = nn.Embedding(2, self.emb_size)

        q_repr_dim = 2 * self.emb_size
        update_dim = q_repr_dim + self.emb_size

        self.q_repr_mlp = nn.Sequential(
            nn.Linear(q_repr_dim, q_repr_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(q_repr_dim, q_repr_dim),
        )

        # Question requirement ball: center and scalar scope radius.
        self.W_mu_q = nn.Linear(q_repr_dim, self.d_h)
        self.W_r_q = nn.Linear(q_repr_dim, 1)

        # Student knowledge coverage ball.
        self.mu_h0 = nn.Parameter(torch.zeros(self.d_h))
        init_radius = max(float(init_radius), 1e-6)
        init_rho = math.log(math.exp(init_radius) - 1.0)
        self.rho_h0 = nn.Parameter(torch.full((1,), init_rho))
        self.gru_mu = nn.GRUCell(update_dim, self.d_h)
        self.gru_radius = nn.GRUCell(update_dim + 3, 1)
        self.state_dropout = nn.Dropout(dropout)

        # Center-based matching branch.
        self.W_proj_h = nn.Linear(self.d_h, self.d_p)
        self.W_proj_q = nn.Linear(self.d_h, self.d_p)
        self.point_head = nn.Sequential(
            nn.Linear(5 * self.d_p, self.d_p),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_p, 1),
        )

        # Coverage branch uses explicit ball coverage features.
        self.coverage_head = nn.Sequential(
            nn.Linear(6, 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
        )
        self.confidence_head = nn.Sequential(
            nn.Linear(8, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        self.coverage_gate_logit = nn.Parameter(torch.tensor(float(coverage_gate_init)))

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

    def question_requirement_ball(self, q_repr):
        mu_q = self.W_mu_q(q_repr)
        r_q = F.softplus(self.W_r_q(q_repr)) + self.eps
        return mu_q, r_q

    def initial_student_ball(self, batch_size, device):
        mu_h = self.mu_h0.unsqueeze(0).expand(batch_size, -1)
        rho_h = self.rho_h0.unsqueeze(0).expand(batch_size, -1)
        return mu_h, rho_h

    def _radius(self, rho):
        return F.softplus(rho) + self.eps

    def update_state(self, mu_h, rho_h, q_repr_t, r_t, mu_q_t, r_q_t):
        r_idx = self._response_to_index(r_t)
        e_r = self.response_emb(r_idx)
        update_input = torch.cat([q_repr_t, e_r], dim=-1)

        mu_h_next = self.gru_mu(update_input, mu_h)
        mu_h_next = self.state_dropout(mu_h_next)

        with torch.no_grad():
            center_dist = torch.linalg.norm(mu_h - mu_q_t, dim=-1, keepdim=True)
        r_h = self._radius(rho_h)
        radius_input = torch.cat([update_input, r_h, r_q_t, center_dist], dim=-1)
        rho_h_next = self.gru_radius(radius_input, rho_h).clamp(min=-8.0, max=8.0)
        return mu_h_next, rho_h_next

    def coverage_aware_bbp(self, mu_h, rho_h, mu_q, r_q):
        r_h = self._radius(rho_h)
        mu_h_p = self.W_proj_h(mu_h)
        mu_q_p = self.W_proj_q(mu_q)
        center_diff = mu_h_p - mu_q_p
        abs_diff = torch.abs(center_diff)
        prod = mu_h_p * mu_q_p
        center_dist = torch.linalg.norm(center_diff, dim=-1, keepdim=True)

        point_features = torch.cat([mu_h_p, mu_q_p, center_diff, abs_diff, prod], dim=-1)
        point_logit = self.point_head(point_features).squeeze(-1)

        coverage_margin = r_h - center_dist - r_q
        overlap_margin = r_h + r_q - center_dist
        log_radius_ratio = torch.log((r_h + self.eps) / (r_q + self.eps))
        coverage_features = torch.cat(
            [coverage_margin, overlap_margin, log_radius_ratio, center_dist, r_h, r_q],
            dim=-1,
        )
        coverage_logit = self.coverage_head(coverage_features).squeeze(-1)

        gate = torch.sigmoid(self.coverage_gate_logit) * self.max_coverage_weight
        logit = point_logit + gate * coverage_logit
        p_hat = torch.sigmoid(logit)
        p_center = torch.sigmoid(point_logit)
        p_coverage = torch.sigmoid(coverage_logit)

        conf_input = torch.cat([coverage_features, point_logit.unsqueeze(-1), coverage_logit.unsqueeze(-1)], dim=-1)
        confidence = torch.sigmoid(self.confidence_head(conf_input).squeeze(-1))

        return {
            "p_hat": p_hat,
            "p_center": p_center,
            "p_coverage": p_coverage,
            "point_logit": point_logit,
            "coverage_logit": coverage_logit,
            "theta": point_logit,
            "confidence": confidence,
            "coverage_gate": gate.expand_as(point_logit),
            "r_h": r_h.squeeze(-1),
            "r_q": r_q.squeeze(-1),
            "center_dist": center_dist.squeeze(-1),
            "coverage_margin": coverage_margin.squeeze(-1),
            "overlap_margin": overlap_margin.squeeze(-1),
            "log_radius_ratio": log_radius_ratio.squeeze(-1),
        }

    def forward(self, q, c, r):
        if q is None or c is None:
            raise ValueError("CGBKT requires both question and concept sequences.")

        q = q.long()
        c = c.long()
        r = r.float()
        if q.dim() != 2:
            raise ValueError(f"CGBKT expects q shape [B, T], but got {tuple(q.shape)}")

        batch_size, seq_len = q.shape
        device = q.device
        if seq_len < 2:
            empty = torch.zeros(batch_size, 0, device=device)
            return {
                "y": empty,
                "y_ball": empty,
                "y_center": empty,
                "y_coverage": empty,
                "theta": empty,
                "coverage_logit": empty,
                "confidence": empty,
                "coverage_gate": empty,
                "r_h_mean": empty,
                "r_d_mean": empty,
                "center_dist": empty,
                "coverage_margin": empty,
                "overlap_margin": empty,
            }

        q_repr_all = self.get_question_repr(q, c)
        mu_q_all, r_q_all = self.question_requirement_ball(q_repr_all)
        mu_h, rho_h = self.initial_student_ball(batch_size, device)

        outputs = {
            "y": [],
            "y_ball": [],
            "y_center": [],
            "y_coverage": [],
            "theta": [],
            "coverage_logit": [],
            "confidence": [],
            "coverage_gate": [],
            "r_h_mean": [],
            "r_d_mean": [],
            "center_dist": [],
            "coverage_margin": [],
            "overlap_margin": [],
            "log_radius_ratio": [],
        }

        for t in range(seq_len - 1):
            q_t = q[:, t]
            q_repr_t = q_repr_all[:, t, :]
            mu_q_t = mu_q_all[:, t, :]
            r_q_t = r_q_all[:, t, :]

            mu_next, rho_next = self.update_state(mu_h, rho_h, q_repr_t, r[:, t], mu_q_t, r_q_t)
            valid_cur = ((q_t >= 0) & (r[:, t] >= 0)).float().unsqueeze(-1)
            mu_h = valid_cur * mu_next + (1.0 - valid_cur) * mu_h
            rho_h = valid_cur * rho_next + (1.0 - valid_cur) * rho_h

            pred = self.coverage_aware_bbp(mu_h, rho_h, mu_q_all[:, t + 1, :], r_q_all[:, t + 1, :])
            outputs["y"].append(pred["p_hat"])
            outputs["y_ball"].append(pred["p_hat"])
            outputs["y_center"].append(pred["p_center"])
            outputs["y_coverage"].append(pred["p_coverage"])
            outputs["theta"].append(pred["theta"])
            outputs["coverage_logit"].append(pred["coverage_logit"])
            outputs["confidence"].append(pred["confidence"])
            outputs["coverage_gate"].append(pred["coverage_gate"])
            outputs["r_h_mean"].append(pred["r_h"])
            outputs["r_d_mean"].append(pred["r_q"])
            outputs["center_dist"].append(pred["center_dist"])
            outputs["coverage_margin"].append(pred["coverage_margin"])
            outputs["overlap_margin"].append(pred["overlap_margin"])
            outputs["log_radius_ratio"].append(pred["log_radius_ratio"])

        return {key: torch.stack(value, dim=1) for key, value in outputs.items()}
