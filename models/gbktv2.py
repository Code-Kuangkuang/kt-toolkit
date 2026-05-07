import torch
import torch.nn as nn

from core.registry import MODEL_REGISTRY
from models.gbkt import GBKT


@MODEL_REGISTRY.register("gbktv2")
class GBKTV2(GBKT):
    """Conservative GBKT v2.

    This version intentionally keeps the original GBKT ball dynamics intact.
    The previous experimental v2 added concept attention, time decay, item bias,
    and AUC loss together; on ASSIST2009 that was a clear negative transfer.

    The only structural addition here is a QIKT-style next-concept auxiliary
    decoder. It gives direct concept-level supervision while preserving the
    original GBKT prediction path that already works well on ASSIST2009.
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
        super().__init__(
            num_q=num_q,
            num_c=num_c,
            emb_size=emb_size,
            d_h=d_h,
            d_g=d_g,
            d_p=d_p,
            dropout=dropout,
            init_radius=init_radius,
            eps=eps,
            **kwargs,
        )
        self.model_name = "gbktv2"
        self.concept_next_head = nn.Sequential(
            nn.Linear(self.d_h + self.emb_size, self.d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_h, self.num_c),
        )
        self.fusion_logit_weight = nn.Parameter(torch.tensor(0.0))

    def _concept_next_prob(self, concept_logits, c_next):
        cidx = self._sanitize_concept_ids(c_next)
        probs = torch.sigmoid(concept_logits)

        if cidx.dim() == 1:
            safe_idx = cidx.clamp(max=self.num_c - 1)
            gathered = probs.gather(dim=-1, index=safe_idx.unsqueeze(-1)).squeeze(-1)
            return torch.where(cidx == self.num_c, torch.zeros_like(gathered), gathered)

        mask = cidx != self.num_c
        safe_idx = cidx.clamp(max=self.num_c - 1)
        gathered = probs.gather(
            dim=-1,
            index=safe_idx,
        )
        gathered = gathered * mask.float()
        denom = mask.float().sum(dim=-1).clamp(min=1.0)
        return gathered.sum(dim=-1) / denom

    def forward(self, q, c, r, it=None):
        if q is None or c is None:
            raise ValueError("GBKTV2 requires both question and concept sequences.")

        q = q.long()
        c = c.long()
        r = r.float()

        if q.dim() != 2:
            raise ValueError(f"GBKTV2 expects q shape [B, T], but got {tuple(q.shape)}")

        batch_size, seq_len = q.shape
        device = q.device
        if seq_len < 2:
            empty = torch.zeros(batch_size, 0, device=device)
            return {
                "y": empty,
                "y_ball": empty,
                "y_concept_next": empty,
                "theta": empty,
                "confidence": empty,
                "r_h_mean": empty,
                "r_d_mean": empty,
            }

        q_repr_all = self.get_question_repr(q, c)
        mu_d_all, r_d_all = self.question_difficulty_ball(q_repr_all)

        mu_h = self.mu_h0.unsqueeze(0).expand(batch_size, -1)
        r_h = torch.nn.functional.softplus(self.r_h0).unsqueeze(0).expand(batch_size, -1)

        p_list, p_ball_list, p_concept_list = [], [], []
        theta_list, conf_list = [], []
        r_h_list, r_d_list = [], []
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
            p_ball = pred["p_hat"]

            if c.dim() == 2:
                c_emb_next = self.get_avg_concept_emb(c[:, t + 1])
                c_next = c[:, t + 1]
            else:
                c_emb_next = self.get_avg_concept_emb(c[:, t + 1:t + 2, :]).squeeze(1)
                c_next = c[:, t + 1, :]
            concept_logits = self.concept_next_head(torch.cat([mu_h, c_emb_next], dim=-1))
            p_concept = self._concept_next_prob(concept_logits, c_next)

            ball_logit = torch.logit(p_ball.clamp(1e-5, 1.0 - 1e-5))
            concept_logit = torch.logit(p_concept.clamp(1e-5, 1.0 - 1e-5))
            fusion_w = torch.sigmoid(self.fusion_logit_weight) * 0.25
            p_fused = torch.sigmoid(ball_logit + fusion_w * concept_logit)

            p_list.append(p_fused)
            p_ball_list.append(p_ball)
            p_concept_list.append(p_concept)
            theta_list.append(pred["theta"])
            conf_list.append(pred["confidence"])
            r_h_list.append(r_h.mean(dim=-1))
            r_d_list.append(r_d_next.mean(dim=-1))

        return {
            "y": torch.stack(p_list, dim=1),
            "y_ball": torch.stack(p_ball_list, dim=1),
            "y_concept_next": torch.stack(p_concept_list, dim=1),
            "theta": torch.stack(theta_list, dim=1),
            "confidence": torch.stack(conf_list, dim=1),
            "r_h_mean": torch.stack(r_h_list, dim=1),
            "r_d_mean": torch.stack(r_d_list, dim=1),
        }
