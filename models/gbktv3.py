import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODEL_REGISTRY
from models.gbktv2 import GBKTV2


@MODEL_REGISTRY.register("gbktv3")
class GBKTV3(GBKTV2):
    """GBKT v3: item-aware and time-aware extension of GBKT v2.

    Compared with GBKTV2, this version keeps the same ball backbone and
    concept-next decoder, then adds:
    - a next-question decoder for item-level supervision;
    - interval/use-time features in KAB plausibility and state uncertainty;
    - a light time-forgetting step before each knowledge update.
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
        max_time_minutes=43200,
        max_use_time_ms=600000,
        use_time_center_decay=False,
        fusion_gate_init=0.02,
        max_fusion_weight=0.30,
        max_concept_fusion_weight=0.30,
        max_question_fusion_weight=0.0,
        use_residual_fusion=True,
        use_concept_readout=True,
        concept_readout_init=0.05,
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
        self.model_name = "gbktv3"
        self.max_time_minutes = float(max_time_minutes)
        self.max_use_time_ms = float(max_use_time_ms)
        self.use_time_center_decay = bool(use_time_center_decay)
        self.max_fusion_weight = float(max_fusion_weight)
        self.max_concept_fusion_weight = (
            float(max_concept_fusion_weight)
            if max_concept_fusion_weight is not None
            else self.max_fusion_weight
        )
        self.max_question_fusion_weight = (
            float(max_question_fusion_weight)
            if max_question_fusion_weight is not None
            else self.max_fusion_weight
        )
        self.register_buffer(
            "fusion_max_weights",
            torch.tensor(
                [self.max_concept_fusion_weight, self.max_question_fusion_weight],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.use_residual_fusion = bool(use_residual_fusion)
        self.use_concept_readout = bool(use_concept_readout)

        self.question_next_head = nn.Sequential(
            nn.Linear(self.d_h + 2 * self.emb_size + 2 * self.d_g, self.d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_h, 1),
        )

        self.time_feat_proj = nn.Sequential(
            nn.Linear(2, self.d_p),
            nn.GELU(),
            nn.Linear(self.d_p, self.d_p),
        )
        self.time_state_proj = nn.Sequential(
            nn.Linear(2, self.d_h),
            nn.GELU(),
            nn.Linear(self.d_h, self.d_h),
        )
        self.W_pl_time = nn.Sequential(
            nn.Linear(4 * self.d_p + 1, self.d_p),
            nn.ReLU(),
            nn.Linear(self.d_p, self.d_p),
        )
        self.W_mu_delta_time = nn.Linear(4 * self.emb_size + 2 * self.d_h, self.d_h)
        self.W_r_delta_time = nn.Linear(self.d_p + self.d_h, self.d_h)
        self.time_decay_head = nn.Linear(2, self.d_h)
        self.time_radius_head = nn.Linear(2, self.d_h)

        self.time_kab_strength = nn.Parameter(torch.tensor(-2.5))
        self.time_forget_strength = nn.Parameter(torch.tensor(-3.0))

        readout_dim = self.d_h + self.emb_size + 2 * self.d_g
        self.concept_readout_mu = nn.Sequential(
            nn.Linear(readout_dim, self.d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_h, self.d_h),
        )
        self.concept_readout_radius = nn.Sequential(
            nn.Linear(readout_dim, self.d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_h, self.d_h),
        )
        self.concept_readout_strength = nn.Parameter(
            torch.tensor(self._logit_init(float(concept_readout_init)))
        )
        self._init_small_last(self.concept_readout_mu)
        self._init_small_last(self.concept_readout_radius)

        self.fusion_gate = nn.Sequential(
            nn.Linear(self.d_h + 2 * self.emb_size + 2 * self.d_g + 2, self.d_h),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.d_h, 2),
        )
        self._init_fusion_gate(float(fusion_gate_init))

    @staticmethod
    def _logit_init(init_weight):
        init_weight = max(min(float(init_weight), 1.0 - 1e-4), 1e-4)
        return math.log(init_weight / (1.0 - init_weight))

    @staticmethod
    def _init_small_last(module, gain=1e-3):
        final_layer = module[-1]
        nn.init.xavier_uniform_(final_layer.weight, gain=gain)
        nn.init.zeros_(final_layer.bias)

    def _init_fusion_gate(self, init_weight):
        init_bias = []
        for max_weight in [self.max_concept_fusion_weight, self.max_question_fusion_weight]:
            if max_weight <= 1e-6:
                init_bias.append(self._logit_init(1e-4))
                continue
            branch_init = max(min(init_weight, max_weight - 1e-4), 1e-4)
            init_bias.append(self._logit_init(branch_init / max_weight))
        final_layer = self.fusion_gate[-1]
        nn.init.zeros_(final_layer.weight)
        with torch.no_grad():
            final_layer.bias.copy_(torch.tensor(init_bias, dtype=final_layer.bias.dtype))

    def _align_time_sequence(self, seq, batch_size, seq_len, device):
        if seq is None:
            return torch.zeros(batch_size, seq_len, device=device)
        seq = seq.to(device=device).float()
        if seq.dim() == 1:
            seq = seq.unsqueeze(0)
        if seq.size(1) < seq_len:
            pad = torch.zeros(seq.size(0), seq_len - seq.size(1), device=device, dtype=seq.dtype)
            seq = torch.cat([seq, pad], dim=1)
        return seq[:, :seq_len]

    def _build_time_features(self, it, ut, batch_size, seq_len, device):
        interval = self._align_time_sequence(it, batch_size, seq_len, device)
        use_time = self._align_time_sequence(ut, batch_size, seq_len, device)

        interval = interval.clamp(min=0.0, max=self.max_time_minutes)
        use_time = use_time.clamp(min=0.0, max=self.max_use_time_ms)

        interval = torch.log1p(interval) / torch.log1p(
            torch.tensor(self.max_time_minutes, device=device)
        )
        use_time = torch.log1p(use_time) / torch.log1p(
            torch.tensor(self.max_use_time_ms, device=device)
        )
        return torch.stack([interval, use_time], dim=-1)

    @staticmethod
    def _time_presence(time_feat):
        return (time_feat.abs().sum(dim=-1, keepdim=True) > 0).float()

    def _apply_time_forgetting(self, mu_h, r_h, time_feat):
        interval = time_feat[:, :1]
        strength = torch.sigmoid(self.time_forget_strength)
        radius_delta = F.softplus(self.time_radius_head(time_feat))

        if self.use_time_center_decay:
            decay_rate = F.softplus(self.time_decay_head(time_feat))
            decay = torch.exp(-strength * decay_rate * interval)
            mu_h = mu_h * decay
        r_h = (r_h + strength * radius_delta * interval).clamp(min=self.eps, max=50.0)
        return mu_h, r_h

    def _condition_state_for_next_concept(self, mu_h, r_h, c_emb_next, mu_d_next, r_d_next):
        if not self.use_concept_readout:
            return mu_h, r_h

        readout_input = torch.cat([mu_h, c_emb_next, mu_d_next, r_d_next], dim=-1)
        strength = torch.sigmoid(self.concept_readout_strength)
        mu_shift = self.concept_readout_mu(readout_input)
        if self.use_point_space:
            return mu_h + strength * mu_shift, r_h
        radius_shift = torch.tanh(self.concept_readout_radius(readout_input))
        mu_h_next = mu_h + strength * mu_shift
        r_h_next = (r_h + strength * radius_shift).clamp(min=self.eps, max=50.0)
        return mu_h_next, r_h_next

    def _fuse_logits(self, ball_logit, concept_logit, question_logit, concept_w, question_w):
        if self.use_residual_fusion:
            return (
                ball_logit
                + concept_w * (concept_logit - ball_logit)
                + question_w * (question_logit - ball_logit)
            )
        return ball_logit + concept_w * concept_logit + question_w * question_logit

    def knowledge_acquisition_ball_time(self, mu_h, r_h, mu_d, r_d, q_repr, r_t, time_feat):
        mu_h_p, r_h_p, mu_d_p, r_d_p = self._project_balls(mu_h, r_h, mu_d, r_d)

        center_diff = mu_h_p - mu_d_p
        radius_sum = r_h_p + r_d_p
        effective_diff = self._effective_diff(
            center_diff,
            radius_sum,
            use_radius=self.use_kab_radius_features,
        )
        radius_feature = self._radius_feature(radius_sum, use_radius=self.use_kab_radius_features)

        r_idx = self._response_to_index(r_t)
        sign = r_idx.float().mul(2.0).sub(1.0).unsqueeze(-1)
        pl_input = torch.cat(
            [
                sign * center_diff,
                effective_diff,
                radius_feature,
                sign,
            ],
            dim=-1,
        )

        presence = self._time_presence(time_feat)
        time_p = self.time_feat_proj(time_feat) * presence
        time_state = self.time_state_proj(time_feat) * presence
        time_strength = torch.sigmoid(self.time_kab_strength)

        plausibility_logits = self.W_pl(pl_input)
        plausibility_time = self.W_pl_time(torch.cat([pl_input, time_p], dim=-1))
        plausibility = torch.sigmoid(plausibility_logits + time_strength * plausibility_time)

        e_r = self.response_emb(r_idx)
        acq_input = torch.cat([q_repr, e_r, mu_h], dim=-1)
        acq_time_input = torch.cat([q_repr, e_r, mu_h, time_state], dim=-1)
        mu_delta_raw = self.W_mu_delta(acq_input) + time_strength * self.W_mu_delta_time(acq_time_input)
        pl_gate = torch.sigmoid(self.W_pl2h(plausibility))
        mu_delta = pl_gate * mu_delta_raw

        r_delta_base = self.W_r_delta(1.0 - plausibility)
        r_delta_time = self.W_r_delta_time(torch.cat([1.0 - plausibility, time_state], dim=-1))
        r_delta = F.softplus(r_delta_base + time_strength * r_delta_time)
        return mu_delta, r_delta, plausibility

    def forward(self, q, c, r, it=None, ut=None):
        if q is None or c is None:
            raise ValueError("GBKTV3 requires both question and concept sequences.")

        q = q.long()
        c = c.long()
        r = r.float()

        if q.dim() != 2:
            raise ValueError(f"GBKTV3 expects q shape [B, T], but got {tuple(q.shape)}")

        batch_size, seq_len = q.shape
        device = q.device
        if seq_len < 2:
            empty = torch.zeros(batch_size, 0, device=device)
            return {
                "y": empty,
                "y_ball": empty,
                "y_concept_next": empty,
                "y_question_next": empty,
                "fusion_concept_weight": empty,
                "fusion_question_weight": empty,
                "theta": empty,
                "confidence": empty,
                "r_h_mean": empty,
                "r_d_mean": empty,
            }

        q_repr_all = self.get_question_repr(q, c)
        mu_d_all, r_d_all = self.question_difficulty_ball(q_repr_all)
        time_feat_all = self._build_time_features(it, ut, batch_size, seq_len, device)

        mu_h = self.mu_h0.unsqueeze(0).expand(batch_size, -1)
        r_h = self.initial_student_radius(batch_size, device)

        p_list, p_ball_list, p_concept_list, p_question_list = [], [], [], []
        fusion_concept_weight_list, fusion_question_weight_list = [], []
        theta_list, conf_list = [], []
        r_h_list, r_d_list = [], []
        mu_h_history_list = []

        for t in range(seq_len - 1):
            q_t = q[:, t]
            r_t = r[:, t]
            q_repr_t = q_repr_all[:, t, :]
            mu_d_t, r_d_t = mu_d_all[:, t, :], r_d_all[:, t, :]
            time_feat_t = time_feat_all[:, t, :]
            if t == 0:
                time_feat_t = torch.zeros_like(time_feat_t)

            mu_h_t, r_h_t = self._apply_time_forgetting(mu_h, r_h, time_feat_t)

            if c.dim() == 2:
                c_emb_t = self.get_avg_concept_emb(c[:, t])
            else:
                c_emb_t = self.get_avg_concept_emb(c[:, t:t + 1, :]).squeeze(1)

            mu_delta, r_delta, _ = self.knowledge_acquisition_ball_time(
                mu_h_t, r_h_t, mu_d_t, r_d_t, q_repr_t, r_t, time_feat_t
            )
            mu_h_next, r_h_next = self.update_state(
                mu_h_t, r_h_t, q_repr_t, r_t, mu_delta, r_delta, c_emb_t
            )

            if mu_h_history_list:
                mu_h_history = torch.stack(mu_h_history_list, dim=1)
                mu_h_next = self.ball_attn(mu_h_next, mu_h_history)

            valid_cur = (q_t >= 0).float().unsqueeze(-1)
            mu_h = valid_cur * mu_h_next + (1.0 - valid_cur) * mu_h
            r_h = valid_cur * r_h_next + (1.0 - valid_cur) * r_h
            mu_h_history_list.append(mu_h.detach())

            q_repr_next = q_repr_all[:, t + 1, :]
            mu_d_next, r_d_next = mu_d_all[:, t + 1, :], r_d_all[:, t + 1, :]

            if c.dim() == 2:
                c_emb_next = self.get_avg_concept_emb(c[:, t + 1])
                c_next = c[:, t + 1]
            else:
                c_emb_next = self.get_avg_concept_emb(c[:, t + 1:t + 2, :]).squeeze(1)
                c_next = c[:, t + 1, :]

            mu_h_readout, r_h_readout = self._condition_state_for_next_concept(
                mu_h, r_h, c_emb_next, mu_d_next, r_d_next
            )
            pred = self.ball_to_ball_predict(mu_h_readout, r_h_readout, mu_d_next, r_d_next)
            p_ball = pred["p_hat"]

            concept_logits = self.concept_next_head(torch.cat([mu_h_readout, c_emb_next], dim=-1))
            p_concept = self._concept_next_prob(concept_logits, c_next)

            question_logits = self.question_next_head(
                torch.cat([mu_h_readout, q_repr_next, mu_d_next, r_d_next], dim=-1)
            ).squeeze(-1)
            p_question = torch.sigmoid(question_logits)

            ball_logit = torch.logit(p_ball.clamp(1e-5, 1.0 - 1e-5))
            concept_logit = torch.logit(p_concept.clamp(1e-5, 1.0 - 1e-5))
            question_logit = torch.logit(p_question.clamp(1e-5, 1.0 - 1e-5))
            fusion_context = torch.cat(
                [mu_h_readout, q_repr_next, mu_d_next, r_d_next, time_feat_all[:, t + 1, :]],
                dim=-1,
            )
            fusion_scales = self.fusion_max_weights.to(device=fusion_context.device, dtype=fusion_context.dtype)
            fusion_weights = torch.sigmoid(self.fusion_gate(fusion_context)) * fusion_scales
            concept_w = fusion_weights[:, 0]
            question_w = fusion_weights[:, 1]
            fused_logit = self._fuse_logits(
                ball_logit,
                concept_logit,
                question_logit,
                concept_w,
                question_w,
            )
            p_fused = torch.sigmoid(fused_logit)

            p_list.append(p_fused)
            p_ball_list.append(p_ball)
            p_concept_list.append(p_concept)
            p_question_list.append(p_question)
            fusion_concept_weight_list.append(concept_w)
            fusion_question_weight_list.append(question_w)
            theta_list.append(pred["theta"])
            conf_list.append(pred["confidence"])
            r_h_list.append(r_h.mean(dim=-1))
            r_d_list.append(r_d_next.mean(dim=-1))

        return {
            "y": torch.stack(p_list, dim=1),
            "y_ball": torch.stack(p_ball_list, dim=1),
            "y_concept_next": torch.stack(p_concept_list, dim=1),
            "y_question_next": torch.stack(p_question_list, dim=1),
            "fusion_concept_weight": torch.stack(fusion_concept_weight_list, dim=1),
            "fusion_question_weight": torch.stack(fusion_question_weight_list, dim=1),
            "theta": torch.stack(theta_list, dim=1),
            "confidence": torch.stack(conf_list, dim=1),
            "r_h_mean": torch.stack(r_h_list, dim=1),
            "r_d_mean": torch.stack(r_d_list, dim=1),
        }
