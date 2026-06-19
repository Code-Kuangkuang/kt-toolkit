import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from core.registry import MODEL_REGISTRY
from models.gbktv3 import GBKTV3


def _softplus_inverse(value):
    value = max(float(value), 1e-8)
    if value > 20.0:
        return value
    return math.log(math.expm1(value))


class DistanceAwareHistoricalBallAttention(nn.Module):
    """Historical center attention with monotonic sequence/time recency bias."""

    def __init__(
        self,
        d_h,
        n_heads=4,
        dropout=0.1,
        sequence_distance_init=0.005,
        time_distance_init=0.0,
        base_attn=None,
        base_norm=None,
    ):
        super().__init__()
        self.attn = base_attn or nn.MultiheadAttention(
            embed_dim=d_h,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm = base_norm or nn.LayerNorm(d_h)
        self.raw_sequence_gamma = nn.Parameter(
            torch.tensor(_softplus_inverse(sequence_distance_init), dtype=torch.float32)
        )
        self.raw_time_beta = nn.Parameter(
            torch.tensor(_softplus_inverse(time_distance_init), dtype=torch.float32)
        )

    def _distance_bias(self, batch_size, seq_distance, time_distance, device, dtype):
        bias = None

        if seq_distance is not None:
            seq_distance = seq_distance.to(device=device, dtype=dtype).clamp(min=0.0)
            if seq_distance.dim() == 1:
                seq_distance = seq_distance.unsqueeze(0).expand(batch_size, -1)
            gamma = F.softplus(self.raw_sequence_gamma).to(dtype=dtype)
            bias = -gamma * seq_distance

        if time_distance is not None:
            time_distance = time_distance.to(device=device, dtype=dtype).clamp(min=0.0)
            if time_distance.dim() == 1:
                time_distance = time_distance.unsqueeze(0).expand(batch_size, -1)
            beta = F.softplus(self.raw_time_beta).to(dtype=dtype)
            time_bias = -beta * time_distance
            bias = time_bias if bias is None else bias + time_bias

        if bias is None:
            return None
        return bias.unsqueeze(1).repeat_interleave(self.attn.num_heads, dim=0)

    def forward(self, mu_h_current, mu_h_history, seq_distance=None, time_distance=None):
        if mu_h_history.size(1) == 0:
            return mu_h_current

        query = mu_h_current.unsqueeze(1)
        attn_mask = self._distance_bias(
            batch_size=mu_h_history.size(0),
            seq_distance=seq_distance,
            time_distance=time_distance,
            device=mu_h_history.device,
            dtype=mu_h_history.dtype,
        )
        attn_out, _ = self.attn(
            query=query,
            key=mu_h_history,
            value=mu_h_history,
            attn_mask=attn_mask,
        )
        enhanced = self.norm(query + attn_out)
        return enhanced.squeeze(1)


@MODEL_REGISTRY.register("gbktv4")
class GBKTV4(GBKTV3):
    """GBKT v4: cleaned GBKTV3-parity path.

    The current v4 keeps the GBKTV3 time-aware ball backbone, concept readout,
    history selection, and dynamic concept fusion. Item-level difficulty
    calibration and radius-aware fallback fusion were removed after ablation
    showed they did not improve ASSIST2009 fold-0 performance.
    """

    def __init__(
        self,
        *args,
        history_mode="ema",
        use_history_attention=None,
        history_window=0,
        history_stride=1,
        history_ema_alpha=0.90,
        use_sequence_distance_attention=True,
        use_time_distance_attention=False,
        sequence_distance_init=0.005,
        time_distance_init=0.0,
        use_scalar_item_difficulty=False,
        scalar_item_difficulty_init=0.0,
        use_time_aware_kab=False,
        use_time_forgetting=False,
        use_question_branch=False,
        use_dynamic_fusion=True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.model_name = "gbktv4"
        if use_history_attention is not None:
            history_mode = "attention" if bool(use_history_attention) else "none"
        self.history_mode = str(history_mode).lower()
        self.use_history_attention = self.history_mode == "attention"
        self.history_window = max(int(history_window), 0)
        self.history_stride = max(int(history_stride), 1)
        self.history_ema_alpha = float(history_ema_alpha)
        self.use_sequence_distance_attention = bool(use_sequence_distance_attention)
        self.use_time_distance_attention = bool(use_time_distance_attention)
        self.use_scalar_item_difficulty = bool(use_scalar_item_difficulty)
        self.use_time_aware_kab = bool(use_time_aware_kab)
        self.use_time_forgetting = bool(use_time_forgetting)
        self.use_question_branch = bool(use_question_branch)
        self.use_dynamic_fusion = bool(use_dynamic_fusion)
        base_ball_attn = self.ball_attn
        attn_heads = getattr(getattr(base_ball_attn, "attn", None), "num_heads", 4)
        attn_dropout = getattr(getattr(base_ball_attn, "attn", None), "dropout", 0.0)
        self.ball_attn = DistanceAwareHistoricalBallAttention(
            self.d_h,
            n_heads=attn_heads,
            dropout=float(attn_dropout),
            sequence_distance_init=sequence_distance_init,
            time_distance_init=time_distance_init,
            base_attn=getattr(base_ball_attn, "attn", None),
            base_norm=getattr(base_ball_attn, "norm", None),
        )
        self.ema_history_gate = nn.Sequential(
            nn.Linear(2 * self.d_h + self.emb_size, self.d_h),
            nn.GELU(),
            nn.Linear(self.d_h, self.d_h),
        )
        if self.use_scalar_item_difficulty:
            self.item_difficulty = nn.Embedding(
                self.num_q + 1,
                1,
                padding_idx=self.num_q,
            )
            nn.init.constant_(self.item_difficulty.weight, float(scalar_item_difficulty_init))
            with torch.no_grad():
                self.item_difficulty.weight[self.num_q].zero_()
        else:
            self.item_difficulty = None

    def _init_ema_history(self, mu_h):
        return torch.zeros_like(mu_h)

    def _update_ema_history(self, history_state, mu_h):
        alpha = min(max(self.history_ema_alpha, 0.0), 0.999)
        return alpha * history_state + (1.0 - alpha) * mu_h.detach()

    def _apply_ema_history(self, mu_h_next, history_state, c_emb_t):
        gate_input = torch.cat([mu_h_next, history_state, c_emb_t], dim=-1)
        gate = torch.sigmoid(self.ema_history_gate(gate_input))
        return (1.0 - gate) * mu_h_next + gate * history_state

    def _select_history(self, history):
        if not history:
            return None, None

        selected = history
        if self.history_window > 0:
            selected = selected[-self.history_window:]
        if self.history_stride > 1:
            selected = selected[::self.history_stride]
        if not selected:
            return None, None

        states = []
        steps = []
        for idx, item in enumerate(selected):
            if isinstance(item, tuple):
                state, step = item
            else:
                state, step = item, idx
            states.append(state)
            steps.append(int(step))
        return torch.stack(states, dim=1), steps

    @staticmethod
    def _build_sequence_distance(current_step, history_steps, device, dtype):
        if current_step is None or history_steps is None:
            return None
        distances = [max(int(current_step) - int(step), 1) for step in history_steps]
        return torch.tensor(distances, device=device, dtype=dtype)

    @staticmethod
    def _build_time_distance(current_step, history_steps, time_feat_all):
        if current_step is None or history_steps is None or time_feat_all is None:
            return None
        if time_feat_all.size(1) <= 1:
            return None

        interval = time_feat_all[:, :, 0]
        distances = []
        for step in history_steps:
            start = max(min(int(step) + 1, int(current_step) + 1), 0)
            end = max(min(int(current_step) + 1, interval.size(1)), 0)
            if start >= end:
                distances.append(
                    torch.zeros(interval.size(0), device=interval.device, dtype=interval.dtype)
                )
            else:
                distances.append(interval[:, start:end].sum(dim=1))
        if not distances:
            return None
        return torch.stack(distances, dim=1)

    def _maybe_apply_history_attention(
        self,
        mu_h_next,
        history,
        current_step=None,
        time_feat_all=None,
    ):
        if self.history_mode != "attention":
            return mu_h_next

        mu_h_history, history_steps = self._select_history(history)
        if mu_h_history is None:
            return mu_h_next

        seq_distance = None
        if self.use_sequence_distance_attention:
            seq_distance = self._build_sequence_distance(
                current_step,
                history_steps,
                mu_h_history.device,
                mu_h_history.dtype,
            )

        time_distance = None
        if self.use_time_distance_attention:
            time_distance = self._build_time_distance(current_step, history_steps, time_feat_all)

        return self.ball_attn(
            mu_h_next,
            mu_h_history,
            seq_distance=seq_distance,
            time_distance=time_distance,
        )

    def _get_scalar_item_difficulty(self, q_next, dtype):
        if self.item_difficulty is None:
            return None
        qidx = self._sanitize_question_ids(q_next)
        return self.item_difficulty(qidx).squeeze(-1).to(dtype=dtype)

    def ball_to_ball_predict(self, mu_h, r_h, mu_d, r_d, q_next=None):
        pred = super().ball_to_ball_predict(mu_h, r_h, mu_d, r_d)
        if not self.use_scalar_item_difficulty or q_next is None:
            return pred

        item_difficulty = self._get_scalar_item_difficulty(q_next, pred["p_hat"].dtype)
        if item_difficulty is None:
            return pred

        base_logit = torch.logit(pred["p_hat"].clamp(1e-5, 1.0 - 1e-5))
        pred["b"] = pred["b"] + item_difficulty
        pred["p_hat"] = torch.sigmoid(base_logit - pred["a"] * item_difficulty)
        pred["item_difficulty"] = item_difficulty
        return pred

    def forward(self, q, c, r, it=None, ut=None):
        if q is None or c is None:
            raise ValueError("GBKTV4 requires both question and concept sequences.")

        q = q.long()
        c = c.long()
        r = r.float()

        if q.dim() != 2:
            raise ValueError(f"GBKTV4 expects q shape [B, T], but got {tuple(q.shape)}")

        batch_size, seq_len = q.shape
        device = q.device
        if seq_len < 2:
            empty = torch.zeros(batch_size, 0, device=device)
            return {
                "y": empty,
                "y_ball": empty,
                "y_concept_next": empty,
                "fusion_concept_weight": empty,
                "theta": empty,
                "confidence": empty,
                "r_h_mean": empty,
                "r_d_mean": empty,
            }

        q_repr_all = self.get_question_repr(q, c)
        mu_d_all, r_d_all = self.question_difficulty_ball(q_repr_all)
        if self.use_time_aware_kab or self.use_time_forgetting:
            time_feat_all = self._build_time_features(it, ut, batch_size, seq_len, device)
        else:
            time_feat_all = torch.zeros(batch_size, seq_len, 2, device=device)

        mu_h = self.mu_h0.unsqueeze(0).expand(batch_size, -1)
        r_h = self.initial_student_radius(batch_size, device)
        ema_history = self._init_ema_history(mu_h)

        p_list, p_ball_list, p_concept_list = [], [], []
        p_question_list = [] if self.use_question_branch else None
        fusion_concept_weight_list = []
        fusion_question_weight_list = [] if self.use_question_branch else None
        theta_list, conf_list = [], []
        r_h_list, r_d_list = [], []
        item_difficulty_list = [] if self.use_scalar_item_difficulty else None
        mu_h_history_list = []

        for t in range(seq_len - 1):
            q_t = q[:, t]
            r_t = r[:, t]
            q_repr_t = q_repr_all[:, t, :]
            mu_d_t, r_d_t = mu_d_all[:, t, :], r_d_all[:, t, :]
            time_feat_t = time_feat_all[:, t, :]
            if t == 0:
                time_feat_t = torch.zeros_like(time_feat_t)

            if self.use_time_forgetting:
                mu_h_t, r_h_t = self._apply_time_forgetting(mu_h, r_h, time_feat_t)
            else:
                mu_h_t, r_h_t = mu_h, r_h

            if c.dim() == 2:
                c_emb_t = self.get_avg_concept_emb(c[:, t])
            else:
                c_emb_t = self.get_avg_concept_emb(c[:, t:t + 1, :]).squeeze(1)

            if self.use_time_aware_kab:
                mu_delta, r_delta, _ = self.knowledge_acquisition_ball_time(
                    mu_h_t, r_h_t, mu_d_t, r_d_t, q_repr_t, r_t, time_feat_t
                )
            else:
                mu_delta, r_delta, _ = self.knowledge_acquisition_ball(
                    mu_h_t, r_h_t, mu_d_t, r_d_t, q_repr_t, r_t
                )
            mu_h_next, r_h_next = self.update_state(
                mu_h_t, r_h_t, q_repr_t, r_t, mu_delta, r_delta, c_emb_t
            )
            if self.history_mode == "ema":
                mu_h_next = self._apply_ema_history(mu_h_next, ema_history, c_emb_t)
            else:
                mu_h_next = self._maybe_apply_history_attention(
                    mu_h_next,
                    mu_h_history_list,
                    current_step=t,
                    time_feat_all=time_feat_all,
                )

            valid_cur = (q_t >= 0).float().unsqueeze(-1)
            mu_h = valid_cur * mu_h_next + (1.0 - valid_cur) * mu_h
            r_h = valid_cur * r_h_next + (1.0 - valid_cur) * r_h

            if self.history_mode == "ema":
                ema_candidate = self._update_ema_history(ema_history, mu_h)
                ema_history = valid_cur * ema_candidate + (1.0 - valid_cur) * ema_history
            elif self.history_mode == "attention":
                mu_h_history_list.append((mu_h.detach(), t))
                if self.history_window > 0 and len(mu_h_history_list) > self.history_window:
                    del mu_h_history_list[:-self.history_window]

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
            pred = self.ball_to_ball_predict(
                mu_h_readout,
                r_h_readout,
                mu_d_next,
                r_d_next,
                q_next=q[:, t + 1],
            )
            p_ball = pred["p_hat"]

            concept_logits = self.concept_next_head(torch.cat([mu_h_readout, c_emb_next], dim=-1))
            p_concept = self._concept_next_prob(concept_logits, c_next)

            ball_logit = torch.logit(p_ball.clamp(1e-5, 1.0 - 1e-5))
            concept_logit = torch.logit(p_concept.clamp(1e-5, 1.0 - 1e-5))
            if self.use_question_branch:
                question_logits = self.question_next_head(
                    torch.cat([mu_h_readout, q_repr_next, mu_d_next, r_d_next], dim=-1)
                ).squeeze(-1)
                p_question = torch.sigmoid(question_logits)
            else:
                p_question = None

            if self.use_dynamic_fusion:
                question_logit = torch.logit(
                    p_question.clamp(1e-5, 1.0 - 1e-5)
                    if p_question is not None
                    else p_ball.clamp(1e-5, 1.0 - 1e-5)
                )
                fusion_context = torch.cat(
                    [mu_h_readout, q_repr_next, mu_d_next, r_d_next, time_feat_all[:, t + 1, :]],
                    dim=-1,
                )
                fusion_scales = self.fusion_max_weights.to(
                    device=fusion_context.device, dtype=fusion_context.dtype
                )
                fusion_weights = torch.sigmoid(self.fusion_gate(fusion_context)) * fusion_scales
                concept_w = fusion_weights[:, 0]
                question_w = fusion_weights[:, 1] if self.use_question_branch else torch.zeros_like(concept_w)
                fused_logit = self._fuse_logits(
                    ball_logit,
                    concept_logit,
                    question_logit,
                    concept_w,
                    question_w,
                )
            else:
                concept_w = torch.sigmoid(self.fusion_logit_weight) * self.max_concept_fusion_weight
                concept_w = concept_w.expand_as(ball_logit)
                question_w = torch.zeros_like(concept_w)
                fused_logit = self._fuse_logits(
                    ball_logit,
                    concept_logit,
                    ball_logit,
                    concept_w,
                    question_w,
                )
            p_fused = torch.sigmoid(fused_logit)

            p_list.append(p_fused)
            p_ball_list.append(p_ball)
            p_concept_list.append(p_concept)
            if p_question_list is not None and p_question is not None:
                p_question_list.append(p_question)
            fusion_concept_weight_list.append(concept_w)
            if fusion_question_weight_list is not None and question_w is not None:
                fusion_question_weight_list.append(question_w)
            theta_list.append(pred["theta"])
            conf_list.append(pred["confidence"])
            r_h_list.append(r_h.mean(dim=-1))
            r_d_list.append(r_d_next.mean(dim=-1))
            if item_difficulty_list is not None:
                item_difficulty = pred.get("item_difficulty")
                if item_difficulty is None:
                    item_difficulty = torch.zeros_like(p_ball)
                item_difficulty_list.append(item_difficulty)

        outputs = {
            "y": torch.stack(p_list, dim=1),
            "y_ball": torch.stack(p_ball_list, dim=1),
            "y_concept_next": torch.stack(p_concept_list, dim=1),
            "fusion_concept_weight": torch.stack(fusion_concept_weight_list, dim=1),
            "theta": torch.stack(theta_list, dim=1),
            "confidence": torch.stack(conf_list, dim=1),
            "r_h_mean": torch.stack(r_h_list, dim=1),
            "r_d_mean": torch.stack(r_d_list, dim=1),
        }
        if p_question_list:
            outputs["y_question_next"] = torch.stack(p_question_list, dim=1)
        if fusion_question_weight_list:
            outputs["fusion_question_weight"] = torch.stack(fusion_question_weight_list, dim=1)
        if item_difficulty_list:
            outputs["item_difficulty"] = torch.stack(item_difficulty_list, dim=1)
        return outputs
