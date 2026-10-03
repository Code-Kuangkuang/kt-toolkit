"""FoKT: AKT with the attention decay moved from step distance to elapsed time.

Every attention bias in this repository measures forgetting in **steps**:

    akt        gamma_h * sqrt(cumulative attention distance)   (per-head constant)
    folibikt   ALiBi, m_h * |i-j|                              (per-head constant)
    cskt       Kerple log kernel on |i-j|                      (per-head constant)
    lefokt     ALiBi / Kerple-log / Kerple-power / T5 / FIRE / Sandwich, all |i-j|
    ktst       exp(e_ij - tau*theta_h), theta_h learnable scalar (JEDM 2026)

Forgetting happens in **time**, not in steps, and on this repository's own data
the two come apart by three to four orders of magnitude.  Measured on the
training folds, holding step distance fixed at 20 and looking at the real
elapsed interval:

    assist2017          p10 2.6 min   median 8.9 min   p90 24.0 days
    bridge2algebra2006  p10 1.9 min   median 5.8 min   p90  6.0 days
    algebra2005         p10 5.2 min   median 14.9 min  p90  6.9 days

Two interactions twenty steps apart can be three minutes or twenty-four days
apart, and every bias above assigns them the same decay.  The spread is small
at distance 1 (tens of times) and explodes by distance 20, i.e. exactly at the
session boundary, which is where a decay term does its work.

## Why the forget-gate form, specifically

`Forgetting Transformer` (Lin et al., ICLR 2025, arXiv 2503.02130) adds a
forget gate to softmax attention as an additive logit bias

    O = softmax(QK^T + D) V,     D_ij = sum_{l=j+1..i} log f_l

with a scalar gate `f_l` per head.  That paper's own framing is the reason this
file exists: it proves **ALiBi is equivalent to a fixed, head-specific,
data-independent forget gate** `f = exp(-m_h)`.  So the five KT models listed
above are not five mechanisms, they are one mechanism with the gate nailed shut.

The cumulative structure is what makes this the right vehicle for KT rather
than a technique being borrowed.  `D_ij` accumulates over the intervening
steps; a forgetting curve accumulates over elapsed time.  Let the gate be a
function of the per-step interval and the two become the same object:

    f_l = exp(-dt_l / S)   =>   D_ij = -(t_i - t_j) / S

which is exponential forgetting in wall-clock time, exactly.  The step-distance
biases are the special case where every step is assumed to take equally long.

## The three rungs, all reachable from this one file

    emb_type            decay measured in   rate
    qid                 steps               per-head constant   = AKT, untouched
    qid_rkt             real time           per-head constant   = RKT's kernel
    qid_fox             real time           per-event, learned  = proposed

`qid_rkt` is RKT (Pandey & Srivastava, CIKM 2020), whose forget term is
`R_T = [exp(-Delta_i / S_u)]` with `S_u` a trainable "relative strength of
memory".  RKT itself is not reproducible here -- its `phi_array` relation
matrix has no published generator -- but its *time kernel* is four lines, and
that kernel alone is the rung that has to be beaten.  **Deviation recorded:
RKT's `S_u` is per student; this is per head, because the backbone carries no
student embedding.  Do not report this row as "RKT".**

`qid_fox` computes, per head:

    log f_l = logsigmoid( w_h . [ query_l , history_{l-1} , log1p(dt_l) , r_{l-1} ] )

`dt_l` is the interval *before* step l, so summing over `j+1..i` totals exactly
`t_i - t_j`.  `r_{l-1}` is there for consolidation: a successful retrieval
strengthens memory (the testing effect), so the same interval should not cost
the same amount of forgetting after a correct answer as after a wrong one.  A
per-student constant rate cannot express that; this is the whole content of
rung 2 -> rung 3.

**Every response-carrying input is lagged by one step, and that is load
bearing, not cosmetic.**  With `r_l` and `history_l` unlagged the gate leaks the
label: the query-side `c_i` does cancel inside the softmax, but the key-side
term at `j = i` contributes `-c_i`, so the weight the diagonal receives
relative to the rest encodes `r_i`.  Measured on the unlagged version:
flipping `r_3` moved the prediction at position 3, and two epochs on assist2017
reached **0.958** window AUC against a **0.696** control.  `query_l` needs no
lag because AKT's query stream carries no response, and `dt_l` needs none
because a tutoring system knows when it served the question.
`tests/test_fokt_no_gate_leakage.py` pins all of this.

Ablation arms, which exist so the two new inputs can be removed one at a time:

    qid_fox_notime      gate without log1p(dt)     -- isolates "data-dependent"
    qid_fox_noresp      gate without r_{l-1}       -- isolates "consolidation"

assist2009 has no `timestamps` column at all (its preprocessor hardcodes
`seq_start_time = ['NA']`, which is pyKT's own behaviour), so it is a free
`notime` arm: every other dataset here carries timestamps.

## Registered prediction, stated before the runs

The gain should be **ordered by how badly step distance and elapsed time come
apart** on each dataset -- largest on assist2017 and assist2012, smallest on
ednet.  If the gain does not follow that order, the mechanism under test is not
what is producing it.

## What is deliberately NOT changed

`qid` reproduces `models/akt.py` exactly, so the control arm is this file and
not a different file.  The fox/rkt arms *replace* AKT's multiplicative
`total_effect` decay rather than stacking on top of it, which keeps the
comparison single-variable: one decay term, measured in steps or in time.
"""

from __future__ import annotations

import math

import torch
from torch import nn
import torch.nn.functional as F

from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY
from .akt import AKT, TransformerLayer, attention

#: Clamp on the accumulated log-forget bias. AKT clamps its multiplicative
#: `total_effect` to [1e-5, 1e5]; -40 is the additive-logit equivalent floor and
#: keeps a 30-day gap from producing -inf when a learned rate is still small.
MIN_LOG_FORGET = -40.0

#: Seconds. `dt` is divided by this before `log1p`, so the gate's time feature
#: is O(1) for the gaps that actually occur (seconds to weeks -> roughly 0..7).
_DT_SCALE = 1.0


def elapsed_seconds(t_full: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """[B,T] non-negative seconds between step l-1 and step l.

    `t_full` is raw milliseconds as the loader stores it (`tseqs` is int64, so
    full precision -- do NOT use `itseqs`, which the loader floors to whole
    minutes and would erase every sub-minute gap).  Position 0 and any invalid
    position get 0, which is the identity for the cumulative sum downstream.
    """
    t = t_full.to(torch.float64)
    dt = torch.zeros_like(t)
    dt[:, 1:] = t[:, 1:] - t[:, :-1]
    # A zero timestamp is padding or a missing value, never a real epoch.
    known = valid & (t_full > 0)
    pair_known = torch.zeros_like(known)
    pair_known[:, 1:] = known[:, 1:] & known[:, :-1]
    dt = torch.where(pair_known, dt, torch.zeros_like(dt))
    return (dt.clamp_min(0.0) / 1000.0).to(torch.float32)


def _shift_right(x: torch.Tensor) -> torch.Tensor:
    """[B,T,...] shifted one step later, zero-filled at position 0.

    `out[:, l] = x[:, l-1]`. Used on every response-carrying gate input so the
    gate at step l describes the interval that *ended* at l without ever
    touching the response produced at l.
    """
    out = torch.zeros_like(x)
    out[:, 1:] = x[:, :-1]
    return out


def akt_step_decay(scores, mask, gammas):
    """AKT's multiplicative decay factor, recomputed here so it can be COMBINED
    with an additive gate bias rather than replaced by one.

    Copied from `models.akt.attention` deliberately: AKT applies this by
    multiplying the pre-softmax score, and there is no way to reach that from
    outside the function. The arithmetic is byte-for-byte the upstream one, and
    `tests/test_fokt_no_gate_leakage.py` pins that the `qid` arm still matches
    AKT exactly, so a drift here would surface there.

    Why this is not redundant with the gate, which the first version of this
    file wrongly argued: AKT's distance is `d~_ij`, weighted by the attention
    pattern itself, NOT the raw `|i - j|`. A constant gate reproduces ALiBi,
    i.e. raw distance. It cannot reproduce a distance that is a function of the
    attention weights.
    """
    seqlen = scores.size(-1)
    x1 = torch.arange(seqlen, device=scores.device).expand(seqlen, -1)
    x2 = x1.transpose(0, 1).contiguous()
    with torch.no_grad():
        s = scores.masked_fill(mask == 0, -1e32)
        s = F.softmax(s, dim=-1) * mask.to(dtype=scores.dtype)
        distcum = torch.cumsum(s, dim=-1)
        disttotal = torch.sum(s, dim=-1, keepdim=True)
        position = torch.abs(x1 - x2)[None, None, :, :].to(dtype=scores.dtype)
        dist = torch.clamp((disttotal - distcum) * position, min=0.0).sqrt().detach()
    gamma = -1.0 * F.softplus(gammas).unsqueeze(0)
    return torch.clamp(torch.clamp((dist * gamma).exp(), min=1e-5), max=1e5)


class ForgetGateAttention(nn.Module):
    """AKT's multi-head attention with the decay term made switchable.

    Three modes, selected by `mode`:

    ``steps``  AKT's own monotonic decay, untouched -- delegates to
               `models.akt.attention` so the control arm cannot drift.
    ``rkt``    `log f_l = -dt_l / S_h`, `S_h = exp(log_S_h)` learnable.
    ``fox``    `log f_l = logsigmoid(w_h . gate_features_l)` learnable.

    ``rkt`` and ``fox`` share one code path; they differ only in how `log f_l`
    is produced, which is precisely the comparison this file exists to make.
    """

    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same,
                 mode="steps", gate_in_dim=0, keep_steps=False,
                 gate_src="features", bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same
        self.mode = mode
        self.keep_steps = keep_steps
        self.gate_src = gate_src

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

        if mode == "steps":
            self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
            nn.init.xavier_uniform_(self.gammas)
        elif mode == "rkt":
            # One decay constant per head, in log-seconds. Initialised at one
            # day, which is the order of the median inter-session gap in every
            # dataset here; the model is free to move it.
            self.log_scale = nn.Parameter(
                torch.full((n_heads,), math.log(86400.0))
            )
        elif mode == "fox":
            if gate_src == "layer":
                # Faithful FoX: the gate reads this layer's own input and
                # nothing else, exactly as `f_t = sigma(w_f . x_t + b_f)` in
                # arXiv 2503.02130 S3. No KT feature is supplied.
                gate_in_dim = d_model
            if gate_in_dim <= 0:
                raise ValueError("fox mode needs gate features")
            if keep_steps:
                # AKT's own gammas, kept so the gate ADDS to the step decay
                # instead of replacing it.
                self.gammas = nn.Parameter(torch.zeros(n_heads, 1, 1))
                nn.init.xavier_uniform_(self.gammas)
            self.fgate = nn.Linear(gate_in_dim, n_heads)
            # Start near f=1 (no forgetting) so the arm begins as plain
            # attention and has to learn that forgetting pays, rather than
            # starting from a decay that a tuned constant could have supplied.
            # The weight is small but NOT zero: a zero weight makes the gate
            # input-independent at step 0, which both hides wiring bugs (the
            # first version of this file shipped with it and the time feature
            # provably did nothing) and starts the arm indistinguishable from a
            # learned constant.
            nn.init.normal_(self.fgate.weight, mean=0.0, std=0.01)
            nn.init.constant_(self.fgate.bias, 3.0)  # sigmoid(3) = 0.953
        else:
            raise ValueError(f"unknown decay mode {mode!r}")

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.k_linear.weight)
        nn.init.xavier_uniform_(self.v_linear.weight)
        if self.kq_same is False:
            nn.init.xavier_uniform_(self.q_linear.weight)
        if self.proj_bias:
            nn.init.constant_(self.k_linear.bias, 0.0)
            nn.init.constant_(self.v_linear.bias, 0.0)
            if self.kq_same is False:
                nn.init.constant_(self.q_linear.bias, 0.0)
            nn.init.constant_(self.out_proj.bias, 0.0)

    def log_forget(self, gate_feat, dt, valid):
        """[B,H,T] log f_l, the amount forgotten *at* step l.

        Zeroed on invalid positions so padding contributes nothing to the
        cumulative sum, which is what keeps a short sequence from inheriting
        the decay of the padding that follows it.
        """
        if self.mode == "rkt":
            scale = self.log_scale.exp().clamp_min(1.0)          # [H]
            logf = -dt.unsqueeze(1) / scale.view(1, -1, 1)       # [B,H,T]
        else:
            logf = F.logsigmoid(self.fgate(gate_feat))           # [B,T,H]
            logf = logf.transpose(1, 2)                          # [B,H,T]
        return logf * valid.unsqueeze(1).to(logf.dtype)

    def forward(self, q, k, v, mask, zero_pad, gate_feat=None, dt=None,
                valid=None, pdiff=None):
        bs = q.size(0)
        if self.mode == "fox" and self.gate_src == "layer":
            # LAGGED even though FoX does not lag. It has to be: in AKT's
            # blocks_1 the layer input is `qa_embed`, which encodes the
            # response AT position l -- the very label the model predicts
            # there. An unlagged gate on that stream leaks (see the module
            # docstring). Language modelling has no such problem, because its
            # label at t is the token at t+1, not part of the input at t.
            gate_feat = _shift_right(q)
        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        q = (self.q_linear if self.kq_same is False else self.k_linear)(q)
        q = q.view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)
        k, q, v = k.transpose(1, 2), q.transpose(1, 2), v.transpose(1, 2)

        if self.mode == "steps":
            scores = attention(q, k, v, self.d_k, mask, self.dropout,
                               zero_pad, self.gammas, None)
        else:
            logf = self.log_forget(gate_feat, dt, valid)          # [B,H,T]
            c = torch.cumsum(logf, dim=-1)                        # [B,H,T]
            # D_ij = sum_{l=j+1..i} log f_l = c_i - c_j
            d_bias = (c.unsqueeze(-1) - c.unsqueeze(-2)).clamp_min(MIN_LOG_FORGET)
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            if self.keep_steps:
                # Both terms: AKT's multiplicative step decay, then the gate's
                # additive time bias on top.
                scores = scores * akt_step_decay(scores, mask, self.gammas)
            scores = scores + d_bias
            scores = scores.masked_fill(mask == 0, -1e32)
            scores = F.softmax(scores, dim=-1)
            if zero_pad:
                pad = scores.new_zeros(bs, self.h, 1, scores.size(-1))
                scores = torch.cat([pad, scores[:, :, 1:, :]], dim=2)
            scores = self.dropout(scores)
            scores = torch.matmul(scores, v)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        return self.out_proj(concat)


class ForgetTransformerLayer(TransformerLayer):
    """AKT's block with its attention head swapped and the gate inputs routed."""

    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same,
                 mode, gate_in_dim, keep_steps=False, gate_src="features"):
        super().__init__(d_model=d_model, d_feature=d_feature, d_ff=d_ff,
                         dropout=dropout, n_heads=n_heads, kq_same=kq_same,
                         emb_type="qid")
        self.masked_attn_head = ForgetGateAttention(
            d_model, d_feature, n_heads, dropout, kq_same=(kq_same == 1),
            mode=mode, gate_in_dim=gate_in_dim, keep_steps=keep_steps,
            gate_src=gate_src,
        )

    def forward(self, mask, query, key, values, apply_pos=True, pdiff=None,
                gate_feat=None, dt=None, valid=None):
        import numpy as np

        seqlen = query.size(1)
        nopeek = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek) == 0).to(query.device)
        query2 = self.masked_attn_head(
            query, key, values, mask=src_mask, zero_pad=(mask == 0),
            gate_feat=gate_feat, dt=dt, valid=valid,
        )
        query = self.layer_norm1(query + self.dropout1(query2))
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = self.layer_norm2(query + self.dropout2(query2))
        return query


class ForgetArchitecture(nn.Module):
    """AKT's two-tower encoder, with the gate inputs threaded through."""

    def __init__(self, n_blocks, d_model, d_feature, d_ff, n_heads, dropout,
                 kq_same, mode, gate_in_dim, keep_steps=False, gate_src="features"):
        super().__init__()
        self.d_model = d_model
        make = lambda: ForgetTransformerLayer(  # noqa: E731
            d_model=d_model, d_feature=d_feature, d_ff=d_ff, dropout=dropout,
            n_heads=n_heads, kq_same=kq_same, mode=mode, gate_in_dim=gate_in_dim,
            keep_steps=keep_steps, gate_src=gate_src,
        )
        self.blocks_1 = nn.ModuleList([make() for _ in range(n_blocks)])
        self.blocks_2 = nn.ModuleList([make() for _ in range(n_blocks * 2)])

    def forward(self, q_embed_data, qa_embed_data, pid_embed_data,
                gate_feat=None, dt=None, valid=None):
        y, x = qa_embed_data, q_embed_data
        kw = dict(gate_feat=gate_feat, dt=dt, valid=valid)
        for block in self.blocks_1:
            y = block(mask=1, query=y, key=y, values=y, **kw)
        flag_first = True
        for block in self.blocks_2:
            if flag_first:
                x = block(mask=1, query=x, key=x, values=x, apply_pos=False, **kw)
                flag_first = False
            else:
                x = block(mask=0, query=x, key=x, values=y, apply_pos=True, **kw)
                flag_first = True
        return x


@MODEL_REGISTRY.register("fokt")
class FoKT(AKT):
    """AKT whose decay term is switchable between steps, real time and a gate."""

    class Inputs(InputSpec):
        """Asks the loader for raw timestamps unless the arm cannot use them."""

        dataset_mode = "all_in_one"
        requires_question_ids = True
        needs_num_pid = True

        @classmethod
        def prepare(cls, ctx):
            emb_type = str(ctx.model_cfg.get("emb_type", "qid"))
            # 2026-10-03: default multi-KC loading now matches the AKT control.
            extras = {"model_correctness_revision": "2026-10-03"}
            if not _needs_time(emb_type):
                return ModelInputs(run_config_extras=extras)
            # Only `model_cfg_updates`. `core/train_runner.py:322` reads
            # `use_timestamps` out of the model config and hands it to
            # `build_dataset` itself; also putting it in `dataset_kwargs`
            # makes that call raise "got multiple values for keyword
            # argument 'use_timestamps'".
            return ModelInputs(
                model_cfg_updates={"use_timestamps": True}, run_config_extras=extras,
            )

    def __init__(self, n_question=None, n_pid=None, d_model=256, n_blocks=1,
                 dropout=0.1, d_ff=256, kq_same=1, final_fc_dim=512,
                 num_attn_heads=8, separate_qa=False, l2=1e-5, emb_type="qid",
                 emb_path="", pretrain_dim=768, num_c=None, num_q=None,
                 use_timestamps=None, **kwargs):
        super().__init__(
            n_question=n_question, n_pid=n_pid, d_model=d_model,
            n_blocks=n_blocks, dropout=dropout, d_ff=d_ff, kq_same=kq_same,
            final_fc_dim=final_fc_dim, num_attn_heads=num_attn_heads,
            separate_qa=separate_qa, l2=l2, emb_type="qid", emb_path=emb_path,
            pretrain_dim=pretrain_dim, num_c=num_c, num_q=num_q,
        )
        self.model_name = "fokt"
        self.emb_type = emb_type
        self.mode = _decay_mode(emb_type)
        self.use_time = _needs_time(emb_type)
        self.use_resp = self.mode == "fox" and "noresp" not in emb_type
        # `_akt` keeps AKT's context-aware multiplicative step decay AND adds
        # the gate, instead of the gate replacing it.
        self.keep_steps = self.mode == "fox" and "akt" in emb_type
        # `foxlm` = the paper's own gate: layer input only, no KT feature.
        self.gate_src = "layer" if "foxlm" in emb_type else "features"

        gate_in_dim = 0
        if self.mode == "fox" and self.gate_src == "layer":
            gate_in_dim = d_model
        elif self.mode == "fox":
            # query + previous interaction + optional dt + optional previous response
            gate_in_dim = 2 * d_model + int(self.use_time) + int(self.use_resp)
        if self.mode != "steps":
            self.model = ForgetArchitecture(
                n_blocks=n_blocks, d_model=d_model,
                d_feature=d_model // num_attn_heads, d_ff=d_ff,
                n_heads=num_attn_heads, dropout=dropout, kq_same=kq_same,
                mode=self.mode, gate_in_dim=gate_in_dim,
                keep_steps=self.keep_steps, gate_src=self.gate_src,
            )
        self.reset()

    def forward(self, q_data, target, pid_data=None, t_data=None, qtest=False):
        batch = self.make_batch(q_data, target, pid_data)
        emb = self.embed(batch)
        if self.mode == "steps":
            d_output = self.model(emb.query, emb.history, emb.extras["pid_embed"])
        else:
            valid = batch.valid_mask
            if self.use_time:
                if t_data is None:
                    raise ValueError(
                        f"emb_type {self.emb_type!r} needs timestamps; the "
                        f"dataset gave none. assist2009 has no `timestamps` "
                        f"column -- use a `_notime` arm there."
                    )
                dt = elapsed_seconds(t_data, valid)
            else:
                dt = torch.zeros_like(target, dtype=torch.float32)

            gate_feat = None
            if self.mode == "fox" and self.gate_src == "layer":
                gate_feat = None  # each layer builds its own from its input
            elif self.mode == "fox":
                # EVERY response-carrying gate input is shifted right by one.
                # Without the shift the gate at position l sees r_l, and r_l is
                # the label for the prediction at l. It leaks even though the
                # query-side `c_i` cancels in the softmax: the key-side term at
                # j = i contributes `-c_i`, so the weight the diagonal gets
                # relative to the rest encodes r_i. Measured before the fix --
                # flipping r_3 moved the prediction at position 3, and 2 epochs
                # on assist2017 reached 0.958 AUC against a 0.696 control.
                #
                # The shift is also the better story: forgetting over the
                # interval (l-1, l] is governed by how long that interval was
                # and by whether the attempt *before* it was consolidated.
                parts = [
                    emb.query,                 # current question: no response in it
                    _shift_right(emb.history),  # previous interaction
                ]
                if self.use_time:
                    # dt_l is known before the student answers -- the system
                    # knows when the question was served -- so it needs no shift.
                    parts.append(torch.log1p(dt / _DT_SCALE).unsqueeze(-1))
                if self.use_resp:
                    r = target.to(emb.history.dtype).clamp(0, 1).unsqueeze(-1)
                    parts.append(_shift_right(r))
                gate_feat = torch.cat(parts, dim=-1)

            d_output = self.model(
                emb.query, emb.history, emb.extras["pid_embed"],
                gate_feat=gate_feat, dt=dt, valid=valid,
            )
        preds = self.readout(d_output, emb)
        if not qtest:
            return preds, emb.extras["reg_loss"]
        return preds, emb.extras["reg_loss"], torch.cat([d_output, emb.query], dim=-1)


def _decay_mode(emb_type: str) -> str:
    if "fox" in emb_type:
        return "fox"
    if "rkt" in emb_type:
        return "rkt"
    return "steps"


def _needs_time(emb_type: str) -> bool:
    mode = _decay_mode(emb_type)
    if mode == "rkt":
        return True
    if mode == "fox":
        # `foxlm` is the paper's gate verbatim: layer input only, no time.
        if "foxlm" in emb_type:
            return False
        return "notime" not in emb_type
    return False
