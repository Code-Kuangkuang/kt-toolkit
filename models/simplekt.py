import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from core.model_inputs import InputSpec, ModelInputs
from core.registry import MODEL_REGISTRY
from core.backbone import Embeddings, SeqBatch, infer_valid_mask
from .multi_concept import pool_concept_embeddings, pool_interaction_embeddings


class Dim:
    batch = 0
    seq = 1
    feature = 2


class LearnedShrinkageDifficulty(nn.Module):
    """Frozen difficulty whose shrinkage *strength* is fitted, not fixed at alpha.

    `compute_item_difficulty_logodds` shrinks an item's training-fold rate
    towards a target with weight `alpha / (n_i + alpha)` at a fixed alpha=10.
    That weight is a one-parameter member of a two-parameter family, because

        alpha / (n + alpha)  ==  sigmoid(log alpha - log n)

    exactly. This module learns `(slope, level)` in

        w_i = sigmoid(level - slope * log n_i)

    so `slope=1, level=log 10` -- the initialisation -- reproduces the fixed arm
    element for element at step 0. The fixed arm is therefore nested inside this
    one, which is what makes the comparison a test rather than two unrelated
    models.

    Why this is worth fitting at all: `alpha` is a count, so the same alpha is a
    different amount of shrinkage at different observations-per-item. At
    alpha=10 the weight is 75% on algebra2005 (3.3 responses per item) and 4.8%
    on assist2017 (196). `level` lets the data set that, and `slope` lets it set
    how fast shrinkage should fall off as evidence accumulates -- the closed
    form asserts a rate of exactly 1 in log n, which nothing has checked.

    Two parameters total, shared by every item. That matters for the premise
    this whole line rests on: a per-item parameter could recover the per-item
    counts by memorising them, which is the thing `qid_frozen` exists to rule
    out, while two numbers cannot. What is still frozen is each item's *rate*
    and its *target*; only the mixing weight moves.

    `slope` is left unconstrained. A fitted `slope <= 0` would mean shrinking
    harder as evidence accumulates, which is not a hypothesis anyone holds; it
    is reported rather than prevented, because seeing it would say the setup is
    broken and clamping it would hide that.

    Shape and call signature match the `nn.Embedding(num_pid + 1, 1)` it
    replaces, so the two `self.difficult_param(pid_data)` call sites are
    unchanged.
    """

    def __init__(self, rate, target, count, base_rate, init_alpha=10.0,
                 zero_unseen=True):
        super().__init__()
        self.zero_unseen = bool(zero_unseen)
        rate = torch.as_tensor(rate, dtype=torch.float32)
        target = torch.as_tensor(target, dtype=torch.float32)
        count = torch.as_tensor(count, dtype=torch.float32)
        self.register_buffer("rate", rate)
        self.register_buffer("target", target)
        self.register_buffer("seen", count > 0)
        # Clamped so an unseen item's log is finite; its weight is forced to 1
        # below, so the value never reaches the output.
        self.register_buffer("log_count", torch.log(count.clamp(min=1.0)))

        base_rate = float(min(max(base_rate, 1e-6), 1 - 1e-6))
        self.register_buffer(
            "base_logit", torch.tensor(math.log(base_rate / (1.0 - base_rate)))
        )
        # Shape [1] rather than scalar: SimpleKT.reset() calls p.size(0) on every
        # parameter, which raises on a 0-dim tensor.
        self.slope = nn.Parameter(torch.tensor([1.0]))
        self.level = nn.Parameter(torch.tensor([math.log(float(init_alpha))]))

    def weights(self):
        """`[num_q+1]` shrinkage weight per item, 1 where the item is unseen."""
        w = torch.sigmoid(self.level - self.slope * self.log_count)
        return torch.where(self.seen, w, torch.ones_like(w))

    def table(self):
        """`[num_q+1, 1]` standardised log-odds, differentiable in the weights.

        Mirrors `_standardised_logodds` step for step, including `unbiased=False`
        on the spread: numpy's `.std()` is ddof=0, and a ddof=1 spread here would
        make the two arms differ at initialisation by a factor no hypothesis
        asked for.
        """
        w = self.weights()
        smoothed = ((1.0 - w) * self.rate + w * self.target).clamp(1e-6, 1 - 1e-6)
        logodds = torch.log(smoothed / (1.0 - smoothed)) - self.base_logit
        seen_values = logodds[self.seen]
        scale = seen_values.std(unbiased=False)
        logodds = torch.where(scale > 0, logodds / scale.clamp(min=1e-12), logodds)
        # `_standardised_logodds` ends by zeroing unseen rows, and the two arms
        # have to agree on it. `_cold` turns that off: under a grouped target an
        # item with no training-fold responses then keeps its group's rate,
        # which is the case grouping exists for.
        if self.zero_unseen:
            logodds = torch.where(self.seen, logodds, torch.zeros_like(logodds))
        padding = torch.zeros_like(logodds)
        padding[:-1] = logodds[:-1]
        return padding.unsqueeze(-1)

    def forward(self, pid_data):
        return F.embedding(pid_data, self.table())


class ShrunkItemEmbedding(nn.Module):
    """A full-width item embedding shrunk towards its concepts, weight fitted.

    `qid_frozen*` established that shrinking an item's difficulty towards its
    concept group beats both a global target (+0.0027) and a size-matched random
    partition (+0.0023, t=5.72), and that the gain concentrates where shrinkage
    theory says it should: +0.0143 on items seen 1-4 times against +0.0012 on
    items seen 20+ times. All of that was measured on ONE number per item,
    because that is what `qid_frozen` freezes.

    CGMKT shrinks 256 numbers per item instead -- its question table is
    initialised from the mean concept vector and pulled back towards it by a
    graph hop every forward pass -- and scores 0.7942 against simpleKT's 0.7858
    with 12% FEWER parameters. That gap is the one structural difference between
    the two lines that has never been tested here, so this arm tests it:

        table = (1 - w_i) * own_i + w_i * target_i

        own_i     free [num_q+1, d], what `qid` already learns
        target_i  masked mean of this question's concept embeddings
        w_i       sigmoid(level - slope * log n_i), the same two-parameter
                  family models/simplekt.py::LearnedShrinkageDifficulty uses,
                  initialised at slope=1, level=log alpha so that it starts as
                  the closed form alpha/(n_i+alpha)

    Everything else stays as `qid`: the table still enters through
    `pid_embed * q_embed_diff`, so a difference against `qid` is the shrinkage
    and nothing else. An item with no training-fold responses gets `w=1` and so
    sits entirely on its concepts, which is the vector analogue of the `_cold`
    arm that was worth +0.064 on exactly those items.

    `use_text` adds the BGE concept vectors to the concept table. CGMKT's
    largest measured component was those vectors (+0.0025 overall, +0.0178 on
    unseen items), but it reaches them through the same low-data channel this
    shrinkage already uses, so the two may simply overlap. That is what the
    `_text` variant is for; it is not assumed to help.
    """

    def __init__(self, num_q, d, concept_map, item_count, text_embedding=None,
                 init_alpha=10.0):
        super().__init__()
        concept_map = torch.as_tensor(concept_map, dtype=torch.long)
        item_count = torch.as_tensor(item_count, dtype=torch.float32)
        num_c = int(concept_map.max().item()) + 1

        self.own = nn.Parameter(torch.empty(num_q + 1, d))
        nn.init.xavier_uniform_(self.own)
        self.concept = nn.Parameter(torch.empty(num_c, d))
        nn.init.xavier_uniform_(self.concept)

        self.use_text = text_embedding is not None
        if self.use_text:
            text = torch.as_tensor(text_embedding, dtype=torch.float32)
            if text.shape[0] != num_c:
                raise ValueError(
                    f"text_embedding has {text.shape[0]} concept rows against "
                    f"{num_c} concepts in the question-concept map."
                )
            # INITIALISE the concept table from the text, as CGMKT does, rather
            # than adding a projected copy to it every forward pass.
            #
            # The additive form was tried first and is broken: `self.text` is a
            # buffer and only the projection is learnable, so the target carried
            # whatever scale a fresh Linear happened to give it. Measured on
            # assist2009: projected-text norm 28.48 against a concept table of
            # 1.23, i.e. the free table contributed 4% of the target and five
            # epochs did not pull it back. Items with few responses sit almost
            # entirely on that target (w=0.89 at n=1), so they took the full hit
            # -- AUC on items seen 1-4 times fell 0.0249 (t=-9.08) while items
            # seen 20+ times, where w is near 0, were unaffected. That sign
            # pattern is the scale bug, not a finding about text.
            #
            # As an initialisation the scale is fixed once, matched to what
            # xavier would have produced, and training can move it.
            with torch.no_grad():
                projected = nn.Linear(text.shape[1], d)(text)
                projected = projected / projected.norm(dim=1, keepdim=True).clamp(min=1e-8)
                projected = projected * self.concept.norm(dim=1).mean()
                self.concept.copy_(projected)

        # `concept_map` is [num_q, max_concepts] with -1 padding and covers real
        # questions only; the padding row is appended so the table is
        # [num_q+1, ...] like every other item table here.
        padded = torch.full((num_q + 1, concept_map.shape[1]), -1, dtype=torch.long)
        padded[: concept_map.shape[0]] = concept_map
        self.register_buffer("qmap", padded)
        self.register_buffer("has_concept", (padded >= 0).any(dim=1))
        self.register_buffer("seen", item_count > 0)
        self.register_buffer("log_count", torch.log(item_count.clamp(min=1.0)))

        self.slope = nn.Parameter(torch.tensor([1.0]))
        self.level = nn.Parameter(torch.tensor([math.log(float(init_alpha))]))

    def table(self):
        # Text, when used, is baked into `self.concept` at construction; there
        # is no separate additive path. See __init__ for why.
        concept = self.concept

        mask = self.qmap >= 0                                     # [num_q+1, K]
        pooled = F.embedding(self.qmap.clamp(min=0), concept)     # [num_q+1, K, d]
        pooled = pooled * mask.unsqueeze(-1).to(pooled.dtype)
        counts = mask.sum(1, keepdim=True).clamp(min=1).to(pooled.dtype)
        target = pooled.sum(1) / counts                           # [num_q+1, d]

        w = torch.sigmoid(self.level - self.slope * self.log_count)
        # No responses of its own -> sit entirely on the concepts. A question
        # with no concepts either has nothing to borrow, so it keeps `own`.
        w = torch.where(self.seen, w, torch.ones_like(w))
        w = torch.where(self.has_concept, w, torch.zeros_like(w))
        return (1.0 - w).unsqueeze(-1) * self.own + w.unsqueeze(-1) * target

    def forward(self, pid_data):
        return F.embedding(pid_data, self.table())


@MODEL_REGISTRY.register("simplekt")
class SimpleKT(nn.Module):
    class Inputs(InputSpec):
        """Declares what this model needs.

        Derives nothing under every `emb_type` upstream ships.  The one exception
        is this repository's `qid_frozen`, which replaces the learned per-item
        Rasch scalar with the training folds' own difficulty counts; that is a
        derived feature and says so, so the run stamps
        `feature_fit_scope: train_folds` and cannot be put in a table beside a
        run stamped `none`.
        """
        supports_multi_concept = True

        needs_num_pid = True

        @classmethod
        def prepare(cls, ctx):
            emb_type = str(ctx.model_cfg.get("emb_type", "qid"))

            # `qid_shrunkvec[_text]`: the full-width analogue of the frozen
            # scalar arms. Needs the same counts they do, plus the
            # question-concept map that says what each item shrinks towards.
            if "shrunkvec" in emb_type:
                import json
                import os

                import numpy as _np

                from datasets.feature_utils import (
                    compute_item_difficulty_ingredients,
                )
                from models.kc_graph_utils import build_question_concept_map

                dpath = ctx.dataset_cfg["dpath"]
                num_q = int(ctx.dataset_cfg["num_q"])
                num_c = int(ctx.dataset_cfg["num_c"])
                with open(os.path.join(dpath, "keyid2idx.json"), encoding="utf-8") as fh:
                    max_concepts = int(json.load(fh)["max_concepts"])
                parts = compute_item_difficulty_ingredients(
                    dpath,
                    ctx.resolve_file(
                        ctx.quelevel_key("train_valid_file"), "train_valid_file"
                    ),
                    num_q=num_q,
                    folds=ctx.train_folds(),
                )
                kwargs = {
                    "item_count": parts["count"],
                    "concept_map": build_question_concept_map(dpath, num_q, max_concepts),
                }
                extras = {
                    "frozen_difficulty_alpha": float(
                        ctx.model_cfg.get("frozen_difficulty_alpha", 10.0)
                    ),
                    "frozen_difficulty_folds": ctx.train_folds(),
                    "frozen_difficulty_weight": "learned",
                    "item_shrinkage": "vector",
                    "item_shrinkage_text": "text" in emb_type,
                    "frozen_difficulty_seen_items": int((parts["count"] > 0).sum()),
                }
                if "text" in emb_type:
                    from models.kc_graph_utils import load_kc_text_embeddings

                    kwargs["text_embedding"] = load_kc_text_embeddings(
                        ctx.dataset_name, num_c, ctx.root_dir
                    )
                return ModelInputs(
                    model_kwargs=kwargs,
                    run_config_extras=extras,
                    # The counts are fitted on the training folds; the
                    # question-concept map is static metadata the loaders
                    # already hand every model through `cseqs`.
                    feature_fit_scope="train_folds",
                )

            if "frozen" not in emb_type:
                return ModelInputs()

            from datasets.feature_utils import compute_item_difficulty_logodds

            # What the frozen counts are shrunk *towards*. `qid_frozen` uses the
            # global rate, which is one number shared by every item and so adds
            # nothing an item can be told apart by; `_grouped` uses the item's
            # concept group minus itself; `_grouprand` keeps the group sizes and
            # reshuffles the members. The last one is the control, and it is the
            # arm that decides the question: without it, a win for `_grouped`
            # cannot be separated from "any shrinkage target other than a
            # constant helps". Compare MHAKT (TOIS 2026), whose RM-HG ablation
            # isolates exactly this and is never quoted in its own text.
            grouping = None
            if "grouprand" in emb_type:
                grouping = "random"
            elif "grouped" in emb_type:
                grouping = "concept"

            alpha = float(ctx.model_cfg.get("frozen_difficulty_alpha", 10.0))
            group_seed = int(ctx.model_cfg.get("frozen_difficulty_group_seed", 3407))

            # `_cold` reads group membership from the Q-matrix instead of from
            # the training rows, and stops resetting unseen items to 0. Without
            # it, an item no training row touches is a singleton group shrunk to
            # the global rate and then zeroed anyway -- so `_grouped` and
            # `_grouprand` are bit-identical on those items, which on assist2009
            # is 625 of 17,738. Those are the items with nothing of their own to
            # go on, i.e. the ones the grouping hypothesis is about.
            # `_alias` collapses concept ids that name the same skill before
            # groups are formed. Concept ids are not one-to-one with skills: on
            # assist2009 twenty of the 123 ids share a name with another id, so
            # grouping by id splits eight skills across twenty groups and two
            # items testing the same thing shrink towards different targets.
            # It is the cheap half of the question BGE embeddings answer in
            # CGMKT -- alias merging needs the names only, not the vectors --
            # so running it first says how much of that is plain de-duplication.
            merge_alias = "alias" in emb_type
            concept_alias = None
            if merge_alias:
                if grouping is None:
                    raise ValueError(
                        "emb_type contains 'alias' but no grouping; merging "
                        "concept ids changes nothing when every item shrinks "
                        "towards the same global rate. Use '_grouped_alias'."
                    )
                from models.kc_graph_utils import load_concept_alias_map

                concept_alias = load_concept_alias_map(
                    ctx.dataset_name, int(ctx.dataset_cfg["num_c"]), ctx.root_dir
                )

            cold_start = "cold" in emb_type
            if cold_start and grouping is None:
                raise ValueError(
                    "emb_type contains 'cold' but no grouping; shrinking an "
                    "unseen item towards the global rate and keeping it is the "
                    "global rate written twice, not a cold-start arm. Use "
                    "'_grouped_cold' or '_grouprand_cold'."
                )

            # `_learnw` keeps the same rows, folds and grouping but hands over
            # the unshrunk parts, so the shrinkage weight can be fitted inside
            # the model instead of being fixed at `alpha`. `alpha` survives as
            # the initialisation, which is what makes the fixed arm the t=0
            # state of this one.
            if "learnw" in emb_type:
                from datasets.feature_utils import (
                    compute_item_difficulty_ingredients,
                )

                parts = compute_item_difficulty_ingredients(
                    ctx.dataset_cfg["dpath"],
                    ctx.resolve_file(
                        ctx.quelevel_key("train_valid_file"), "train_valid_file"
                    ),
                    num_q=int(ctx.dataset_cfg["num_q"]),
                    folds=ctx.train_folds(),
                    grouping=grouping,
                    group_seed=group_seed,
                    cold_start=cold_start,
                    concept_alias=concept_alias,
                )
                extras = {
                    "frozen_difficulty_alpha": alpha,  # initialisation only
                    "frozen_difficulty_folds": ctx.train_folds(),
                    "frozen_difficulty_grouping": grouping or "global",
                    "frozen_difficulty_weight": "learned",
                    "frozen_difficulty_cold_start": cold_start,
                    "frozen_difficulty_merge_alias": merge_alias,
                    "frozen_difficulty_seen_items": int((parts["count"] > 0).sum()),
                }
                if grouping == "random":
                    extras["frozen_difficulty_group_seed"] = group_seed
                return ModelInputs(
                    model_kwargs={"difficulty_ingredients": parts},
                    run_config_extras=extras,
                    feature_fit_scope="train_folds",
                )

            table = compute_item_difficulty_logodds(
                ctx.dataset_cfg["dpath"],
                ctx.resolve_file(
                    ctx.quelevel_key("train_valid_file"), "train_valid_file"
                ),
                num_q=int(ctx.dataset_cfg["num_q"]),
                folds=ctx.train_folds(),
                alpha=alpha,
                grouping=grouping,
                group_seed=group_seed,
                cold_start=cold_start,
                concept_alias=concept_alias,
            )
            extras = {
                "frozen_difficulty_alpha": alpha,
                "frozen_difficulty_folds": ctx.train_folds(),
                "frozen_difficulty_nonzero": int((table != 0).sum()),
                "frozen_difficulty_grouping": grouping or "global",
                "frozen_difficulty_cold_start": cold_start,
                "frozen_difficulty_merge_alias": merge_alias,
            }
            if grouping == "random":
                # A random partition is only reproducible with its seed, and a
                # control whose result cannot be reproduced is not a control.
                extras["frozen_difficulty_group_seed"] = group_seed
            return ModelInputs(
                model_kwargs={"frozen_item_difficulty": table},
                run_config_extras=extras,
                feature_fit_scope="train_folds",
            )

    def __init__(
        self,
        num_c,
        num_q,
        num_pid=None,
        emb_size=None,
        num_blocks=None,
        dropout=0.2,
        d_ff=256,
        num_layers=2,
        num_attn_heads=8,
        seq_len=200,
        kq_same=1,
        final_fc_dim=512,
        final_fc_dim2=256,
        separate_qa=False,
        emb_type="qid",
        frozen_item_difficulty=None,
        difficulty_ingredients=None,
        frozen_difficulty_alpha=10.0,
        item_count=None,
        concept_map=None,
        text_embedding=None,
        **kwargs
    ):
        super().__init__()
        if emb_size is None:
            emb_size = kwargs.pop("d_model", 128)
        if num_blocks is None:
            num_blocks = kwargs.pop("n_blocks", 2)
        if num_pid is None:
            num_pid = num_q

        self.model_name = "simplekt"
        self.num_c = num_c
        self.num_q = num_q
        self.num_pid = num_pid
        self.emb_size = emb_size
        self.dropout = dropout
        self.kq_same = kq_same
        self.separate_qa = separate_qa
        self.emb_type = emb_type

        embed_l = emb_size

        # `qid_frozen` is scalar-width too: the whole point is that one number
        # per item is all the data supports, so it plugs into the same slot the
        # learned scalar occupies.
        self.frozen_difficulty = emb_type.find("frozen") != -1
        # `_learnw` keeps everything about the frozen arm except the one number
        # that says how hard to shrink; see LearnedShrinkageDifficulty.
        self.learned_shrinkage = emb_type.find("learnw") != -1
        # Full-width shrinkage keeps `qid`'s multiplicative combination, so it
        # must NOT take the scalar branch below.
        self.vector_shrinkage = emb_type.find("shrunkvec") != -1
        # `_additive` drops the multiplicative Rasch form; see the embedding
        # stage. Composable with `shrunkvec`, since they change different things.
        self.additive_item = emb_type.find("additive") != -1

        # Problem ID embedding (difficulty)
        if self.num_pid > 0:
            if self.additive_item and (
                emb_type.find("scalar") != -1 or (
                    self.frozen_difficulty and not self.vector_shrinkage)):
                raise ValueError(
                    "emb_type combines 'additive' with a scalar item table. "
                    "Adding a [.., 1] tensor to a [.., d] embedding broadcasts "
                    "it into a constant bias on every dimension, which is not "
                    "the additive item representation this arm is testing. Use "
                    "'qid_additive' or 'qid_shrunkvec_additive'."
                )
            if self.vector_shrinkage:
                if item_count is None or concept_map is None:
                    raise ValueError(
                        "emb_type contains 'shrunkvec' but item_count/concept_map "
                        "were not supplied. They come from SimpleKT.Inputs.prepare, "
                        "which only runs through the training runner."
                    )
                self.difficult_param = ShrunkItemEmbedding(
                    num_q=self.num_pid, d=embed_l,
                    concept_map=concept_map, item_count=item_count,
                    text_embedding=text_embedding,
                    init_alpha=frozen_difficulty_alpha,
                )
            elif emb_type.find("scalar") != -1 or self.frozen_difficulty:
                self.difficult_param = nn.Embedding(self.num_pid + 1, 1)
            else:
                self.difficult_param = nn.Embedding(self.num_pid + 1, embed_l)
            self.q_embed_diff = nn.Embedding(self.num_c + 1, embed_l)
            self.qa_embed_diff = nn.Embedding(2 * self.num_c + 1, embed_l)

        # Question embedding
        if emb_type.startswith("qid"):
            self.q_embed = nn.Embedding(self.num_c, embed_l)
            if self.separate_qa:
                self.qa_embed = nn.Embedding(2 * self.num_c + 1, embed_l)
            else:
                self.qa_embed = nn.Embedding(2, embed_l)

        # `_gru` swaps the sequence module and nothing else: same input layer,
        # same prediction head, same everything the embedding stage builds.
        # It is the last undecomposed difference between this model and CGMKT,
        # which runs one GRU over an additively fused dual branch and scores
        # 0.7942 where every input-layer mechanism transferred here
        # (vector shrinkage +0.0020, text +0.0018, additive form -0.0003) came
        # back non-significant.
        #
        # A GRU at d=256 has 394,752 parameters against the transformer's
        # 711,168, so a gain here cannot be capacity.
        self.gru_backbone = emb_type.find("gru") != -1
        if self.gru_backbone:
            self.model = nn.GRU(emb_size, emb_size, batch_first=True)
        else:
            self.model = SimpleKTArchitecture(
            num_c=num_c,
            num_blocks=num_blocks,
            n_heads=num_attn_heads,
            dropout=dropout,
            d_model=emb_size,
            d_feature=emb_size // num_attn_heads,
            d_ff=d_ff,
            kq_same=self.kq_same,
            seq_len=seq_len,
            # `_lnwrap` (see SimpleKTArchitecture) is the FlucKT-wrapper
            # transplant arm; every other emb_type leaves it off.
            lnwrap=emb_type.find("lnwrap") != -1,
            )

        self.out = nn.Sequential(
            nn.Linear(emb_size + embed_l, final_fc_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim, final_fc_dim2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(final_fc_dim2, 1),
        )

        self.reset()

        # After `reset`, which zeroes every `num_pid + 1`-row table and would
        # otherwise wipe this straight back out.
        if self.frozen_difficulty:
            if self.num_pid <= 0:
                raise ValueError(
                    "emb_type 'frozen' needs question ids; this dataset gave "
                    "num_pid=0."
                )
            if self.learned_shrinkage:
                if difficulty_ingredients is None:
                    raise ValueError(
                        "emb_type contains 'learnw' but no difficulty_ingredients "
                        "were supplied. They come from SimpleKT.Inputs.prepare, "
                        "which only runs through the training runner."
                    )
                counts = np.asarray(difficulty_ingredients["count"])
                if counts.shape[0] != self.num_pid + 1:
                    raise ValueError(
                        f"difficulty_ingredients have {counts.shape[0]} rows, "
                        f"expected num_pid + 1 = {self.num_pid + 1}."
                    )
                # Replaces the Embedding outright; the call signature matches, so
                # the two `self.difficult_param(pid_data)` sites do not change.
                self.difficult_param = LearnedShrinkageDifficulty(
                    rate=difficulty_ingredients["rate"],
                    target=difficulty_ingredients["target"],
                    count=counts,
                    base_rate=difficulty_ingredients["base_rate"],
                    init_alpha=frozen_difficulty_alpha,
                    zero_unseen=not difficulty_ingredients.get("cold_start", False),
                )
                return
            if frozen_item_difficulty is None:
                raise ValueError(
                    "emb_type contains 'frozen' but no frozen_item_difficulty "
                    "was supplied. It comes from SimpleKT.Inputs.prepare, which "
                    "only runs through the training runner."
                )
            table = torch.as_tensor(
                frozen_item_difficulty, dtype=torch.float32
            ).reshape(-1, 1)
            if table.shape[0] != self.num_pid + 1:
                raise ValueError(
                    f"frozen_item_difficulty has {table.shape[0]} rows, "
                    f"expected num_pid + 1 = {self.num_pid + 1}."
                )
            with torch.no_grad():
                self.difficult_param.weight.copy_(table)
            self.difficult_param.weight.requires_grad_(False)

    def reset(self):
        for p in self.parameters():
            if p.size(0) == self.num_pid + 1 and self.num_pid > 0:
                torch.nn.init.constant_(p, 0.0)

    def base_emb(self, q_data, target):
        q_embed_data = pool_concept_embeddings(
            self.q_embed, q_data, self.num_c
        )
        if self.separate_qa:
            qa_embed_data = pool_interaction_embeddings(
                self.qa_embed, q_data, target, self.num_c
            )
        else:
            qa_embed_data = self.qa_embed(target) + q_embed_data
        return q_embed_data, qa_embed_data

    # -- Stages. See core/backbone.py for why these exist. --

    def make_batch(
        self,
        qseqs,
        rseqs,
        cseqs,
        qshft,
        cshft,
        rshft,
        pidseqs=None,
        pidshft=None,
        valid_mask=None,
        **kwargs,
    ):
        """Stitch the loader's (current, shifted) pairs back into full sequences.

        The Rasch path needs question ids and raises without them, but only when
        it runs; a concept-only dataset reaches here with `pid_data=None` and
        that is legal, so the check stays in `embed` where the condition is.
        """
        q = qseqs.long() if qseqs is not None else None
        c = cseqs.long() if cseqs is not None else q
        if c is None:
            raise ValueError("SimpleKT requires concept sequences or question sequences.")
        r = rseqs.long()
        qshft = qshft.long() if qshft is not None else None
        cshft = cshft.long() if cshft is not None else qshft
        if cshft is None:
            raise ValueError("SimpleKT requires shifted concept or question sequences.")
        rshft = rshft.long()

        # Match pykt: concepts drive base embeddings, questions drive problem difficulty.
        q_data = torch.cat((c[:, 0:1], cshft), dim=1)
        target = torch.cat((r[:, 0:1], rshft), dim=1)

        if pidseqs is not None:
            pid = pidseqs.long()
            next_pid = pidshft.long() if pidshft is not None else qshft
        else:
            pid = q
            next_pid = qshft
        pid_data = (
            torch.cat((pid[:, 0:1], next_pid), dim=1)
            if pid is not None and next_pid is not None
            else None
        )

        if valid_mask is None:
            valid_mask = infer_valid_mask(q_data)
        return SeqBatch(
            concepts=q_data,
            responses=target,
            questions=pid_data,
            valid_mask=valid_mask,
        )

    def embed(self, batch):
        q_data, target, pid_data = batch.concepts, batch.responses, batch.questions

        # Base embeddings
        if self.emb_type.startswith("qid"):
            q_embed_data, qa_embed_data = self.base_emb(q_data, target)

        # Add problem difficulty
        if self.num_pid > 0 and self.emb_type.find("norasch") == -1:
            if pid_data is None:
                raise ValueError(
                    "SimpleKT Rasch difficulty requires qseqs/shft_qseqs "
                    "or pidseqs/shft_pidseqs. Set num_pid=0 for concept-only data."
                )
            if self.additive_item:
                # CGMKT adds its item representation outright -- `x_t = x_pro +
                # x_kc` -- where simpleKT multiplies it by a concept-dependent
                # direction. The Rasch form constrains the item term to a
                # scaling along `q_embed_diff`; the additive form does not, and
                # that is the last undecomposed difference between the two
                # models here (~0.0035 of CGMKT's 0.0084 margin remained after
                # its text embeddings, question branch and mastery module were
                # each measured).
                #
                # `q_embed_diff` is unused under this branch, so the arm has
                # 31,744 FEWER parameters than `qid` at d=256. A gain therefore
                # cannot be capacity.
                q_embed_data = q_embed_data + self.difficult_param(pid_data)
            elif self.emb_type.find("aktrasch") == -1:
                q_embed_diff_data = pool_concept_embeddings(
                    self.q_embed_diff, q_data, self.num_c
                )
                pid_embed_data = self.difficult_param(pid_data)
                q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data
            else:
                q_embed_diff_data = pool_concept_embeddings(
                    self.q_embed_diff, q_data, self.num_c
                )
                pid_embed_data = self.difficult_param(pid_data)
                q_embed_data = q_embed_data + pid_embed_data * q_embed_diff_data

                qa_embed_diff_data = self.qa_embed_diff(target)
                qa_embed_data = qa_embed_data + pid_embed_data * (
                    qa_embed_diff_data + q_embed_diff_data
                )

        return Embeddings(query=q_embed_data, history=qa_embed_data)

    def encode(self, emb):
        if self.gru_backbone:
            # SimpleKTArchitecture attends with `np.triu(..., k=0)` and
            # `zero_pad=True`, so position t sees interactions 0..t-1 and never
            # its own response. A GRU's h[t] covers 0..t inclusive, so it is
            # shifted by one and the first position zeroed to match exactly.
            # Without this the model reads the label it is predicting; the
            # contract test in tests/test_model_contracts.py checks for it.
            hidden, _ = self.model(emb.history)
            shifted = torch.zeros_like(hidden)
            shifted[:, 1:] = hidden[:, :-1]
            return shifted
        # Pass through transformer
        return self.model(emb.query, emb.history)

    def _logits(self, hidden, emb):
        concat_q = torch.cat([hidden, emb.query], dim=-1)
        return self.out(concat_q).squeeze(-1)

    def readout(self, hidden, emb):
        """Probabilities, so that every backbone's `readout` returns the same
        thing a plugin can hand straight back to a trainer."""
        return torch.sigmoid(self._logits(hidden, emb))

    def pack_output(self, preds, emb):
        """What `forward` hands back, so a plugged variant returns the same
        shape to the same trainer."""
        return preds

    def forward(
        self,
        qseqs,
        rseqs,
        cseqs,
        qshft,
        cshft,
        rshft,
        pidseqs=None,
        pidshft=None,
        return_features=False,
        **kwargs,
    ):
        emb = self.embed(
            self.make_batch(
                qseqs, rseqs, cseqs, qshft, cshft, rshft, pidseqs, pidshft
            )
        )
        d_output = self.encode(emb)
        output = self._logits(d_output, emb)

        preds = torch.sigmoid(output)
        if return_features:
            return {
                "preds": preds,
                "logits": output,
                "hidden": d_output,
                "question_embed": emb.query,
            }
        return preds


class SimpleKTArchitecture(nn.Module):
    def __init__(
        self,
        num_c,
        num_blocks,
        d_model,
        d_feature,
        d_ff,
        n_heads,
        dropout,
        kq_same,
        seq_len,
        lnwrap=False,
    ):
        super().__init__()
        self.d_model = d_model

        # `lnwrap` bolts FlucKT's FrequencyLayer wrapper -- LN(s + Dropout(s)),
        # i.e. the layer with its gate pinned at beta==1 -- onto both streams
        # of a *different* backbone. On assist2012 that pinned layer reproduced
        # full FlucKT's window AUC to -0.0001 (five folds), so the transplant
        # asks whether the +0.004-class margin is a portable regulariser or an
        # AKT-backbone artefact. Placement matches FlucKT exactly: after the
        # position embedding, before the first block, on both the query and the
        # interaction stream, same layer for both. Adds only the LayerNorm's
        # 2*d affine parameters (FlucKT's wrapper LN is affine too; 128 of
        # ~306k here), so any difference against `qid` cannot be capacity.
        self.lnwrap = lnwrap
        if lnwrap:
            self.ln_wrap_dropout = nn.Dropout(dropout)
            self.ln_wrap_norm = nn.LayerNorm(d_model)

        self.blocks_2 = nn.ModuleList(
            [
                TransformerLayer(
                    d_model=d_model,
                    d_feature=d_model // n_heads,
                    d_ff=d_ff,
                    dropout=dropout,
                    n_heads=n_heads,
                    kq_same=kq_same,
                )
                for _ in range(num_blocks)
            ]
        )
        self.position_emb = CosinePositionalEmbedding(d_model=self.d_model, max_len=seq_len)

    def forward(self, q_embed_data, qa_embed_data):
        seqlen = q_embed_data.size(1)

        q_posemb = self.position_emb(q_embed_data)
        q_embed_data = q_embed_data + q_posemb
        qa_posemb = self.position_emb(qa_embed_data)
        qa_embed_data = qa_embed_data + qa_posemb

        qa_pos_embed = qa_embed_data
        q_pos_embed = q_embed_data

        y = qa_pos_embed
        x = q_pos_embed

        if self.lnwrap:
            x = self.ln_wrap_norm(x + self.ln_wrap_dropout(x))
            y = self.ln_wrap_norm(y + self.ln_wrap_dropout(y))

        # Encoder
        for block in self.blocks_2:
            x = block(mask=0, query=x, key=x, values=y, apply_pos=True)
        return x


class TransformerLayer(nn.Module):
    def __init__(self, d_model, d_feature, d_ff, n_heads, dropout, kq_same):
        super().__init__()
        kq_same = kq_same == 1
        self.masked_attn_head = MultiHeadAttention(
            d_model, d_feature, n_heads, dropout, kq_same=kq_same
        )

        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, mask, query, key, values, apply_pos=True):
        seqlen = query.size(1)
        nopeek_mask = np.triu(np.ones((1, 1, seqlen, seqlen)), k=mask).astype("uint8")
        src_mask = (torch.from_numpy(nopeek_mask) == 0).to(query.device)
        if mask == 0:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=True)
        else:
            query2 = self.masked_attn_head(query, key, values, mask=src_mask, zero_pad=False)

        query = query + self.dropout1((query2))
        query = self.layer_norm1(query)
        if apply_pos:
            query2 = self.linear2(self.dropout(self.activation(self.linear1(query))))
            query = query + self.dropout2((query2))
            query = self.layer_norm2(query)
        return query


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_feature, n_heads, dropout, kq_same, bias=True):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_feature
        self.h = n_heads
        self.kq_same = kq_same

        self.v_linear = nn.Linear(d_model, d_model, bias=bias)
        self.k_linear = nn.Linear(d_model, d_model, bias=bias)
        if kq_same is False:
            self.q_linear = nn.Linear(d_model, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.proj_bias = bias
        self.out_proj = nn.Linear(d_model, d_model, bias=bias)

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

    def forward(self, q, k, v, mask, zero_pad):
        bs = q.size(0)

        k = self.k_linear(k).view(bs, -1, self.h, self.d_k)
        if self.kq_same is False:
            q = self.q_linear(q).view(bs, -1, self.h, self.d_k)
        else:
            q = self.k_linear(q).view(bs, -1, self.h, self.d_k)
        v = self.v_linear(v).view(bs, -1, self.h, self.d_k)

        k = k.transpose(1, 2)
        q = q.transpose(1, 2)
        v = v.transpose(1, 2)
        scores = attention(q, k, v, self.d_k, mask, self.dropout, zero_pad)

        concat = scores.transpose(1, 2).contiguous().view(bs, -1, self.d_model)
        output = self.out_proj(concat)
        return output


def attention(q, k, v, d_k, mask, dropout, zero_pad):
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    bs, head, seqlen = scores.size(0), scores.size(1), scores.size(2)

    scores.masked_fill_(mask == 0, -1e32)
    scores = F.softmax(scores, dim=-1)
    if zero_pad:
        pad_zero = torch.zeros(bs, head, 1, seqlen).to(scores.device)
        scores = torch.cat([pad_zero, scores[:, :, 1:, :]], dim=2)
    scores = dropout(scores)
    output = torch.matmul(scores, v)
    return output


class CosinePositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = 0.1 * torch.randn(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.weight = nn.Parameter(pe, requires_grad=False)

    def forward(self, x):
        return self.weight[:, : x.size(Dim.seq), :]
