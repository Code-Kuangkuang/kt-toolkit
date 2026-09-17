"""Every registered model must satisfy the same basic contract.

The repository has 38 registered models and, before this file, hand-written
tests for about a third of them. A model added without a test could be broken in
ways nothing catches until someone runs a full training job: an unregistered
trainer, a constructor that no longer matches its config block, predictions that
do not line up with `smasks`, NaN gradients, or a forward pass that lets a
future response influence a past prediction.

This walks the registry instead of naming models, so a new model is covered the
moment it is registered.

What is checked, per model:

  1. model and trainer are both registered under the same key
  2. a hyperparameter block exists in configs/kt_config.json
  3. the model constructs from that block
  4. a small batch runs forward and backward
  5. predictions and targets agree with each other and with `smasks.sum()`
  6. loss, predictions and gradients are all finite
  7. flipping a future response leaves earlier predictions untouched

Checks 3-7 need a batch, and a few models need inputs this harness cannot
synthesise faithfully (a precomputed graph, a booster embedding). Those are
reported as explicit skips with a reason rather than passing silently -- the
skip list is the honest statement of what is still uncovered.
"""

import json
import os
import sys
import unittest
from pathlib import Path

# Must precede `import torch`: cuBLAS reads this when it creates its handle, and
# without it bmm on CUDA is non-deterministic at the 1e-3 level -- large enough
# to swamp the causality check below. core/run_support.py::set_seed sets the same
# value in the real training path.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.trainers  # noqa: F401  -- registers trainers
import models  # noqa: F401  -- registers models
from core.factory import build_model, build_trainer
from core.model_inputs import spec_for
from core.registry import MODEL_REGISTRY, TRAINER_REGISTRY
from core.train_runner import MODEL_NAME_ALIASES, resolve_dataset_mode
from datasets.init_dataset import resolve_concept_mode

KT_CONFIG = json.loads((ROOT / "configs" / "kt_config.json").read_text(encoding="utf-8"))


def canonical_models():
    """Registered keys that name a distinct model, not an alias of one.

    Three classes are registered twice -- DKTForget as `dkt-forget`, HQAFKT as
    `hqaf_kt`, LEFOKT_AKT as `lefokt` -- so the registry reports 38 entries for
    35 models. `train_one_fold` resolves through MODEL_NAME_ALIASES before it
    ever reaches the registry, so those second registrations are unreachable;
    they are excluded here rather than tested twice.
    """
    return sorted(k for k in MODEL_REGISTRY.get_all() if k not in MODEL_NAME_ALIASES)

BATCH, SEQ_LEN, NUM_C, NUM_Q, MAX_CONCEPTS = 4, 12, 7, 11, 3

# A batch is one position shorter than the configured sequence length: the
# dataset builds inputs from cur[:-1] and shifted targets from cur[1:], so
# seq_len 200 yields 199 positions. Models that concatenate a start token back on
# reject a batch that ignores this -- keenkt raises outright, hd_simplekt and
# rekt fail on a size mismatch.
SEQ_POSITIONS = SEQ_LEN - 1

# Thirteen model files hold a module-level
# `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
# and use it instead of the device passed in, so on a CUDA machine they cannot
# run on CPU at all. The harness therefore uses the same device the runner would
# rather than forcing CPU; see "Known Structural Debt" in docs/architecture.md.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Models whose forward needs an artefact this harness does not build. Each entry
# is a reason, not a mute: the point is that the gap stays visible.
NEEDS_REAL_ARTEFACTS = {
    "gkt": "needs a concept-transition graph built from real sequences",
    "dgekt": "needs a hypergraph and transition matrices built from real sequences",
    "dkt_pebg": "needs a pretrained PEBG booster embedding file",
    "hawkes": "runs in double precision with its own init, applied by train_runner",
}

# Violations recorded rather than skipped. The tests assert that a recorded
# violation still fails, so an entry must be deleted when its model is fixed and
# a model that starts violating without an entry breaks the build.
#
# All three held IEKT until 2026-09-17: it scored past smasks, returned logits
# where the evaluator expects probabilities, and sampled its policy during eval.
# Empty is the intended steady state, not a sign the checks are unused.
KNOWN_RANGE_VIOLATIONS = {}

KNOWN_NONDETERMINISM = {}

KNOWN_ALIGNMENT_VIOLATIONS = {}

# Specs whose `prepare` reads the dataset from disk -- difficulty maps, gap
# tables, time vocabularies, graphs. Calling them against synthetic sizes would
# only test whether a stub context can imitate a data directory, so the values
# they would compute are supplied below instead and the real code path is
# covered by a training run.
FITS_FROM_REAL_DATA = {
    "dimkt", "hqaf", "lpkt", "hdkt", "dkt_forget", "dgekt", "gkt", "dkt_pebg",
}

# What those specs would have produced, at sizes this harness can use. Values are
# arbitrary but must exceed the ids the synthetic batch generates.
COMPUTED_CONSTRUCTOR_ARGS = {
    "dkt_forget": {"num_rgap": 8, "num_sgap": 8, "num_pcount": 8},
    "lpkt": {"num_at": 128, "num_it": 16},
    "hdkt": {"num_at": 128, "num_it": 16},
    "hqaf": {"num_type": 16},
}

# train_runner.py:293 sets this from the resolved mode, and it decides whether
# LPKT's initial knowledge state is a learned parameter or noise re-drawn on
# every forward. Omitting it put LPKT on a path real runs never take.
RUNTIME_CONCEPT_MODELS = {"lpkt"}

# train_runner.py:295 injects num_pid for these from dataset_cfg["num_q"].
NEEDS_NUM_PID = {
    "simplekt", "ukt", "stablekt", "sparsekt", "robustkt", "dtransformer",
    "lefokt_akt", "hqaf",
}


def runner_dataset_mode(model_name):
    """The mode the runner would resolve, rather than one the test invents.

    `spec.dataset_mode or "one_by_one"` looked reasonable and was wrong: only
    gkt declares a spec so far, so every other model fell through to
    one_by_one, while train_config sets all_in_one globally and
    ALL_IN_ONE_MODELS pins eighteen more.
    """
    return resolve_dataset_mode(
        model_name,
        KT_CONFIG.get("train_config", {}),
        KT_CONFIG.get(model_name, {}),
        overrides=None,
        spec=spec_for(MODEL_REGISTRY.get(model_name)),
    )


def runner_concept_shape(model_name, dataset_mode):
    """3-D concepts only when the dataset would actually produce them.

    all_in_one is not enough. Models outside MULTI_CONCEPT_MODELS -- rekt and
    gkt keep per-KC state arrays indexed by a single skill id -- are fed
    `cseqs[:, 0]`, so they see 2-D concepts even in all_in_one. Handing rekt a
    [B, T, K] tensor fails inside models/rekt.py:102 rather than anywhere the
    harness can explain.
    """
    override = KT_CONFIG.get(model_name, {}).get("concept_mode")
    multi = resolve_concept_mode(model_name, override) == "multi"
    return "3d" if (dataset_mode == "all_in_one" and multi) else "2d"


def config_for(model_name):
    """The model's config block minus the keys the runner routes elsewhere."""
    cfg = dict(KT_CONFIG.get(model_name, {}))
    for key in (
        "learning_rate", "optimizer", "dpath", "emb_path", "dataset_mode",
        "concept_mode", "eval_window", "num_epochs", "batch_size",
        # Passed explicitly by build_model, so leaving it here duplicates it --
        # the same shape of collision `other_config_keys` guards against in the
        # runner.
        "emb_type",
    ):
        cfg.pop(key, None)
    return cfg


def synth_batch(dataset_mode, concept_shape="3d", seq_len=SEQ_POSITIONS, device=DEVICE):
    """A batch shaped like KTDataset.__getitem__ after collation.

    Dtypes are the ones a real loader produces, checked against an assist2009
    batch: ids int64, `rseqs` float32, `masks` and `smasks` bool. Several
    trainers rely on the incoming dtype rather than casting, so a wrong guess
    surfaces as "Found dtype Long but expected Float" or "masked_select:
    expected BoolTensor" inside the model rather than in the harness.

    Every optional feature key is present, so models that consume one find it
    without the harness having to know which models those are. `masks` and
    `smasks` are all true, which keeps the alignment check exact.
    """
    g = torch.Generator().manual_seed(0)

    def ids(high, shape):
        return torch.randint(0, high, shape, generator=g).long().to(device)

    concept_dims = (
        (BATCH, seq_len, MAX_CONCEPTS) if concept_shape == "3d" else (BATCH, seq_len)
    )
    responses = torch.randint(0, 2, (BATCH, seq_len), generator=g).float().to(device)
    shft_responses = torch.randint(0, 2, (BATCH, seq_len), generator=g).float().to(device)

    batch = {
        "qseqs": ids(NUM_Q, (BATCH, seq_len)),
        "shft_qseqs": ids(NUM_Q, (BATCH, seq_len)),
        "cseqs": ids(NUM_C, concept_dims),
        "shft_cseqs": ids(NUM_C, concept_dims),
        "rseqs": responses,
        "shft_rseqs": shft_responses,
        "masks": torch.ones(BATCH, seq_len, dtype=torch.bool, device=device),
        "smasks": torch.ones(BATCH, seq_len, dtype=torch.bool, device=device),
        # Timestamps and gaps.
        "tseqs": ids(100, (BATCH, seq_len)),
        "shft_tseqs": ids(100, (BATCH, seq_len)),
        "itseqs": ids(10, (BATCH, seq_len)),
        "shft_itseqs": ids(10, (BATCH, seq_len)),
        "utseqs": ids(100, (BATCH, seq_len)),
        "shft_utseqs": ids(100, (BATCH, seq_len)),
        # dkt_forget.
        "rgaps": ids(5, (BATCH, seq_len)),
        "shft_rgaps": ids(5, (BATCH, seq_len)),
        "sgaps": ids(5, (BATCH, seq_len)),
        "shft_sgaps": ids(5, (BATCH, seq_len)),
        "pcounts": ids(5, (BATCH, seq_len)),
        "shft_pcounts": ids(5, (BATCH, seq_len)),
        # dimkt / hqaf difficulty.
        "sdseqs": ids(5, (BATCH, seq_len)),
        "shft_sdseqs": ids(5, (BATCH, seq_len)),
        "qdseqs": ids(5, (BATCH, seq_len)),
        "shft_qdseqs": ids(5, (BATCH, seq_len)),
        # hqaf question-type and time-bucket attributes.
        "quseqs": ids(8, (BATCH, seq_len)),
        "shft_quseqs": ids(8, (BATCH, seq_len)),
        "pTseqs": ids(8, (BATCH, seq_len)),
        "shft_pTseqs": ids(8, (BATCH, seq_len)),
        # atdkt history.
        "historycorrs": torch.rand(BATCH, seq_len, generator=g).to(device),
        "shft_historycorrs": torch.rand(BATCH, seq_len, generator=g).to(device),
    }
    return batch


def make_model_and_trainer(model_name, dataset_mode, device=DEVICE):
    cfg = config_for(model_name)
    spec = spec_for(MODEL_REGISTRY.get(model_name))

    kwargs = dict(cfg)
    kwargs.update(COMPUTED_CONSTRUCTOR_ARGS.get(model_name, {}))
    # Declarative parts of the spec always apply.
    if spec.needs_num_pid:
        kwargs.setdefault("num_pid", NUM_Q)
    if model_name in RUNTIME_CONCEPT_MODELS:
        kwargs["use_runtime_concepts"] = dataset_mode == "all_in_one"
    # `prepare` is only consulted for specs that compute from the synthetic
    # sizes rather than from real files. A spec that reads the dataset is
    # exercised by a real training run instead -- see FITS_FROM_REAL_DATA.
    if model_name not in FITS_FROM_REAL_DATA:
        kwargs.update(spec.prepare(_MinimalContext(model_name, dataset_mode)).model_kwargs)

    model = build_model(
        model_name,
        num_c=NUM_C,
        num_q=NUM_Q,
        emb_type=KT_CONFIG.get(model_name, {}).get("emb_type", "qid"),
        seq_len=SEQ_LEN,
        device=device,
        dpath="",
        **kwargs,
    ).to(device)

    trainer = build_trainer(
        model_name,
        model=model,
        train_loader=None,
        valid_loader=None,
        optimizer=torch.optim.Adam(model.parameters(), lr=1e-3),
        num_epochs=1,
        device=device,
        hooks=None,
        test_loader=None,
    )
    return model, trainer


class _MinimalContext:
    """Enough of RunContext for a spec's declarative path on synthetic sizes."""

    def __init__(self, model_name, dataset_mode):
        self.model_name = model_name
        self.dataset_name = "synthetic"
        self.fold_id = 0
        self.dataset_mode = dataset_mode
        self.model_cfg = config_for(model_name)
        self.dataset_cfg = {"num_c": NUM_C, "num_q": NUM_Q, "folds": [0, 1], "dpath": ""}
        self.train_cfg = {}
        self.root_dir = str(ROOT)
        self.resolve_file = lambda primary, fallback: primary

    def train_folds(self):
        return [1]

    def quelevel_key(self, base_key):
        return f"{base_key}_quelevel" if self.dataset_mode == "all_in_one" else base_key


def forward(trainer, batch):
    """Call `_forward_batch` across its two shapes and return (pred, target, loss)."""
    try:
        result = trainer._forward_batch(batch)
    except TypeError as exc:
        if "train" not in str(exc):
            raise
        result = trainer._forward_batch(batch, train=True)
    # akt returns (pred, target, reg_loss, loss); everyone else (pred, target, loss).
    pred, target = result[0], result[1]
    loss = result[-1]
    return pred, target, loss


class ModelContractTest(unittest.TestCase):
    """One subTest per registered model, so one failure names its model."""

    @classmethod
    def setUpClass(cls):
        cls.model_names = canonical_models()
        cls.trainer_names = set(TRAINER_REGISTRY.get_all())

    def test_1_every_model_has_a_trainer(self):
        for name in self.model_names:
            with self.subTest(model=name):
                self.assertTrue(
                    name in self.trainer_names,
                    f"model {name!r} is registered but no trainer is; "
                    "check core/trainers/__init__.py",
                )

    def test_2_every_trainer_has_a_model(self):
        registered = set(MODEL_REGISTRY.get_all())
        for name in sorted(self.trainer_names):
            with self.subTest(trainer=name):
                self.assertTrue(
                    name in registered,
                    f"trainer {name!r} is registered but no model is; "
                    "check models/__init__.py",
                )

    def test_3_every_model_has_a_config_block(self):
        for name in self.model_names:
            with self.subTest(model=name):
                self.assertTrue(
                    name in KT_CONFIG,
                    f"no {name!r} block in configs/kt_config.json; "
                    "train_one_fold raises before building anything",
                )

    def test_4_forward_backward_and_alignment(self):
        covered, skipped = [], []
        for name in self.model_names:
            if name in NEEDS_REAL_ARTEFACTS:
                skipped.append(name)
                continue
            with self.subTest(model=name):
                mode = runner_dataset_mode(name)
                model, trainer = make_model_and_trainer(name, mode)
                batch = synth_batch(mode, runner_concept_shape(name, mode))

                pred, target, loss = forward(trainer, batch)

                # 5. alignment
                self.assertEqual(
                    pred.numel(), target.numel(),
                    f"{name}: {pred.numel()} predictions against {target.numel()} targets",
                )
                scored = int(batch["smasks"].sum())
                if name in KNOWN_ALIGNMENT_VIOLATIONS:
                    self.assertNotEqual(
                        pred.numel(), scored,
                        f"{name} now aligns with smasks. If that was the intent, "
                        f"delete its KNOWN_ALIGNMENT_VIOLATIONS entry.\n"
                        f"{KNOWN_ALIGNMENT_VIOLATIONS[name]}",
                    )
                else:
                    self.assertEqual(
                        pred.numel(), scored,
                        f"{name}: {pred.numel()} scored positions but smasks selects "
                        f"{scored}. Its AUC would cover a different position set "
                        "than every other model in the same table.",
                    )

                # 6. finiteness, forward
                self.assertTrue(torch.isfinite(loss).all(), f"{name}: non-finite loss")
                self.assertTrue(torch.isfinite(pred).all(), f"{name}: non-finite predictions")
                in_range = bool(((pred >= 0) & (pred <= 1)).all())
                if name in KNOWN_RANGE_VIOLATIONS:
                    self.assertFalse(
                        in_range,
                        f"{name} now emits probabilities. If that was the intent, "
                        f"delete its KNOWN_RANGE_VIOLATIONS entry.\n"
                        f"{KNOWN_RANGE_VIOLATIONS[name]}",
                    )
                else:
                    self.assertTrue(
                        in_range,
                        f"{name}: predictions fall outside [0, 1] "
                        f"(min {pred.detach().min():.4f}, max {pred.detach().max():.4f}), "
                        "so the head is not a probability and the 0.5 accuracy "
                        "threshold in _score_loader is meaningless for it.",
                    )

                # 6. finiteness, backward
                loss.backward()
                grads = [
                    (n, p.grad) for n, p in model.named_parameters()
                    if p.grad is not None
                ]
                self.assertTrue(grads, f"{name}: backward produced no gradients at all")
                for pname, grad in grads:
                    self.assertTrue(
                        torch.isfinite(grad).all(),
                        f"{name}: non-finite gradient in {pname}",
                    )
                covered.append(name)

        print(
            f"\n  contract: {len(covered)}/{len(self.model_names)} models exercised, "
            f"{len(skipped)} skipped ({', '.join(skipped)})"
        )

    @unittest.skipUnless(torch.cuda.is_available(),
                         "only meaningful when a GPU is present to be wrongly preferred")
    def test_4b_a_model_built_on_cpu_stays_on_cpu(self):
        """A GPU being available must not decide where a model runs.

        Several files hold a module-level
        `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
        and reach for it instead of the device they were handed, so on a machine
        with a GPU they could not be forced onto CPU at all. That blocks a
        CPU-only CI job, and the same defect silently ignores --gpu_id on a
        multi-GPU machine: IEKT rebuilt `device` from scratch inside its own
        constructor, so `cuda:1` still built on `cuda:0`.

        Skipped without CUDA, where the module-level default happens to be right
        and the test would prove nothing.
        """
        for name in self.model_names:
            if name in NEEDS_REAL_ARTEFACTS:
                continue
            with self.subTest(model=name):
                mode = runner_dataset_mode(name)
                _, trainer = make_model_and_trainer(name, mode, device="cpu")
                trainer.model.eval()
                batch = synth_batch(mode, runner_concept_shape(name, mode), device="cpu")

                with torch.no_grad():
                    pred, _, _ = forward(trainer, batch)

                self.assertEqual(
                    pred.device.type, "cpu",
                    f"{name}: built on CPU but produced predictions on "
                    f"{pred.device}. Something reached for a module-level device "
                    "instead of the one it was given.",
                )

    def test_5_inference_is_deterministic_in_eval_mode(self):
        """Two forwards over the same batch, in eval, must give the same answer.

        This runs before the causality check and not as part of it, because a
        non-deterministic forward makes that check unreadable: any difference
        after flipping a response could be the leak or could be the noise. When
        LPKT first failed the causality check I attributed the difference to
        cuBLAS reduction ordering, which was wrong -- models/lpkt.py:252 calls
        nn.init.xavier_uniform_ inside forward whenever initial_knowledge is
        None, re-drawing the whole initial state on every pass. Seeding made it
        reproduce exactly, which was the signal, and a separate determinism test
        would have said so immediately.

        That path is reachable: initial_knowledge is a learned parameter only
        when use_runtime_concepts is on, which the runner ties to all_in_one. In
        one_by_one LPKT re-randomises its initial knowledge state every forward,
        so its evaluation is not reproducible.
        """
        for name in self.model_names:
            if name in NEEDS_REAL_ARTEFACTS:
                continue
            with self.subTest(model=name):
                mode = runner_dataset_mode(name)
                _, trainer = make_model_and_trainer(name, mode)
                trainer.model.eval()
                batch = synth_batch(mode, runner_concept_shape(name, mode))

                with torch.no_grad():
                    first, _, _ = forward(trainer, batch)
                    second, _, _ = forward(trainer, batch)

                drift = float((first - second).abs().max())
                if name in KNOWN_NONDETERMINISM:
                    self.assertGreater(
                        drift, 1e-6,
                        f"{name} is now deterministic in eval. If that was the "
                        f"intent, delete its KNOWN_NONDETERMINISM entry.\n"
                        f"{KNOWN_NONDETERMINISM[name]}",
                    )
                else:
                    self.assertLessEqual(
                        drift, 1e-6,
                        f"{name}: the same batch scored twice in eval mode differs "
                        f"by {drift:.3e}. Evaluation is not reproducible; look for "
                        "an nn.init call or a sampling step inside forward that is "
                        "not gated on self.training.",
                    )

    def test_6_a_future_response_cannot_move_a_past_prediction(self):
        """The leak that matters most: prediction at t must not see response t.

        `rseqs[:, -1]` is the last input response and legitimately feeds only the
        final prediction. Flipping it must leave every earlier prediction alone.

        Determinism is established by the test above, so any movement here is
        attributable. The threshold stays calibrated rather than fixed, because
        CUDA reductions can still reorder at the 1e-7 level.
        """
        for name in self.model_names:
            if name in NEEDS_REAL_ARTEFACTS or name in KNOWN_ALIGNMENT_VIOLATIONS:
                continue
            with self.subTest(model=name):
                mode = runner_dataset_mode(name)
                _, trainer = make_model_and_trainer(name, mode)
                trainer.model.eval()

                batch = synth_batch(mode, runner_concept_shape(name, mode))
                flipped = dict(batch)
                flipped["rseqs"] = batch["rseqs"].clone()
                flipped["rseqs"][:, -1] = 1 - flipped["rseqs"][:, -1]

                with torch.no_grad():
                    base, _, _ = forward(trainer, batch)
                    repeat, _, _ = forward(trainer, batch)
                    after, _, _ = forward(trainer, flipped)

                # smasks is all true, so pred is [B, T] flattened row-major and
                # column t maps back to position t.
                past = slice(None, -1)
                noise = (base.view(BATCH, -1)[:, past]
                         - repeat.view(BATCH, -1)[:, past]).abs().max()
                moved = (base.view(BATCH, -1)[:, past]
                         - after.view(BATCH, -1)[:, past]).abs().max()

                self.assertLessEqual(
                    float(moved), max(float(noise), 1e-6) * 10,
                    f"{name}: flipping the LAST response moved earlier predictions "
                    f"by up to {float(moved):.3e}, against a run-to-run noise floor "
                    f"of {float(noise):.3e}. The forward pass lets a response "
                    "influence a prediction at or before its own step.",
                )


if __name__ == "__main__":
    unittest.main(verbosity=2)
