"""Every hyperparameter in kt_config.json must reach something that reads it.

`core/factory.py::_filter_to_signature` drops any keyword a constructor does not
name, and many models take `**kwargs` and absorb the rest. Both are deliberate --
model and trainer take different subsets of one config block -- and both are
silent. So `dropuot: 0.2` would leave the model on its default dropout, the run
would finish, and the number would be reported as if the setting had applied.

Nothing catches that at runtime by construction, so it is caught here instead: a
key is acceptable if something plausibly reads it, and a typo reaches nothing.

A key counts as consumed when it is

  - a parameter anywhere in the model's or trainer's constructor MRO, which
    covers a wrapper forwarding **kwargs to the real implementation (DTransformer
    and StableKT both do this, and a signature-only check wrongly calls their
    n_know and d_ff orphans);
  - named in the model's, trainer's, or a runner-side consumer's source; or
  - in NON_MODEL_CONFIG_KEYS, the set the runner strips before construction.
"""

import inspect
import json
import re
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import core.trainers  # noqa: F401  -- registers trainers
import models  # noqa: F401  -- registers models
from core.registry import MODEL_REGISTRY, TRAINER_REGISTRY
from core.train_runner import MODEL_NAME_ALIASES, NON_MODEL_CONFIG_KEYS

KT_CONFIG = json.loads((ROOT / "configs" / "kt_config.json").read_text(encoding="utf-8"))

# Supplied by the runner or the CLI rather than by the model's block.
RUNNER_SUPPLIED = {
    "emb_type", "learning_rate", "use_timestamps", "dpath", "dataset_mode",
    "concept_mode", "eval_window", "num_epochs", "batch_size", "optimizer",
    "seq_len", "num_at", "num_it",
}

# Files that read a model's config without being the model or its trainer.
RUNNER_SIDE_CONSUMERS = (
    ROOT / "core" / "run_support.py",        # build_optimizer: weight_decay
    ROOT / "core" / "train_runner.py",
    ROOT / "strategies" / "dkt_pebg_strategy.py",
)


def _class_chain(name):
    """The model and trainer classes for `name`, with their full MROs."""
    classes = [MODEL_REGISTRY.get(name)]
    if name in TRAINER_REGISTRY.get_all():
        classes.append(TRAINER_REGISTRY.get(name))

    seen, chain = set(), []
    for cls in classes:
        for base in cls.__mro__:
            if base in seen or base is object:
                continue
            seen.add(base)
            chain.append(base)
    return chain


def consumed_names(name):
    """Every identifier that could plausibly read a config key for this model."""
    params, sources = set(), []
    for base in _class_chain(name):
        try:
            params |= set(inspect.signature(base.__init__).parameters)
        except (TypeError, ValueError):
            pass
        try:
            sources.append(Path(inspect.getfile(base)).read_text(encoding="utf-8"))
        except (TypeError, OSError):
            pass
    for path in RUNNER_SIDE_CONSUMERS:
        if path.exists():
            sources.append(path.read_text(encoding="utf-8"))
    return params, "\n".join(sources)


def orphan_keys(name):
    block = KT_CONFIG.get(name, {})
    params, source = consumed_names(name)
    orphans = []
    for key in block:
        if key in RUNNER_SUPPLIED or key in NON_MODEL_CONFIG_KEYS or key in params:
            continue
        if re.search(r"\b" + re.escape(key) + r"\b", source):
            continue
        orphans.append(key)
    return orphans


def canonical_models():
    return sorted(k for k in MODEL_REGISTRY.get_all() if k not in MODEL_NAME_ALIASES)


class ConfigKeysAreConsumedTest(unittest.TestCase):
    def test_no_hyperparameter_is_silently_ignored(self):
        for name in canonical_models():
            if name not in KT_CONFIG:
                continue
            with self.subTest(model=name):
                orphans = orphan_keys(name)
                self.assertEqual(
                    orphans, [],
                    f"configs/kt_config.json sets {orphans} for {name!r}, and "
                    "nothing reads them -- not the constructor MRO, not the "
                    "model or trainer source, not a runner-side consumer. Either "
                    "it is a typo, or the setting is dead and should be deleted.",
                )

    def test_a_planted_typo_is_caught(self):
        """The check has to be able to fail, or it is decorative."""
        KT_CONFIG.setdefault("dkt", {})["dropuot"] = 0.2
        try:
            self.assertIn("dropuot", orphan_keys("dkt"))
        finally:
            KT_CONFIG["dkt"].pop("dropuot")

    def test_a_wrapper_forwarding_kwargs_is_not_a_false_alarm(self):
        """DTransformer's registered class forwards n_know to its parent.

        A signature-only check on the wrapper calls that an orphan, which is how
        the first version of this audit produced ten false positives.
        """
        self.assertIn("n_know", KT_CONFIG.get("dtransformer", {}))
        self.assertEqual(orphan_keys("dtransformer"), [])


class NonModelConfigKeysTest(unittest.TestCase):
    def test_the_strip_list_has_no_dead_entries(self):
        """It collected nine entries from deleted a removed model models before anyone
        looked. An entry no config block uses is either dead or a typo."""
        used = set()
        for block in KT_CONFIG.values():
            if isinstance(block, dict):
                used |= set(block)
        # `concept_mode` is a documented per-model override that measures the
        # cost of KC truncation, so it is absent until someone runs that
        # experiment; `dpath` is defensive, since the runner passes it
        # explicitly to build_model.
        expected_absent = {"concept_mode", "dpath"}
        dead = sorted(NON_MODEL_CONFIG_KEYS - used - expected_absent)
        self.assertEqual(
            dead, [],
            f"NON_MODEL_CONFIG_KEYS strips {dead}, which no config block sets.",
        )

    def test_stripped_keys_do_not_reach_a_constructor(self):
        """Whatever is stripped must genuinely be read by someone else."""
        for name in canonical_models():
            block = KT_CONFIG.get(name, {})
            stripped = set(block) & NON_MODEL_CONFIG_KEYS
            if not stripped:
                continue
            with self.subTest(model=name):
                _, source = consumed_names(name)
                for key in sorted(stripped):
                    self.assertRegex(
                        source, r"\b" + re.escape(key) + r"\b",
                        f"{name!r} sets {key!r} and the runner strips it, but no "
                        "consumer names it either -- it reaches nothing at all.",
                    )


if __name__ == "__main__":
    unittest.main()
