"""The `Inputs` spec contract that core/train_runner.py relies on.

These cover the plumbing only -- that a spec is found, that its declarations
take effect, that a bad one fails loudly. Whether a *moved* model still produces
the same numbers is not a unit-test question; that is
research/check_input_refactor.py, which compares real runs bit-for-bit.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: F401  -- registers model classes
from core.model_inputs import InputSpec, ModelInputs, RunContext, spec_for
from core.registry import MODEL_REGISTRY


def context(**overrides):
    base = dict(
        model_name="test_model",
        dataset_name="assist2009",
        fold_id=0,
        dataset_mode="one_by_one",
        model_cfg={},
        dataset_cfg={"folds": [0, 1, 2, 3, 4], "input_type": ["questions", "concepts"], "num_q": 17737},
        train_cfg={},
        root_dir=str(ROOT),
        resolve_file=lambda primary, fallback: primary,
    )
    base.update(overrides)
    return RunContext(**base)


class SpecLookupTest(unittest.TestCase):
    def test_model_without_a_spec_gets_the_default(self):
        self.assertIs(spec_for(MODEL_REGISTRY.get("dkt")), InputSpec)

    def test_model_with_a_spec_gets_its_own(self):
        spec = spec_for(MODEL_REGISTRY.get("gkt"))
        self.assertIsNot(spec, InputSpec)
        self.assertTrue(issubclass(spec, InputSpec))

    def test_a_non_spec_Inputs_attribute_fails_loudly(self):
        class Bogus:
            Inputs = "not a class"

        with self.assertRaises(TypeError) as ctx:
            spec_for(Bogus)
        self.assertIn("InputSpec", str(ctx.exception))

    def test_every_registered_spec_is_well_formed(self):
        """A malformed spec must not wait for a training run to surface."""
        for name in MODEL_REGISTRY.get_all():
            with self.subTest(model=name):
                spec = spec_for(MODEL_REGISTRY.get(name))
                self.assertTrue(issubclass(spec, InputSpec))
                self.assertIn(spec.dataset_mode, (None, "all_in_one", "one_by_one"))


class DefaultSpecTest(unittest.TestCase):
    def test_default_prepare_asks_for_nothing(self):
        got = InputSpec.prepare(context())
        self.assertEqual(got.model_kwargs, {})
        self.assertEqual(got.dataset_kwargs, {})
        self.assertEqual(got.model_cfg_updates, {})
        self.assertEqual(got.run_config_extras, {})

    def test_needs_num_pid_injects_the_question_count(self):
        class Spec(InputSpec):
            needs_num_pid = True

        self.assertEqual(Spec.prepare(context()).model_kwargs["num_pid"], 17737)

    def test_post_build_returns_the_model_unchanged_by_default(self):
        sentinel = object()
        self.assertIs(InputSpec.post_build(sentinel, context()), sentinel)


class ValidationTest(unittest.TestCase):
    def test_question_requirement_passes_on_a_question_dataset(self):
        class Spec(InputSpec):
            requires_question_ids = True

        Spec.validate(context())  # must not raise

    def test_question_requirement_names_the_dataset_when_it_fails(self):
        class Spec(InputSpec):
            requires_question_ids = True

        ctx = context(dataset_cfg={"input_type": ["concepts"], "num_q": 0})
        with self.assertRaises(ValueError) as err:
            Spec.validate(ctx)
        message = str(err.exception)
        self.assertIn("assist2009", message)
        self.assertIn("test_model", message)

    def test_question_requirement_rejects_zero_num_q(self):
        class Spec(InputSpec):
            requires_question_ids = True

        with self.assertRaises(ValueError):
            Spec.validate(context(dataset_cfg={"input_type": ["questions"], "num_q": 0}))


class RunContextTest(unittest.TestCase):
    def test_train_folds_excludes_the_current_fold(self):
        self.assertEqual(context(fold_id=2).train_folds(), [0, 1, 3, 4])

    def test_train_folds_handles_a_string_fold_id(self):
        self.assertEqual(context(fold_id="2").train_folds(), [0, 1, 3, 4])

    def test_quelevel_key_follows_the_resolved_mode(self):
        one = context(dataset_mode="one_by_one")
        allin = context(dataset_mode="all_in_one")
        self.assertEqual(one.quelevel_key("train_valid_file"), "train_valid_file")
        self.assertEqual(
            allin.quelevel_key("train_valid_file"), "train_valid_file_quelevel"
        )


class GktSpecTest(unittest.TestCase):
    def test_gkt_declares_only_a_graph(self):
        """The migrated case: one model_kwarg, no config mutation, no side effects."""
        spec = spec_for(MODEL_REGISTRY.get("gkt"))
        import json

        data_config = json.loads((ROOT / "configs" / "data_config.json").read_text(encoding="utf-8"))
        dataset_cfg = dict(data_config["assist2009"])
        dataset_cfg["dpath"] = str(ROOT / dataset_cfg["dpath"])

        got = spec.prepare(context(model_name="gkt", dataset_cfg=dataset_cfg))
        self.assertEqual(set(got.model_kwargs), {"graph"})
        self.assertEqual(got.model_kwargs["graph"].shape[0], dataset_cfg["num_c"])
        self.assertEqual(got.dataset_kwargs, {})
        self.assertEqual(got.model_cfg_updates, {})
        self.assertEqual(got.dataset_cfg_updates, {})


if __name__ == "__main__":
    unittest.main()
