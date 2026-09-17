"""The protocol block must record which splits derived inputs were fitted from.

None of these features read responses, so none of them is label leakage. What
they decide is whether a run is transductive -- whether the model's structure
was built already knowing what the test set contains. A reviewer asking "did
your graph see the test set?" should be answerable from an artifact rather than
by reading the runner.

The scopes are genuinely mixed across models: dimkt, hqaf and lpkt/hdkt already
pass `folds=train_folds`, while dkt_forget, gkt and dgekt read every split. That
is why the value is recorded per run at the site that does the fitting, rather
than assumed or kept in a lookup table -- a table drifts away from the code,
which is exactly what went wrong with the ALL_IN_ONE_MODELS membership sets.
"""

import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import models  # noqa: F401  -- registers model classes
from core.model_inputs import spec_for
from core.registry import MODEL_REGISTRY
from datasets.init_dataset import FIT_SCOPES, protocol_stamp

DATA_CONFIG = json.loads((ROOT / "configs" / "data_config.json").read_text(encoding="utf-8"))


class ProtocolStampTest(unittest.TestCase):
    def test_defaults_to_fitting_nothing(self):
        """Most models derive no inputs from the data at all."""
        stamp = protocol_stamp("dkt", "all_in_one", 4)
        self.assertEqual(stamp["feature_fit_scope"], "none")
        self.assertEqual(stamp["graph_scope"], "none")

    def test_records_what_it_is_given(self):
        stamp = protocol_stamp(
            "dkt_forget", "all_in_one", 4,
            feature_fit_scope="train_valid_test", graph_scope="none",
        )
        self.assertEqual(stamp["feature_fit_scope"], "train_valid_test")

    def test_rejects_an_unknown_scope(self):
        """A typo must fail at the stamp, not become a value in an artifact."""
        with self.assertRaises(ValueError) as err:
            protocol_stamp("dkt", "all_in_one", 4, feature_fit_scope="train")
        self.assertIn("train_valid_test", str(err.exception))

        with self.assertRaises(ValueError):
            protocol_stamp("dkt", "all_in_one", 4, graph_scope="everything")

    def test_the_stamp_carries_every_field_the_key_groups_on(self):
        stamp = protocol_stamp("dkt", "all_in_one", 4)
        for key in ("dataset_mode", "concept_mode", "max_concepts",
                    "concepts_visible", "score_repeated_kc", "eval_window",
                    "feature_fit_scope", "graph_scope"):
            self.assertIn(key, stamp)


class ProtocolKeyTest(unittest.TestCase):
    """A field recorded but not grouped on is decorative.

    The first version of protocol_key read five of the eight fields, so a
    `train_folds` run and a `train_valid_test` run still landed in the same
    table -- the scope sat in the artifact and changed nothing.
    """

    def _key(self, **overrides):
        from scripts.run_baseline_table import protocol_key

        protocol = protocol_stamp("dkt", "all_in_one", 4)
        protocol.update(overrides)
        return protocol_key({"protocol": protocol})

    def test_differing_feature_scope_does_not_group_together(self):
        self.assertNotEqual(
            self._key(feature_fit_scope="train_folds"),
            self._key(feature_fit_scope="train_valid_test"),
        )

    def test_differing_graph_scope_does_not_group_together(self):
        self.assertNotEqual(
            self._key(graph_scope="none"),
            self._key(graph_scope="train_valid_test"),
        )

    def test_differing_max_concepts_does_not_group_together(self):
        self.assertNotEqual(self._key(max_concepts=4), self._key(max_concepts=7))

    def test_identical_protocols_do_group_together(self):
        self.assertEqual(self._key(), self._key())

    def test_a_run_predating_a_field_is_unknown_not_compatible(self):
        """An older artifact recorded nothing; it must not be assumed to match.

        Defaulting a missing scope to `train_folds` would silently merge runs
        whose scope nobody recorded into a table of runs that did record it.
        """
        from scripts.run_baseline_table import protocol_key

        old = protocol_stamp("dkt", "all_in_one", 4)
        del old["feature_fit_scope"]
        del old["graph_scope"]

        self.assertNotEqual(protocol_key({"protocol": old}), self._key())
        self.assertIn("unknown", protocol_key({"protocol": old}))

    def test_a_run_with_no_protocol_block_is_still_unplaceable(self):
        from scripts.run_baseline_table import protocol_key

        self.assertIsNone(protocol_key({}))

    def test_describe_names_a_transductive_run(self):
        from scripts.run_baseline_table import describe_protocol

        text = describe_protocol(self._key(graph_scope="train_valid_test"))
        self.assertIn("transductive", text)

    def test_every_scope_name_is_accepted(self):
        for scope in FIT_SCOPES:
            with self.subTest(scope=scope):
                protocol_stamp("dkt", "all_in_one", 4, feature_fit_scope=scope,
                               graph_scope=scope)


class GktSpecScopeTest(unittest.TestCase):
    """GKT reports its own scope, the same way it reports its graph."""

    def _prepare(self, graph_type, transductive=False):
        from tests.test_model_contracts import _MinimalContext

        ctx = _MinimalContext("gkt", "all_in_one")
        ctx.dataset_cfg = dict(DATA_CONFIG["assist2009"])
        ctx.model_cfg = {"graph_type": graph_type}
        ctx.train_cfg = {"pykt_transductive": transductive}
        return spec_for(MODEL_REGISTRY.get("gkt")).prepare(ctx)

    def test_transition_graph_defaults_to_the_training_folds(self):
        """AGENTS.md: a derived feature is fitted on the training folds only.

        Not cosmetic on this dataset: counting assist2009's transitions from
        every split gives 3953 non-zero edges against 3511 from the training
        folds, so 11% of the graph exists only because it saw valid and test.
        """
        extras = self._prepare("transition").run_config_extras
        self.assertEqual(extras["graph_scope"], "train_folds")

    def test_pykt_transductive_restores_the_old_scope(self):
        extras = self._prepare("transition", transductive=True).run_config_extras
        self.assertEqual(extras["graph_scope"], "train_valid_test")

    def test_dense_graph_reads_nothing(self):
        """All ones, so no split contributes to it, under either setting."""
        for transductive in (False, True):
            with self.subTest(transductive=transductive):
                extras = self._prepare("dense", transductive).run_config_extras
                self.assertEqual(extras["graph_scope"], "none")

    def test_the_two_scopes_do_not_share_a_graph_cache(self):
        """A cached graph from one scope must never be served to the other."""
        import re

        source = (ROOT / "models" / "gkt.py").read_text(encoding="utf-8")
        self.assertRegex(source, r'scope = "tvt" if transductive else f"tf\{ctx\.fold_id\}"')
        self.assertIn("{scope}", source)

    def test_reported_scopes_are_valid_names(self):
        for graph_type in ("transition", "dense"):
            for transductive in (False, True):
                with self.subTest(graph_type=graph_type, transductive=transductive):
                    extras = self._prepare(graph_type, transductive).run_config_extras
                    self.assertIn(extras["graph_scope"], FIT_SCOPES)


if __name__ == "__main__":
    unittest.main()
