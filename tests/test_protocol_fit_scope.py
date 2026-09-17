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

    def test_the_existing_fields_are_untouched(self):
        """Adding fields must not disturb the ones run_baseline_table groups on.

        scripts/run_baseline_table.py::protocol_key reads five keys by name, so
        older runs missing the new fields still group with newer ones.
        """
        stamp = protocol_stamp("dkt", "all_in_one", 4)
        for key in ("dataset_mode", "concept_mode", "max_concepts",
                    "concepts_visible", "score_repeated_kc", "eval_window"):
            self.assertIn(key, stamp)

    def test_every_scope_name_is_accepted(self):
        for scope in FIT_SCOPES:
            with self.subTest(scope=scope):
                protocol_stamp("dkt", "all_in_one", 4, feature_fit_scope=scope,
                               graph_scope=scope)


class GktSpecScopeTest(unittest.TestCase):
    """GKT reports its own scope, the same way it reports its graph."""

    def _prepare(self, graph_type):
        from tests.test_model_contracts import _MinimalContext

        ctx = _MinimalContext("gkt", "all_in_one")
        ctx.dataset_cfg = dict(DATA_CONFIG["assist2009"])
        ctx.model_cfg = {"graph_type": graph_type}
        return spec_for(MODEL_REGISTRY.get("gkt")).prepare(ctx)

    def test_transition_graph_is_transductive(self):
        """Counted from the train and test sequence files, with no fold filter."""
        extras = self._prepare("transition").run_config_extras
        self.assertEqual(extras["graph_scope"], "train_valid_test")

    def test_dense_graph_reads_nothing(self):
        """All ones, so no split contributes to it."""
        extras = self._prepare("dense").run_config_extras
        self.assertEqual(extras["graph_scope"], "none")

    def test_reported_scopes_are_valid_names(self):
        for graph_type in ("transition", "dense"):
            with self.subTest(graph_type=graph_type):
                extras = self._prepare(graph_type).run_config_extras
                self.assertIn(extras["graph_scope"], FIT_SCOPES)


if __name__ == "__main__":
    unittest.main()
