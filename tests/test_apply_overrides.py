"""An override the runner cannot apply must stop the run, not vanish.

`apply_overrides` used to walk its own allowlist and ignore everything else. A
caller that passed a key outside that list got no error and no warning: the run
trained on the config-file default, and `run_config.json` recorded the override
as though it had taken effect. Two runs that differed only in a dropped key
produced bit-identical curves, which reads as "this parameter does nothing".

That is what happened to dgekt's `kd_lambda`, and `kd_lambda` turned out to be
the difference between 0.5856 and 0.7401 test AUC on assist2009 fold 0.

This is the same failure mode tests/test_config_keys_are_consumed.py guards on
the config-file side; this file guards the override path.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.run_support import apply_overrides  # noqa: E402


class ApplyOverridesTest(unittest.TestCase):
    def test_known_keys_reach_the_right_config_block(self):
        train_cfg = {"batch_size": 64, "num_epochs": 200}
        model_cfg = {"learning_rate": 1e-3, "kd_lambda": 5e-6}
        apply_overrides(
            train_cfg,
            model_cfg,
            {"batch_size": 128, "learning_rate": 1e-2, "kd_lambda": 7.8e-8},
        )
        self.assertEqual(train_cfg["batch_size"], 128)
        self.assertEqual(train_cfg["num_epochs"], 200)
        self.assertEqual(model_cfg["learning_rate"], 1e-2)
        self.assertEqual(model_cfg["kd_lambda"], 7.8e-8)

    def test_unknown_key_raises_instead_of_being_dropped(self):
        with self.assertRaises(KeyError) as caught:
            apply_overrides({}, {}, {"kd_lambdaa": 1.0})
        self.assertIn("kd_lambdaa", str(caught.exception))

    def test_unknown_key_raises_even_when_it_is_none(self):
        # A caller that always builds the same dict and leaves unset options as
        # None still names the key, so the typo is just as real.
        with self.assertRaises(KeyError):
            apply_overrides({}, {}, {"learnig_rate": None})

    def test_none_values_on_known_keys_leave_the_config_alone(self):
        train_cfg = {"batch_size": 64}
        model_cfg = {"dropout": 0.1}
        apply_overrides(train_cfg, model_cfg, {"batch_size": None, "dropout": None})
        self.assertEqual(train_cfg["batch_size"], 64)
        self.assertEqual(model_cfg["dropout"], 0.1)

    def test_empty_and_missing_overrides_are_accepted(self):
        apply_overrides({}, {}, {})
        apply_overrides({}, {}, None)

    def test_every_key_scripts_train_py_sends_is_applicable(self):
        """The CLI must not offer an option the runner will refuse."""
        source = (ROOT / "scripts" / "train.py").read_text(encoding="utf-8")
        block = source.split("overrides = {", 1)[1].split("}", 1)[0]
        keys = [
            line.split('"')[1]
            for line in block.splitlines()
            if line.strip().startswith('"')
        ]
        self.assertIn("d_model", keys, "--d-model is defined but never forwarded")
        apply_overrides({}, {}, {key: None for key in keys})


if __name__ == "__main__":
    unittest.main()
