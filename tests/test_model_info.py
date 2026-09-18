"""What `model_info.json` has to get right.

The file exists because `run_config.json` records what a run was asked for and
nothing records what was built. The number that prompted it: on assist2009,
`hd_dkt` has 11.6x the parameters of `dkt`, almost all of it the denoiser's own
question-embedding table. An ablation row reading "dkt vs hd_dkt" is therefore
not only about denoising, and nothing in the artifacts said so.
"""

import json
import tempfile
import unittest
from pathlib import Path

import torch
import torch.nn as nn

from core.model_info import (
    collect_model_info,
    format_model_info,
    save_model_info_once,
)


class Toy(nn.Module):
    """Known counts: 6 + 3 trainable, 4 frozen, 5 buffer elements, 2 direct."""

    def __init__(self):
        super().__init__()
        self.head = nn.Linear(2, 3, bias=True)        # 6 weights + 3 bias
        self.frozen = nn.Embedding(2, 2)              # 4, frozen below
        self.frozen.weight.requires_grad = False
        self.direct = nn.Parameter(torch.zeros(2))    # not inside a child
        self.register_buffer("table", torch.zeros(5))


class CountingTest(unittest.TestCase):
    def setUp(self):
        self.info = collect_model_info(Toy(), device="cpu")

    def test_trainable_and_frozen_are_separate(self):
        """`dkt_pebg` freezes a pretrained embedding. Folding it into the
        trainable count overstates what the optimiser can move."""
        self.assertEqual(self.info["trainable_parameters"], 6 + 3 + 2)
        self.assertEqual(self.info["frozen_parameters"], 4)
        self.assertEqual(self.info["total_parameters"], 15)

    def test_buffers_are_counted_and_kept_out_of_the_parameter_total(self):
        """GKT's adjacency and LPKT's Q-matrix are buffers: real state that
        `parameters()` does not see."""
        self.assertEqual(self.info["buffer_elements"], 5)
        self.assertNotIn(5, [self.info["total_parameters"]])
        self.assertGreater(self.info["total_bytes"], self.info["parameter_bytes"])

    def test_a_parameter_held_directly_on_the_model_still_appears(self):
        """LPKT's `initial_knowledge` is a Parameter on the model itself. A
        per-child breakdown that skipped it would not sum to the total."""
        self.assertIn("(direct)", self.info["by_module"])
        self.assertEqual(self.info["by_module"]["(direct)"]["parameters"], 2)
        self.assertEqual(
            sum(r["parameters"] for r in self.info["by_module"].values()),
            self.info["total_parameters"],
        )

    def test_largest_tensors_are_ordered(self):
        sizes = [t["parameters"] for t in self.info["largest_tensors"]]
        self.assertEqual(sizes, sorted(sizes, reverse=True))

    def test_format_does_not_raise_on_a_plain_model(self):
        self.assertIn("parameters", format_model_info(self.info))


class CompositionTest(unittest.TestCase):
    def test_a_plugged_model_reports_what_the_plugin_costs(self):
        import models  # noqa: F401  -- registers the compositions
        from core.factory import build_model

        info = collect_model_info(
            build_model(
                "hd_dkt", num_c=6, num_q=12, emb_size=8,
                detector_hidden=8, latent_dim=4, dropout=0.0,
            )
        )
        comp = info["composition"]
        self.assertGreater(comp["plugin_parameters"], 0)
        self.assertGreater(comp["size_vs_backbone"], 1.0)
        self.assertEqual(
            comp["backbone_parameters"] + comp["plugin_parameters"],
            info["total_parameters"],
        )
        self.assertIn("the size of the bare backbone", format_model_info(info))

    def test_a_plain_model_reports_no_composition(self):
        from models.dkt import DKT

        self.assertNotIn(
            "composition", collect_model_info(DKT(num_c=6, emb_size=8))
        )


class WriteOnceTest(unittest.TestCase):
    def test_folds_that_build_the_same_model_write_one_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            info = collect_model_info(Toy())
            self.assertIsNotNone(save_model_info_once(tmp, info, fold_id=0))
            for fold in (1, 2, 3, 4):
                self.assertIsNone(save_model_info_once(tmp, info, fold_id=fold))
            self.assertEqual(
                sorted(p.name for p in Path(tmp).iterdir()), ["model_info.json"]
            )
            written = json.loads((Path(tmp) / "model_info.json").read_text())
            self.assertEqual(written["fold"], 0)

    def test_a_fold_that_builds_a_different_model_is_not_hidden(self):
        """`dkt_forget` sizes its gap tables per fold, `lpkt`/`hdkt` their time
        vocabularies. Silently keeping fold 0's file would hide that the model
        changed size underneath a per-fold metric spread."""
        with tempfile.TemporaryDirectory() as tmp:
            save_model_info_once(tmp, collect_model_info(Toy()), fold_id=0)

            bigger = Toy()
            bigger.head = nn.Linear(2, 9)
            path = save_model_info_once(tmp, collect_model_info(bigger), fold_id=3)

            self.assertIsNotNone(path)
            self.assertTrue(path.endswith("model_info_fold3.json"))
            self.assertEqual(
                sorted(p.name for p in Path(tmp).iterdir()),
                ["model_info.json", "model_info_fold3.json"],
            )


if __name__ == "__main__":
    unittest.main()
