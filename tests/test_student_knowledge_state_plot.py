import unittest

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch

from models.removed_model import a removed model
from research.removed_model.plot_student_knowledge_state import (
    CONTINUOUS_CMAPS,
    DIVERGING_CMAPS,
    heatmap_kwargs,
    validate_mastery,
)


class a removed modelStateTrajectoryTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.model = a removed model(
            num_q=12,
            num_c=6,
            emb_size=8,
            d_h=10,
            d_p=5,
            dropout=0.0,
        ).eval()
        self.q = torch.tensor([[1, 2, 3, 4]])
        self.c = torch.tensor(
            [[[0, 1], [1, -1], [2, 3], [3, -1]]]
        )
        self.r = torch.tensor([[1.0, 0.0, 1.0, 1.0]])

    def test_state_shape_alignment_and_bounds(self):
        with torch.inference_mode():
            outputs = self.model(self.q, self.c, self.r)
            trajectory = self.model.get_state_trajectory(self.q, self.c, self.r)
            score = self.model.score_concept_mastery(
                trajectory["student_point"], trajectory["student_radius"]
            )

        self.assertEqual(trajectory["student_point"].shape, (1, 4, 5))
        self.assertEqual(trajectory["student_radius"].shape, (1, 4))
        self.assertEqual(score["mastery"].shape, (1, 4, 6))
        self.assertTrue(torch.isfinite(score["mastery"]).all())
        self.assertTrue(torch.all(score["mastery"] >= 0.0))
        self.assertTrue(torch.all(score["mastery"] <= 1.0))
        self.assertTrue(
            torch.allclose(
                outputs["student_radius"],
                trajectory["student_radius"][:, :-1],
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["y"],
                trajectory["pre_probability"][:, 1:],
            )
        )
        self.assertTrue(
            torch.allclose(
                outputs["pred_base"],
                trajectory["pre_base_probability"][:, 1:],
            )
        )

    def test_prefix_invariance(self):
        with torch.inference_mode():
            full = self.model.get_state_trajectory(self.q, self.c, self.r)
            prefix = self.model.get_state_trajectory(
                self.q[:, :3], self.c[:, :3], self.r[:, :3]
            )
        self.assertTrue(
            torch.allclose(
                full["student_point"][:, :3],
                prefix["student_point"],
                atol=1e-6,
                rtol=1e-6,
            )
        )
        self.assertTrue(
            torch.allclose(
                full["student_radius"][:, :3],
                prefix["student_radius"],
                atol=1e-6,
                rtol=1e-6,
            )
        )

    def test_training_forward_backward_remains_finite(self):
        self.model.train()
        outputs = self.model(self.q, self.c, self.r)
        target = self.r[:, 1:]
        loss = torch.nn.functional.binary_cross_entropy(outputs["y"], target)
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in self.model.parameters()
            if parameter.grad is not None
        ]
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(gradients)
        self.assertTrue(all(torch.isfinite(gradient).all() for gradient in gradients))


class SeabornHeatmapConfigurationTest(unittest.TestCase):
    def test_continuous_colormaps_use_fixed_probability_range(self):
        for cmap in CONTINUOUS_CMAPS:
            kwargs = heatmap_kwargs(cmap)
            self.assertEqual(kwargs["vmin"], 0.0)
            self.assertEqual(kwargs["vmax"], 1.0)
            self.assertNotIn("center", kwargs)

    def test_diverging_colormaps_center_on_half_mastery(self):
        for cmap in DIVERGING_CMAPS:
            kwargs = heatmap_kwargs(cmap)
            self.assertEqual(kwargs["vmin"], 0.0)
            self.assertEqual(kwargs["vmax"], 1.0)
            self.assertEqual(kwargs["center"], 0.5)

    def test_all_supported_colormaps_render_with_seaborn(self):
        data = np.asarray([[0.0, 0.5, 1.0]])
        validate_mastery(data.T)
        for cmap in CONTINUOUS_CMAPS | DIVERGING_CMAPS:
            fig, ax = plt.subplots(figsize=(3, 1))
            sns.heatmap(data, ax=ax, cbar=False, **heatmap_kwargs(cmap))
            plt.close(fig)


if __name__ == "__main__":
    unittest.main()
