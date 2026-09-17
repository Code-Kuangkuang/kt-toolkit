import unittest

import torch
from torch.utils.data import Dataset

from datasets.label_noise import apply_train_label_flip


class _ToyKTDataset(Dataset):
    def __init__(self, include_history=False, include_difficulty=False):
        responses = torch.tensor(
            [[1.0, 0.0, 1.0, -1.0], [0.0, 1.0, 0.0, 1.0]]
        )
        self.dori = {
            "rseqs": responses,
            "masks": torch.tensor(
                [[True, True, False], [True, True, True]]
            ),
            "smasks": torch.tensor(
                [[True, True, False], [True, True, True]]
            ),
        }
        if include_history:
            self.dori["historycorrs"] = torch.zeros_like(responses)
        if include_difficulty:
            self.dori["sdseqs"] = torch.ones_like(responses, dtype=torch.long)

    def __len__(self):
        return len(self.dori["rseqs"])

    def __getitem__(self, index):
        full = torch.where(
            self.dori["rseqs"][index] < 0,
            torch.zeros_like(self.dori["rseqs"][index]),
            self.dori["rseqs"][index],
        )
        mask = self.dori["masks"][index]
        item = {
            "rseqs": full[:-1] * mask,
            "shft_rseqs": full[1:] * mask,
            "masks": mask,
            "smasks": self.dori["smasks"][index],
        }
        if "historycorrs" in self.dori:
            history = self.dori["historycorrs"][index]
            item["historycorrs"] = history[:-1] * mask
            item["shft_historycorrs"] = history[1:] * mask
        return item


class TrainLabelFlipTest(unittest.TestCase):
    def test_exact_deterministic_flips_and_padding_preservation(self):
        clean = _ToyKTDataset()
        original = clean.dori["rseqs"].clone()

        noisy_a = apply_train_label_flip(clean, ratio=0.5, seed=123)
        noisy_b = apply_train_label_flip(clean, ratio=0.5, seed=123)

        info = noisy_a.label_flip_info
        self.assertEqual(info["eligible_count"], 7)
        self.assertEqual(info["flipped_count"], 4)
        self.assertAlmostEqual(info["actual_ratio"], 4 / 7)
        self.assertEqual(info["mask_sha256"], noisy_b.label_flip_info["mask_sha256"])
        torch.testing.assert_close(noisy_a.dori["rseqs"], noisy_b.dori["rseqs"])
        torch.testing.assert_close(clean.dori["rseqs"], original)
        self.assertEqual(noisy_a.dori["rseqs"][0, 3].item(), -1.0)

    def test_current_and_shifted_views_share_each_interaction_flip(self):
        noisy = apply_train_label_flip(_ToyKTDataset(), ratio=1.0, seed=7)
        item = noisy[1]

        # Full position 1 appears as shifted position 0 and current position 1.
        self.assertEqual(item["shft_rseqs"][0].item(), 0.0)
        self.assertEqual(item["rseqs"][1].item(), 0.0)

    def test_validation_source_can_remain_clean(self):
        train_source = _ToyKTDataset()
        valid_source = _ToyKTDataset()
        valid_before = valid_source.dori["rseqs"].clone()

        apply_train_label_flip(train_source, ratio=1.0, seed=9)

        torch.testing.assert_close(valid_source.dori["rseqs"], valid_before)

    def test_response_history_is_recomputed_from_flipped_labels(self):
        noisy = apply_train_label_flip(
            _ToyKTDataset(include_history=True), ratio=1.0, seed=3
        )
        expected = torch.tensor([0.0, 0.5, 1.0 / 3.0, 0.0])
        torch.testing.assert_close(noisy.dori["historycorrs"][0], expected)

    def test_rejects_invalid_ratio_and_clean_label_difficulty_features(self):
        with self.assertRaises(ValueError):
            apply_train_label_flip(_ToyKTDataset(), ratio=1.01, seed=1)
        with self.assertRaisesRegex(ValueError, "difficulty"):
            apply_train_label_flip(
                _ToyKTDataset(include_difficulty=True), ratio=0.1, seed=1
            )


if __name__ == "__main__":
    unittest.main()
