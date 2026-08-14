"""Unit tests for AEMA exact top-k fast-mask selection."""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema import AEMAUpdater, _count_pseudo_boxes


class MultiParamDetector(nn.Module):
    """Detector with multiple parameters of different sizes for top-k testing."""

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Parameter(torch.zeros(100))
        self.layer2 = nn.Parameter(torch.zeros(50))
        self.register_buffer("float_buffer", torch.tensor([0.0], dtype=torch.float32))
        self.register_buffer("int_buffer", torch.tensor([0], dtype=torch.long))

    def forward(self, images, targets=None):
        if targets is None:
            return [
                {
                    "boxes": torch.zeros(0, 4, device=images.device),
                    "labels": torch.zeros(0, dtype=torch.long, device=images.device),
                    "scores": torch.zeros(0, device=images.device),
                }
                for _ in range(images.shape[0])
            ]
        return {"classification": (self.layer1 * images.sum()).sum() +
                (self.layer2 * images.sum()).sum()}


class AEMATopKTest(unittest.TestCase):
    def test_exact_top_k_10_percent(self):
        """With 150 eligible elements, top_ratio=0.1 should select exactly 15."""
        teacher = MultiParamDetector()
        student = MultiParamDetector()
        for p in teacher.parameters():
            p.requires_grad_(False)

        updater = AEMAUpdater(
            fast_alpha=0.5, slow_alpha=0.9,
            top_ratio=0.10, update_interval=1,
        )

        pseudo = [{
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "labels": torch.tensor([0], dtype=torch.long),
            "scores": torch.tensor([0.9]),
        }]

        log = updater.accumulate_and_maybe_update(
            teacher=teacher, student=student,
            images=torch.ones(1, 100),
            pseudo_targets=pseudo,
        )

        self.assertEqual(log["aema_updated"], 1.0)
        self.assertEqual(log["aema_fast_count"], 15)
        self.assertEqual(log["aema_total_count"], 150)
        self.assertAlmostEqual(log["aema_fast_fraction"], 15 / 150, places=5)

    def test_many_equal_gradients(self):
        """When many gradients are equal (e.g. all zeros), top-k must still
        select exactly k elements, not more."""
        teacher = MultiParamDetector()
        student = MultiParamDetector()
        # Make all weights equal so gradients will be similar
        with torch.no_grad():
            for p in teacher.parameters():
                p.fill_(0.0)
            for p in student.parameters():
                p.fill_(1.0)
        for p in teacher.parameters():
            p.requires_grad_(False)

        updater = AEMAUpdater(
            fast_alpha=0.5, slow_alpha=0.9,
            top_ratio=0.10, update_interval=1,
        )

        pseudo = [{
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "labels": torch.tensor([0], dtype=torch.long),
            "scores": torch.tensor([0.9]),
        }]

        log = updater.accumulate_and_maybe_update(
            teacher=teacher, student=student,
            images=torch.ones(1, 100),
            pseudo_targets=pseudo,
        )

        self.assertEqual(log["aema_updated"], 1.0)
        # With 150 elements and top_ratio=0.1, should select exactly 15
        self.assertEqual(int(log["aema_fast_count"]), 15)

    def test_many_zeros_gradient(self):
        """When all gradients are zero, top-k still selects exactly k by index."""
        teacher = MultiParamDetector()
        student = MultiParamDetector()
        with torch.no_grad():
            for p in teacher.parameters():
                p.fill_(0.0)
            for p in student.parameters():
                p.fill_(0.0)
        for p in teacher.parameters():
            p.requires_grad_(False)

        updater = AEMAUpdater(
            fast_alpha=0.5, slow_alpha=0.9,
            top_ratio=0.10, update_interval=1,
        )

        # Use non-empty pseudo to trigger gradient accumulation
        pseudo = [{
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "labels": torch.tensor([0], dtype=torch.long),
            "scores": torch.tensor([0.9]),
        }]

        log = updater.accumulate_and_maybe_update(
            teacher=teacher, student=student,
            images=torch.ones(1, 100),
            pseudo_targets=pseudo,
        )

        self.assertEqual(log["aema_updated"], 1.0)
        self.assertEqual(int(log["aema_fast_count"]), 15)

    def test_dtype_and_device_preserved(self):
        """AEMA must preserve dtype and device of parameters."""
        teacher = MultiParamDetector()
        student = MultiParamDetector()
        for p in teacher.parameters():
            p.requires_grad_(False)

        updater = AEMAUpdater(
            fast_alpha=0.997, slow_alpha=0.9996,
            top_ratio=0.10, update_interval=1,
        )

        pseudo = [{
            "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "labels": torch.tensor([0], dtype=torch.long),
            "scores": torch.tensor([0.9]),
        }]

        updater.accumulate_and_maybe_update(
            teacher=teacher, student=student,
            images=torch.ones(1, 100),
            pseudo_targets=pseudo,
        )

        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            self.assertEqual(t_p.dtype, s_p.dtype)
            self.assertEqual(t_p.device, s_p.device)


if __name__ == "__main__":
    unittest.main()
