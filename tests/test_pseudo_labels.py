"""Unit tests for empty pseudo-label handling."""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema import AEMAUpdater, _count_pseudo_boxes
from losses import select_nonempty_pseudo_targets, compute_ir_loss, LossConfig


class DummyDetector(nn.Module):
    """Simple detector that counts forward calls in training mode."""

    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(3))
        self.forward_count = 0

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
        self.forward_count += 1
        # Sum over all non-batch dims to get per-image features matching weight
        feat = images.reshape(images.shape[0], -1)
        return {"classification": (self.weight * feat[:, :3]).sum()}


def _empty_target(device=torch.device("cpu")):
    return {
        "boxes": torch.zeros(0, 4, device=device),
        "labels": torch.zeros(0, dtype=torch.long, device=device),
        "scores": torch.zeros(0, device=device),
    }


def _valid_target(device=torch.device("cpu")):
    return {
        "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]], device=device),
        "labels": torch.tensor([0], dtype=torch.long, device=device),
        "scores": torch.tensor([0.9], device=device),
    }


class SelectNonemptyPseudoTest(unittest.TestCase):
    def test_mixed_empty_and_valid(self):
        """Only images with >= 1 pseudo box should be kept."""
        images = torch.randn(3, 3, 8, 8)
        targets = [_empty_target(), _valid_target(), _empty_target()]

        filt_imgs, filt_tgts, kept = select_nonempty_pseudo_targets(images, targets)

        self.assertEqual(filt_imgs.shape[0], 1)
        self.assertEqual(len(filt_tgts), 1)
        self.assertEqual(kept, [1])
        self.assertEqual(filt_tgts[0]["boxes"].numel(), 4)

    def test_all_empty(self):
        """When all targets are empty, return empty batch."""
        images = torch.randn(2, 3, 8, 8)
        targets = [_empty_target(), _empty_target()]

        filt_imgs, filt_tgts, kept = select_nonempty_pseudo_targets(images, targets)

        self.assertEqual(filt_imgs.shape[0], 0)
        self.assertEqual(len(filt_tgts), 0)
        self.assertEqual(kept, [])

    def test_all_valid(self):
        """When all targets have boxes, keep everything."""
        images = torch.randn(2, 3, 8, 8)
        targets = [_valid_target(), _valid_target()]

        filt_imgs, filt_tgts, kept = select_nonempty_pseudo_targets(images, targets)

        self.assertEqual(filt_imgs.shape[0], 2)
        self.assertEqual(len(filt_tgts), 2)
        self.assertEqual(kept, [0, 1])


class EmptyPseudoLossTest(unittest.TestCase):
    def test_all_empty_pseudo_loss_is_zero(self):
        """When all pseudo targets are empty, loss = 0 and no student forward."""
        student = DummyDetector()
        images = torch.randn(2, 3, 8, 8)
        pseudo = [_empty_target(), _empty_target()]

        loss, log = compute_ir_loss(
            student=student,
            ir_images=images,
            ir_teacher=None,
            config=LossConfig(),
            ir_pseudo_targets=pseudo,
            skip_empty_pseudo=True,
        )

        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(student.forward_count, 0)

    def test_mixed_empty_pseudo_only_valid_contributes(self):
        """Only valid pseudo images should contribute to loss."""
        student = DummyDetector()
        images = torch.randn(3, 3, 8, 8)
        pseudo = [_empty_target(), _valid_target(), _empty_target()]

        loss, log = compute_ir_loss(
            student=student,
            ir_images=images,
            ir_teacher=None,
            config=LossConfig(),
            ir_pseudo_targets=pseudo,
            skip_empty_pseudo=True,
        )

        # Student should be called once (only on the 1 valid image)
        self.assertEqual(student.forward_count, 1)
        self.assertNotEqual(loss.item(), 0.0)


class AEMAEmptyPseudoTest(unittest.TestCase):
    def test_no_gradient_accumulated_when_all_empty(self):
        """AEMA must not accumulate gradient importance when all pseudo are empty."""
        teacher = DummyDetector()
        student = DummyDetector()
        for p in teacher.parameters():
            p.requires_grad_(False)

        updater = AEMAUpdater(
            fast_alpha=0.5, slow_alpha=0.9,
            top_ratio=0.10, update_interval=1,
        )

        pseudo = [_empty_target(), _empty_target()]
        images = torch.randn(2, 3)

        log = updater.accumulate_and_maybe_update(
            teacher=teacher, student=student,
            images=images,
            pseudo_targets=pseudo,
        )

        # No gradient should have been accumulated
        self.assertEqual(len(updater.grad_accum), 0)
        self.assertEqual(log["aema_importance_loss"], 0.0)
        # Teacher forward should not have been called for importance
        self.assertEqual(teacher.forward_count, 0)

    def test_partially_empty_batch(self):
        """AEMA should only use non-empty pseudo images for gradient accumulation."""
        teacher = DummyDetector()
        student = DummyDetector()
        for p in teacher.parameters():
            p.requires_grad_(False)

        updater = AEMAUpdater(
            fast_alpha=0.5, slow_alpha=0.9,
            top_ratio=0.10, update_interval=1,
        )

        pseudo = [_empty_target(), _valid_target(), _empty_target()]
        images = torch.randn(3, 3)

        log = updater.accumulate_and_maybe_update(
            teacher=teacher, student=student,
            images=images,
            pseudo_targets=pseudo,
        )

        # Importance loss should be non-zero (gradient was accumulated and applied)
        self.assertNotEqual(log["aema_importance_loss"], 0.0)
        # Teacher forward should have been called for importance
        self.assertGreater(teacher.forward_count, 0)


if __name__ == "__main__":
    unittest.main()
