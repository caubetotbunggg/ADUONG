"""Unit tests for AAT statistical domain-match attack."""

import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aat import StatisticalDomainAttacker, AATPseudoLabelGenerator, _merge_two_pseudo


class StatisticalDomainAttackerTest(unittest.TestCase):
    def test_attack_without_ir_stats_returns_unchanged(self):
        """Before IR stats are set, attack should return images unchanged."""
        attacker = StatisticalDomainAttacker(epsilon=0.02)
        images = torch.randn(2, 3, 32, 32)
        attacked = attacker.attack(images)
        torch.testing.assert_close(attacked, images)

    def test_attack_changes_images_toward_ir(self):
        """After IR stats are set, attack should perturb images."""
        attacker = StatisticalDomainAttacker(epsilon=0.05)
        # IR: high mean, low std
        ir_images = torch.ones(4, 3, 32, 32) * 0.7 + torch.randn(4, 3, 32, 32) * 0.1
        attacker.update_ir_stats(ir_images)

        # RGB: different stats (lower mean)
        rgb_images = torch.ones(2, 3, 32, 32) * 0.3 + torch.randn(2, 3, 32, 32) * 0.2
        attacked = attacker.attack(rgb_images)

        # Attacked images should be different from original
        self.assertFalse(torch.allclose(attacked, rgb_images, atol=1e-6))
        # Attacked images should be closer to IR mean
        rgb_dist = (rgb_images.mean() - attacker._ir_mean.mean()).abs().item()
        attacked_dist = (attacked.mean() - attacker._ir_mean.mean()).abs().item()
        self.assertLess(attacked_dist, rgb_dist,
                        "Attacked images should be closer to IR mean")

    def test_attack_respects_clamp(self):
        """Attacked images should be clamped to valid range."""
        attacker = StatisticalDomainAttacker(epsilon=1.0, clamp_min=0.0, clamp_max=1.0)
        ir_images = torch.ones(4, 3, 32, 32)
        attacker.update_ir_stats(ir_images)

        # Very different images
        images = torch.zeros(2, 3, 32, 32)
        attacked = attacker.attack(images)
        self.assertGreaterEqual(attacked.min().item(), 0.0)
        self.assertLessEqual(attacked.max().item(), 1.0)

    def test_ir_stats_running_update(self):
        """IR stats should update with momentum (EMA)."""
        attacker = StatisticalDomainAttacker(epsilon=0.02)
        ir1 = torch.ones(4, 3, 32, 32) * 0.5
        attacker.update_ir_stats(ir1, momentum=0.9)
        first_mean = attacker._ir_mean.clone()

        ir2 = torch.ones(4, 3, 32, 32) * 0.9
        attacker.update_ir_stats(ir2, momentum=0.9)
        second_mean = attacker._ir_mean.clone()

        # Mean should have shifted toward 0.9 but not fully
        self.assertTrue(torch.all(second_mean > first_mean),
                        "IR mean should increase after seeing higher-mean IR batch")
        self.assertTrue(torch.all(second_mean < 0.9),
                        "IR mean should not jump fully to new batch (momentum)")


class MergeTwoPseudoTest(unittest.TestCase):
    def _make_pseudo(self, boxes, labels, scores=None):
        if scores is None:
            scores = torch.ones(len(labels))
        return {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.long),
            "scores": torch.tensor(scores, dtype=torch.float32),
        }

    def test_merge_adds_non_overlapping(self):
        """Secondary boxes that don't overlap primary should be added."""
        primary = [self._make_pseudo([[10, 10, 50, 50]], [0])]
        secondary = [self._make_pseudo([[100, 100, 150, 150]], [1])]
        merged = _merge_two_pseudo(primary, secondary, iou_threshold=0.5)
        self.assertEqual(merged[0]["boxes"].shape[0], 2)

    def test_merge_skips_overlapping(self):
        """Secondary boxes that overlap primary (IoU >= threshold) should be skipped."""
        primary = [self._make_pseudo([[10, 10, 50, 50]], [0])]
        # Same class, nearly identical box
        secondary = [self._make_pseudo([[11, 11, 51, 51]], [0])]
        merged = _merge_two_pseudo(primary, secondary, iou_threshold=0.5)
        self.assertEqual(merged[0]["boxes"].shape[0], 1)

    def test_merge_different_class_same_location_kept(self):
        """Different class at same location should be kept (not deduped)."""
        primary = [self._make_pseudo([[10, 10, 50, 50]], [0])]
        secondary = [self._make_pseudo([[10, 10, 50, 50]], [1])]
        merged = _merge_two_pseudo(primary, secondary, iou_threshold=0.5)
        self.assertEqual(merged[0]["boxes"].shape[0], 2)

    def test_merge_empty_primary(self):
        """If primary is empty, all secondary boxes should be kept."""
        primary = [self._make_pseudo([], [])]
        secondary = [self._make_pseudo([[10, 10, 50, 50], [60, 60, 100, 100]], [0, 1])]
        merged = _merge_two_pseudo(primary, secondary, iou_threshold=0.5)
        self.assertEqual(merged[0]["boxes"].shape[0], 2)


if __name__ == "__main__":
    unittest.main()
