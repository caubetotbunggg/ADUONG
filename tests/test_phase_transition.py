"""Unit tests for phase transitions and domain-specialized Phase 3."""

import os
import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scheduler import Phase
from config import TrainingConfig


class PhaseTransitionTest(unittest.TestCase):
    def _make_trainer(self):
        """Create a minimal trainer with mock models for transition testing."""
        from trainer import CurriculumDomainAdaptationTrainer
        from torch.utils.data import DataLoader

        student = nn.Linear(3, 3)
        rgb_teacher = nn.Linear(3, 3)
        ir_teacher = nn.Linear(3, 3)

        config = TrainingConfig(device="cpu")
        config.curriculum.phase1_end = 10
        config.curriculum.phase2_end = 20
        config.curriculum.phase3_end = 30

        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        # Mock data loaders
        rgb_loader = MagicMock(spec=DataLoader)
        ir_loader = MagicMock(spec=DataLoader)

        trainer = CurriculumDomainAdaptationTrainer(
            student=student,
            rgb_teacher=rgb_teacher,
            ir_teacher=ir_teacher,
            optimizer=optimizer,
            config=config,
            rgb_loader=rgb_loader,
            ir_loader=ir_loader,
            phase1_best_path="/nonexistent/phase1.pt",
            phase2_best_path="/nonexistent/phase2.pt",
        )
        return trainer

    def test_phase2_to_phase3_copies_student_to_ir_teacher(self):
        """Phase2 → Phase3 should copy student weights into IR teacher and reset ir_aema."""
        trainer = self._make_trainer()

        # Modify student weights so we can detect the copy
        with torch.no_grad():
            for p in trainer.student.parameters():
                p.fill_(42.0)
            for p in trainer.ir_teacher.parameters():
                p.fill_(0.0)
            for p in trainer.rgb_teacher.parameters():
                p.fill_(99.0)

        # Trigger transition
        trainer._on_phase_transition(Phase.PHASE2_RGB_MID, Phase.PHASE3_MID_IR)

        # IR teacher should now have student's weights
        for p in trainer.ir_teacher.parameters():
            self.assertTrue(torch.allclose(p, torch.full_like(p, 42.0)))

        # RGB teacher should NOT be reset
        for p in trainer.rgb_teacher.parameters():
            self.assertTrue(torch.allclose(p, torch.full_like(p, 99.0)))

        # IR AEMA should be reset
        self.assertEqual(trainer.ir_aema.steps_since_update, 0)
        self.assertEqual(len(trainer.ir_aema.grad_accum), 0)

    def test_phase1_to_phase2_copies_student_to_both(self):
        """Phase1 → Phase2 should copy student weights into BOTH teachers."""
        trainer = self._make_trainer()

        with torch.no_grad():
            for p in trainer.student.parameters():
                p.fill_(77.0)
            for p in trainer.rgb_teacher.parameters():
                p.fill_(0.0)
            for p in trainer.ir_teacher.parameters():
                p.fill_(0.0)

        trainer._on_phase_transition(Phase.PHASE1_RGB_WARMUP, Phase.PHASE2_RGB_MID)

        for p in trainer.rgb_teacher.parameters():
            self.assertTrue(torch.allclose(p, torch.full_like(p, 77.0)))
        for p in trainer.ir_teacher.parameters():
            self.assertTrue(torch.allclose(p, torch.full_like(p, 77.0)))


class TrackingDetector(nn.Module):
    """Detector that records input batch sizes and returns valid predictions."""

    def __init__(self, num_classes=3):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_classes))
        self.seen_batch_sizes: list = []

    def forward(self, images, targets=None):
        self.seen_batch_sizes.append(images.shape[0])
        if targets is None:
            return [
                {
                    "boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
                    "labels": torch.tensor([0], dtype=torch.long),
                    "scores": torch.tensor([0.9]),
                }
                for _ in range(images.shape[0])
            ]
        # Return a dummy loss that has a gradient
        loss = (self.weight * images.reshape(images.shape[0], -1)[:, :self.weight.numel()].sum(dim=0)).sum()
        return {"classification": loss}


class DomainSpecializedPhase3Test(unittest.TestCase):
    def test_rgb_teacher_receives_mid_only(self):
        """In Phase 3, RGB teacher should only receive MID images for pseudo generation,
        and IR teacher should only receive IR images."""
        from trainer import CurriculumDomainAdaptationTrainer
        from torch.utils.data import DataLoader
        from batch_types import MidIrBatch

        student = TrackingDetector()
        rgb_teacher = TrackingDetector()
        ir_teacher = TrackingDetector()

        config = TrainingConfig(device="cpu")
        config.curriculum.phase1_end = 10
        config.curriculum.phase2_end = 20
        config.curriculum.phase3_end = 30
        config.loss.p3_rgb_teacher_weight = 0.5
        config.loss.p3_ir_teacher_weight = 0.5
        # Disable AEMA to avoid needing student predictions
        config.ablation.disable_aema = True

        optimizer = torch.optim.SGD(student.parameters(), lr=0.01)

        rgb_loader = MagicMock(spec=DataLoader)
        ir_loader = MagicMock(spec=DataLoader)

        trainer = CurriculumDomainAdaptationTrainer(
            student=student,
            rgb_teacher=rgb_teacher,
            ir_teacher=ir_teacher,
            optimizer=optimizer,
            config=config,
            rgb_loader=rgb_loader,
            ir_loader=ir_loader,
        )

        # Mock _next_mid_ir to return a controlled batch
        mid_imgs = torch.randn(2, 3, 8, 8)
        ir_imgs = torch.randn(2, 3, 8, 8)
        mid_targets = [
            {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
             "labels": torch.tensor([0], dtype=torch.long)},
            {"boxes": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
             "labels": torch.tensor([1], dtype=torch.long)},
        ]
        mock_batch = MidIrBatch(
            mid_images=mid_imgs,
            ir_images=ir_imgs,
            mid_targets=mid_targets,
            n_mid=2,
            n_ir=2,
        )
        trainer._next_mid_ir = MagicMock(return_value=mock_batch)

        # Run the step
        with patch.object(trainer, '_clip_and_step', return_value=0.0):
            trainer.train_mid_ir_step()

        # RGB teacher should have been called with MID images only (2 images)
        rgb_inference_calls = [s for s in rgb_teacher.seen_batch_sizes if s == 2]
        self.assertGreater(len(rgb_inference_calls), 0,
                           f"RGB teacher should see 2 MID images, got: {rgb_teacher.seen_batch_sizes}")

        # IR teacher should have been called with IR images only (2 images)
        ir_inference_calls = [s for s in ir_teacher.seen_batch_sizes if s == 2]
        self.assertGreater(len(ir_inference_calls), 0,
                           f"IR teacher should see 2 IR images, got: {ir_teacher.seen_batch_sizes}")

        # Verify RGB teacher was NOT called with the full batch (4 images)
        self.assertNotIn(4, rgb_teacher.seen_batch_sizes,
                         "RGB teacher should not see the full mixed batch")

        # Verify IR teacher was NOT called with the full batch (4 images)
        self.assertNotIn(4, ir_teacher.seen_batch_sizes,
                         "IR teacher should not see the full mixed batch")


if __name__ == "__main__":
    unittest.main()
