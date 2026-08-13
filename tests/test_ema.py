import os
import sys
import unittest

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ema import AEMAUpdater, ema_update


class TinyDetector(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(values, dtype=torch.float32))
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
        return {"classification": (self.weight * images.sum(dim=0)).sum()}


class AEMAUpdaterTest(unittest.TestCase):
    def test_top_ratio_uses_fast_alpha_and_rest_slow_alpha(self):
        teacher = TinyDetector([0.0, 0.0, 0.0, 0.0])
        student = TinyDetector([10.0, 10.0, 10.0, 10.0])
        for param in teacher.parameters():
            param.requires_grad_(False)

        updater = AEMAUpdater(
            fast_alpha=0.5,
            slow_alpha=0.9,
            top_ratio=0.5,
            update_interval=1,
        )
        pseudo = [{
            "boxes": torch.zeros(0, 4),
            "labels": torch.zeros(0, dtype=torch.long),
            "scores": torch.zeros(0),
        }]
        log = updater.accumulate_and_maybe_update(
            teacher=teacher,
            student=student,
            images=torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            pseudo_targets=pseudo,
        )

        self.assertEqual(log["aema_updated"], 1.0)
        self.assertAlmostEqual(log["aema_fast_fraction"], 0.5)
        torch.testing.assert_close(
            teacher.weight.detach(),
            torch.tensor([1.0, 1.0, 5.0, 5.0]),
        )
        self.assertEqual(updater.steps_since_update, 0)
        self.assertEqual(updater.grad_accum, {})
        self.assertFalse(next(teacher.parameters()).requires_grad)

    def test_ema_copies_integer_buffers(self):
        teacher = TinyDetector([0.0])
        student = TinyDetector([10.0])
        teacher.float_buffer.fill_(0.0)
        student.float_buffer.fill_(10.0)
        teacher.int_buffer.fill_(1)
        student.int_buffer.fill_(7)

        ema_update(teacher, student, alpha=0.5)

        self.assertEqual(float(teacher.float_buffer.item()), 5.0)
        self.assertEqual(int(teacher.int_buffer.item()), 7)


if __name__ == "__main__":
    unittest.main()
