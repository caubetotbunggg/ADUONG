"""
AAT — Adversarial Attacked Teacher (Statistical Domain Match).

Instead of generic FGSM/PGD (which perturbs in random directions), this module
attacks teacher inputs toward the IR domain's statistical distribution
(mean/std of pixel intensities).  The attacked teacher sees "IR-like" images
and may detect objects it misses on clean images.

Flow:
  1. Teacher predicts on clean images → clean_pseudo
  2. Compute statistical distance to IR reference distribution
  3. Perturb images to reduce that distance (push toward IR style)
  4. Teacher predicts on attacked images → attacked_pseudo
  5. Merge clean + attacked via IoU dedup → final pseudo

No discriminator or bounding boxes needed — works with --adv_weight 0.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from losses import filter_pseudo_labels, ThreshType

logger = logging.getLogger(__name__)


def _box_iou_single(box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """IoU between one box [4] and a set of boxes [N, 4]. Returns [N]."""
    x1 = torch.max(box1[0], boxes2[:, 0])
    y1 = torch.max(box1[1], boxes2[:, 1])
    x2 = torch.min(box1[2], boxes2[:, 2])
    y2 = torch.min(box1[3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def _merge_two_pseudo(
    primary: List[Dict[str, torch.Tensor]],
    secondary: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
) -> List[Dict[str, torch.Tensor]]:
    """
    Merge two pseudo-label lists.  Primary boxes are kept; secondary boxes
    that don't overlap any primary box of the same class (IoU < threshold)
    are added.
    """
    merged: List[Dict[str, torch.Tensor]] = []
    for p, s in zip(primary, secondary):
        p_boxes = p["boxes"]
        p_labels = p["labels"]
        p_scores = p.get("scores", torch.zeros(p_boxes.shape[0], device=p_boxes.device))

        s_boxes = s["boxes"]
        s_labels = s["labels"]
        s_scores = s.get("scores", torch.zeros(s_boxes.shape[0], device=s_boxes.device))

        if p_boxes.numel() == 0:
            merged.append({"boxes": s_boxes, "labels": s_labels, "scores": s_scores})
            continue
        if s_boxes.numel() == 0:
            merged.append({"boxes": p_boxes, "labels": p_labels, "scores": p_scores})
            continue

        new_boxes: List[torch.Tensor] = [p_boxes]
        new_labels: List[torch.Tensor] = [p_labels]
        new_scores: List[torch.Tensor] = [p_scores]

        for i in range(s_boxes.shape[0]):
            same_class = p_labels == s_labels[i]
            if same_class.any():
                ious = _box_iou_single(s_boxes[i], p_boxes[same_class])
                max_iou = ious.max().item()
            else:
                max_iou = 0.0
            if max_iou < iou_threshold:
                new_boxes.append(s_boxes[i:i + 1])
                new_labels.append(s_labels[i:i + 1])
                new_scores.append(s_scores[i:i + 1])

        merged.append({
            "boxes": torch.cat(new_boxes, dim=0),
            "labels": torch.cat(new_labels, dim=0),
            "scores": torch.cat(new_scores, dim=0),
        })
    return merged


class StatisticalDomainAttacker:
    """
    Perturb images toward the IR domain's statistical distribution.

    The attack minimises the distance between the input image's per-channel
    mean/std and the IR reference distribution's mean/std.  This pushes
    RGB/MID images to look more "IR-like" without needing a discriminator
    or bounding boxes.

    Usage:
        attacker = StatisticalDomainAttacker(epsilon=0.02)
        attacker.update_ir_stats(ir_images)     # call periodically with real IR
        attacked = attacker.attack(images)       # perturb toward IR stats
    """

    def __init__(
        self,
        epsilon: float = 0.02,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        self.epsilon = epsilon
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        # IR statistics (running estimate, updated with real IR batches)
        self._ir_mean: Optional[torch.Tensor] = None  # [C, 1, 1]
        self._ir_std: Optional[torch.Tensor] = None   # [C, 1, 1]

    def update_ir_stats(self, ir_images: torch.Tensor, momentum: float = 0.9) -> None:
        """
        Update running IR statistics from a batch of real IR images.

        Args:
            ir_images: [B, C, H, W] real IR images
            momentum:  EMA momentum for smoothing (0.9 = slow update)
        """
        with torch.no_grad():
            batch_mean = ir_images.mean(dim=[0, 2, 3], keepdim=True)  # [1, C, 1, 1]
            batch_std = ir_images.std(dim=[0, 2, 3], keepdim=True)    # [1, C, 1, 1]

            if self._ir_mean is None:
                self._ir_mean = batch_mean.detach().clone()
                self._ir_std = batch_std.detach().clone()
            else:
                self._ir_mean = momentum * self._ir_mean + (1 - momentum) * batch_mean
                self._ir_std = momentum * self._ir_std + (1 - momentum) * batch_std

    def attack(self, images: torch.Tensor) -> torch.Tensor:
        """
        Perturb images toward IR statistical distribution.

        Args:
            images: [B, C, H, W] input images (RGB, MID, or IR)

        Returns:
            attacked_images: [B, C, H, W] perturbed toward IR stats
        """
        if self._ir_mean is None or self._ir_std is None:
            # No IR stats yet — return unchanged
            return images

        ir_mean = self._ir_mean.to(device=images.device, dtype=images.dtype)
        ir_std = self._ir_std.to(device=images.device, dtype=images.dtype)

        images_grad = images.detach().clone().requires_grad_(True)

        # Per-image statistics
        img_mean = images_grad.mean(dim=[2, 3], keepdim=True)  # [B, C, 1, 1]
        img_std = images_grad.std(dim=[2, 3], keepdim=True)    # [B, C, 1, 1]

        # Loss: distance to IR distribution (minimise → move toward IR)
        mean_loss = ((img_mean - ir_mean) ** 2).sum()
        std_loss = ((img_std - ir_std) ** 2).sum()
        loss = mean_loss + std_loss

        grad = torch.autograd.grad(loss, images_grad)[0]
        # Move toward IR (minimise distance → subtract gradient)
        attacked = (images - self.epsilon * grad.sign()).detach()
        attacked = attacked.clamp(self.clamp_min, self.clamp_max)
        return attacked



class DiscriminatorDomainAttacker:
    """
    Perturb images toward the IR domain using discriminator gradient.

    Uses the domain discriminator (DANN-style) to guide the attack:
    the gradient of the discriminator's domain classification loss w.r.t.
    input images indicates the direction to push images toward IR.

    Requires a trained discriminator (--adv_weight > 0).

    Usage:
        attacker = DiscriminatorDomainAttacker(
            backbone_fn=student.get_backbone_features,
            discriminator=disc_ir,
            epsilon=0.02,
        )
        attacked = attacker.attack(images, target_domain=1)  # 1 = IR
    """

    def __init__(
        self,
        backbone_fn,
        discriminator: nn.Module,
        epsilon: float = 0.02,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
    ) -> None:
        self.backbone_fn = backbone_fn
        self.discriminator = discriminator
        self.epsilon = epsilon
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def attack(self, images: torch.Tensor, target_domain: int = 1) -> torch.Tensor:
        """
        Perturb images so discriminator classifies them as target_domain.

        Args:
            images:        [B, C, H, W] input images
            target_domain:  0 = MID (source), 1 = IR (target)

        Returns:
            attacked_images: [B, C, H, W] perturbed toward target domain
        """
        images_grad = images.detach().clone().requires_grad_(True)

        # Extract backbone features (gradient flows to images)
        features = self.backbone_fn(images_grad)

        # Discriminator classifies domain
        logits = self.discriminator(features)  # [B, 2]

        # Loss: make discriminator think these are target domain
        target_labels = torch.full(
            (images.shape[0],), target_domain, dtype=torch.long,
            device=images.device,
        )
        loss = F.cross_entropy(logits, target_labels)

        grad = torch.autograd.grad(loss, images_grad)[0]
        # Move toward target domain (minimise loss → subtract gradient)
        attacked = (images - self.epsilon * grad.sign()).detach()
        attacked = attacked.clamp(self.clamp_min, self.clamp_max)
        return attacked

    def update_ir_stats(self, ir_images: torch.Tensor, momentum: float = 0.9) -> None:
        """No-op for discriminator mode (uses discriminator directly)."""
        pass

class AATPseudoLabelGenerator:
    """
    Generate pseudo-labels using Adversarial Attacked Teacher.

    Supports two attack modes:
      - "statistical": perturb toward IR mean/std (no discriminator needed)
      - "discriminator": use DANN discriminator gradient to push toward IR

    Both modes merge clean + attacked teacher predictions via IoU dedup.
    """

    def __init__(
        self,
        epsilon: float = 0.02,
        merge_iou_threshold: float = 0.5,
        clamp_min: float = 0.0,
        clamp_max: float = 1.0,
        mode: str = "statistical",
        backbone_fn=None,
        discriminator: Optional[nn.Module] = None,
        target_domain: int = 1,
    ) -> None:
        self.mode = mode
        self.merge_iou_threshold = merge_iou_threshold
        self.target_domain = target_domain

        if mode == "discriminator":
            assert backbone_fn is not None, "discriminator mode requires backbone_fn"
            assert discriminator is not None, "discriminator mode requires discriminator"
            self.attacker = DiscriminatorDomainAttacker(
                backbone_fn=backbone_fn,
                discriminator=discriminator,
                epsilon=epsilon,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )
        else:
            self.attacker = StatisticalDomainAttacker(
                epsilon=epsilon,
                clamp_min=clamp_min,
                clamp_max=clamp_max,
            )

    def update_ir_stats(self, ir_images: torch.Tensor, momentum: float = 0.9) -> None:
        """Update IR reference statistics (statistical mode only)."""
        self.attacker.update_ir_stats(ir_images, momentum)

    def generate(
        self,
        teacher: nn.Module,
        images: torch.Tensor,
        conf_thresh: ThreshType,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """
        Generate merged pseudo-labels from clean + attacked teacher.

        Returns:
            (merged_pseudo, raw_clean_predictions)
        """
        was_training = teacher.training

        # --- Clean predictions ---
        teacher.eval()
        with torch.no_grad():
            clean_preds = teacher(images)
        clean_pseudo = filter_pseudo_labels(clean_preds, conf_thresh)

        # --- Attacked predictions ---
        if self.mode == "discriminator":
            attacked_images = self.attacker.attack(images, target_domain=self.target_domain)
        else:
            attacked_images = self.attacker.attack(images)
        with torch.no_grad():
            attacked_preds = teacher(attacked_images)
        attacked_pseudo = filter_pseudo_labels(attacked_preds, conf_thresh)

        teacher.train(was_training)

        # --- Merge: clean is primary, attacked adds non-overlapping boxes ---
        merged = _merge_two_pseudo(clean_pseudo, attacked_pseudo, self.merge_iou_threshold)

        return merged, clean_preds
