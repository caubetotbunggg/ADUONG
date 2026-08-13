"""
Loss functions for each curriculum step.

4-phase mapping:
  Phase 1  → compute_rgb_loss    (RGB only, GT supervised)
  Phase 2  → compute_rgb_mid_loss (mixed batch [RGB | MID], GT on whole batch)
  Phase 3  → compute_mid_ir_loss  (mixed batch [MID | IR], GT on MID slice only)
  Phase 4  → compute_ir_loss     (IR only, ir_teacher pseudo)

Expected detector API (Faster-RCNN / FCOS / DINO style):
  Training:  model(images, targets) → Dict[str, Tensor]  (named loss components)
  Inference: model(images)          → List[Dict]          (boxes, labels, scores)

All teacher forward passes run under torch.no_grad() — callers must not wrap
this module in no_grad() since the student path needs gradients.
"""

from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from config import LossConfig

# Global or per-class confidence threshold
ThreshType = Union[float, Dict[int, float]]

_DEFAULT_THRESH = 0.7   # fallback for classes absent from per-class dict


# ---------------------------------------------------------------------------
# Pseudo-label filtering
# ---------------------------------------------------------------------------

def filter_pseudo_labels(
    predictions: List[Dict[str, torch.Tensor]],
    conf_thresh: ThreshType = 0.7,
) -> List[Dict[str, torch.Tensor]]:
    """
    Filter teacher predictions by confidence score.

    Args:
        predictions : output of model(images) in inference mode
                      each dict has "boxes" [N,4], "labels" [N], "scores" [N]
        conf_thresh : float  — global threshold applied to all classes
                      Dict[int, float] — per-class threshold; classes absent
                      from the dict fall back to _DEFAULT_THRESH (0.7)

    Returns:
        filtered list of dicts (same length as predictions, empty dicts possible)
    """
    pseudo = []
    for pred in predictions:
        scores = pred.get("scores", torch.zeros(0))
        if scores.numel() == 0:
            pseudo.append({
                "boxes":  torch.zeros(0, 4, device=scores.device),
                "labels": torch.zeros(0, dtype=torch.long, device=scores.device),
                "scores": scores,
            })
            continue

        if isinstance(conf_thresh, dict):
            labels = pred["labels"]
            keep = torch.tensor(
                [scores[i].item() >= conf_thresh.get(int(labels[i].item()), _DEFAULT_THRESH)
                 for i in range(len(scores))],
                dtype=torch.bool,
                device=scores.device,
            )
        else:
            keep = scores >= conf_thresh

        pseudo.append({
            "boxes":  pred["boxes"][keep],
            "labels": pred["labels"][keep],
            "scores": pred["scores"][keep],
        })
    return pseudo


def _sum_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum all scalar tensors in a detector loss dict."""
    return sum(v for v in loss_dict.values())


# ---------------------------------------------------------------------------
# Phase 1 — RGB warmup loss
# ---------------------------------------------------------------------------

def compute_rgb_loss(
    student: nn.Module,
    images: torch.Tensor,
    gt_targets: List[Dict[str, torch.Tensor]],
    rgb_teacher: Optional[nn.Module] = None,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,
    rgb_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 1 RGB supervised loss.

      L = p1_gt_weight·L_gt  +  p1_pseudo_weight·L_pseudo(rgb_teacher)

    teacher_images: if provided, teacher infers on this (weak aug);
                    student trains on `images` (strong aug). If None, both use `images`.
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else images

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {}

    # --- Supervised GT loss (always active) ---
    if config.p1_gt_weight > 0.0:
        gt_loss = _sum_loss_dict(student(images, gt_targets)) * config.p1_gt_weight
        components.append(gt_loss)
        log["p1_gt_loss"] = gt_loss.item()

    # --- Optional pseudo-label loss from rgb_teacher (off by default in warmup) ---
    if (rgb_teacher is not None or rgb_pseudo_targets is not None) and config.p1_pseudo_weight > 0.0:
        if rgb_pseudo_targets is None:
            with torch.no_grad():
                pseudo_preds = rgb_teacher(t_images)
            rgb_pseudo_targets = filter_pseudo_labels(pseudo_preds, conf_thresh)
        pseudo_loss = _sum_loss_dict(student(images, rgb_pseudo_targets)) * config.p1_pseudo_weight
        components.append(pseudo_loss)
        log["p1_pseudo_loss"] = pseudo_loss.item()

    total_loss = sum(components) if components else images.sum() * 0.0
    log["p1_total_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# Phase 2 — mixed [RGB | MID] loss
# ---------------------------------------------------------------------------

def compute_rgb_mid_loss(
    student: nn.Module,
    mixed_images: torch.Tensor,                         # student sees this (strong aug)
    gt_targets: List[Dict[str, torch.Tensor]],          # GT for ALL B images (both halves have GT)
    n_rgb: int,                                         # split index (RGB | MID)
    rgb_teacher: nn.Module,
    ir_teacher: nn.Module,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,      # teacher sees this (weak aug)
    rgb_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ir_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 2 mixed-batch loss.

      L = p2_gt           · L_gt(whole batch)
        + p2_rgb_teacher  · L_pseudo(rgb_teacher on whole batch)
        + p2_ir_teacher   · L_pseudo(ir_teacher  on whole batch)

    Both halves of the batch have GT (MID inherits boxes from the RGB it was
    derived from), so the GT loss applies to all B images.
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else mixed_images

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {"n_rgb": float(n_rgb), "n_mid": float(mixed_images.shape[0] - n_rgb)}

    # --- GT loss on whole batch ---
    if config.p2_gt_weight > 0.0 and gt_targets is not None:
        loss = _sum_loss_dict(student(mixed_images, gt_targets)) * config.p2_gt_weight
        components.append(loss)
        log["p2_gt_loss"] = loss.item()

    # --- rgb_teacher pseudo on whole batch ---
    if config.p2_rgb_teacher_weight > 0.0:
        if rgb_pseudo_targets is None:
            with torch.no_grad():
                preds = rgb_teacher(t_images)
            rgb_pseudo_targets = filter_pseudo_labels(preds, conf_thresh)
        loss = _sum_loss_dict(student(mixed_images, rgb_pseudo_targets)) * config.p2_rgb_teacher_weight
        components.append(loss)
        log["p2_rgb_teacher_loss"] = loss.item()

    # --- ir_teacher pseudo on whole batch ---
    if config.p2_ir_teacher_weight > 0.0:
        if ir_pseudo_targets is None:
            with torch.no_grad():
                preds = ir_teacher(t_images)
            ir_pseudo_targets = filter_pseudo_labels(preds, conf_thresh)
        loss = _sum_loss_dict(student(mixed_images, ir_pseudo_targets)) * config.p2_ir_teacher_weight
        components.append(loss)
        log["p2_ir_teacher_loss"] = loss.item()

    total_loss = sum(components) if components else mixed_images.sum() * 0.0
    log["p2_total_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# Phase 3 — mixed [MID | IR] loss
# ---------------------------------------------------------------------------

def compute_mid_ir_loss(
    student: nn.Module,
    mixed_images: torch.Tensor,                          # student sees this (strong aug)
    mid_targets: List[Dict[str, torch.Tensor]],          # GT only for MID slice (length = n_mid)
    n_mid: int,                                          # split index (MID | IR)
    rgb_teacher: nn.Module,
    ir_teacher: nn.Module,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,       # teacher sees this (weak aug)
    rgb_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ir_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 3 mixed-batch loss.

      L = p3_gt           · L_gt(MID slice ONLY)         # IR has no GT
        + p3_rgb_teacher  · L_pseudo(rgb_teacher on whole batch)
        + p3_ir_teacher   · L_pseudo(ir_teacher  on whole batch)

    The GT loss runs as a SEPARATE student forward on the MID slice — this
    avoids feeding empty targets for the IR slice into the detector.
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else mixed_images
    B = mixed_images.shape[0]

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {"n_mid": float(n_mid), "n_ir": float(B - n_mid)}

    # --- GT loss on MID slice only ---
    if config.p3_gt_weight > 0.0 and n_mid > 0 and mid_targets:
        mid_slice = mixed_images[:n_mid]
        loss = _sum_loss_dict(student(mid_slice, mid_targets)) * config.p3_gt_weight
        components.append(loss)
        log["p3_gt_loss"] = loss.item()

    # --- rgb_teacher pseudo on whole batch ---
    if config.p3_rgb_teacher_weight > 0.0:
        if rgb_pseudo_targets is None:
            with torch.no_grad():
                preds = rgb_teacher(t_images)
            rgb_pseudo_targets = filter_pseudo_labels(preds, conf_thresh)
        loss = _sum_loss_dict(student(mixed_images, rgb_pseudo_targets)) * config.p3_rgb_teacher_weight
        components.append(loss)
        log["p3_rgb_teacher_loss"] = loss.item()

    # --- ir_teacher pseudo on whole batch ---
    if config.p3_ir_teacher_weight > 0.0:
        if ir_pseudo_targets is None:
            with torch.no_grad():
                preds = ir_teacher(t_images)
            ir_pseudo_targets = filter_pseudo_labels(preds, conf_thresh)
        loss = _sum_loss_dict(student(mixed_images, ir_pseudo_targets)) * config.p3_ir_teacher_weight
        components.append(loss)
        log["p3_ir_teacher_loss"] = loss.item()

    total_loss = sum(components) if components else mixed_images.sum() * 0.0
    log["p3_total_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# Phase 4 — IR focus loss
# ---------------------------------------------------------------------------

def compute_ir_loss(
    student: nn.Module,
    ir_images: torch.Tensor,
    ir_teacher: nn.Module,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,
    ir_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 4 IR unsupervised loss — ir_teacher pseudo-labels only.

    teacher_images: if provided, teacher infers on this (weak aug);
                    student trains on ir_images (strong aug).
                    If None, both use ir_images.
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else ir_images
    log: Dict[str, float] = {}

    if ir_pseudo_targets is None:
        with torch.no_grad():
            ir_preds = ir_teacher(t_images)
        ir_pseudo_targets = filter_pseudo_labels(ir_preds, conf_thresh)

    total_loss = _sum_loss_dict(student(ir_images, ir_pseudo_targets)) * config.p4_ir_teacher_weight

    log["p4_ir_teacher_loss"] = total_loss.item()
    log["p4_total_loss"]      = total_loss.item()
    return total_loss, log
