"""
Loss functions for each curriculum step.

4-phase mapping:
  Phase 1  → compute_rgb_loss       (RGB only, GT supervised)
  Phase 2  → compute_rgb_mid_loss   (mixed batch [RGB | MID], GT on whole batch)
  Phase 3  → compute_mid_ir_loss    (mixed batch [MID | IR], domain-specialized)
  Phase 4  → compute_ir_loss        (IR only, ir_teacher pseudo)

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


def select_nonempty_pseudo_targets(
    images: torch.Tensor,
    targets: List[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], List[int]]:
    """
    Keep only images whose pseudo target has at least 1 box.

    Images with zero-box pseudo targets are removed from the pseudo-supervised
    detection loss.  This prevents empty pseudo targets from becoming
    background-only supervised targets.

    Returns:
        filtered_images   : [B', C, H, W]  (B' <= B, may be 0)
        filtered_targets  : list of B' dicts
        kept_indices      : list of original-batch indices that were kept
    """
    kept = [i for i, t in enumerate(targets) if t["boxes"].numel() > 0]
    if not kept:
        return images[:0], [], kept
    idx = torch.tensor(kept, dtype=torch.long, device=images.device)
    return images.index_select(0, idx), [targets[i] for i in kept], kept


def _sum_loss_dict(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Sum all scalar tensors in a detector loss dict."""
    return sum(v for v in loss_dict.values())


# ---------------------------------------------------------------------------
# Pseudo-label health stats (for monitoring)
# ---------------------------------------------------------------------------

def compute_pseudo_health_stats(
    raw_predictions: List[Dict[str, torch.Tensor]],
    filtered_pseudo: List[Dict[str, torch.Tensor]],
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """
    Compute pseudo-label health metrics from detached CPU scalars.

    All values are Python floats (no GPU tensors) to avoid memory overhead.

    Args:
        raw_predictions : teacher predictions BEFORE confidence filtering
        filtered_pseudo  : teacher pseudo-labels AFTER confidence filtering
        class_names      : optional list of class names for per-class counts
                           (index 0=person, 1=car, 2=bicycle for FLIR)
    """
    n_images = len(filtered_pseudo)
    if n_images == 0:
        return {
            "pseudo_boxes_before_filter_per_image": 0.0,
            "pseudo_boxes_after_filter_per_image": 0.0,
            "pseudo_empty_image_ratio": 0.0,
            "pseudo_mean_confidence": 0.0,
            "pseudo_median_confidence": 0.0,
        }

    before_count = sum(p.get("scores", torch.zeros(0)).numel() for p in raw_predictions)
    after_count = sum(t["boxes"].numel() // 4 for t in filtered_pseudo)
    empty_count = sum(1 for t in filtered_pseudo if t["boxes"].numel() == 0)

    all_scores: List[float] = []
    for t in filtered_pseudo:
        scores = t.get("scores")
        if scores is not None and scores.numel() > 0:
            all_scores.extend(scores.detach().cpu().tolist())

    per_class: Dict[int, int] = {}
    for t in filtered_pseudo:
        labels = t.get("labels")
        if labels is not None and labels.numel() > 0:
            for lbl in labels.detach().cpu().tolist():
                per_class[int(lbl)] = per_class.get(int(lbl), 0) + 1

    stats: Dict[str, float] = {
        "pseudo_boxes_before_filter_per_image": before_count / n_images,
        "pseudo_boxes_after_filter_per_image": after_count / n_images,
        "pseudo_empty_image_ratio": empty_count / n_images,
        "pseudo_mean_confidence": sum(all_scores) / len(all_scores) if all_scores else 0.0,
        "pseudo_median_confidence": sorted(all_scores)[len(all_scores) // 2] if all_scores else 0.0,
    }

    # Per-class counts (person=0, car=1, bicycle=2 for FLIR)
    default_names = ["person", "car", "bicycle"]
    names = class_names if class_names else default_names
    for idx, name in enumerate(names):
        stats[f"pseudo_boxes_{name}"] = float(per_class.get(idx, 0))

    return stats


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
    skip_empty_pseudo: bool = True,
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
        log["gt_detection_loss"] = gt_loss.item()

    # --- Optional pseudo-label loss from rgb_teacher (off by default in warmup) ---
    if (rgb_teacher is not None or rgb_pseudo_targets is not None) and config.p1_pseudo_weight > 0.0:
        if rgb_pseudo_targets is None:
            with torch.no_grad():
                pseudo_preds = rgb_teacher(t_images)
            rgb_pseudo_targets = filter_pseudo_labels(pseudo_preds, conf_thresh)

        if skip_empty_pseudo:
            p_imgs, p_tgts, _ = select_nonempty_pseudo_targets(images, rgb_pseudo_targets)
            if p_imgs.shape[0] > 0:
                pseudo_loss = _sum_loss_dict(student(p_imgs, p_tgts)) * config.p1_pseudo_weight
                components.append(pseudo_loss)
                log["p1_pseudo_loss"] = pseudo_loss.item()
                log["pseudo_detection_loss"] = pseudo_loss.item()
            else:
                log["p1_pseudo_loss"] = 0.0
                log["pseudo_detection_loss"] = 0.0
        else:
            pseudo_loss = _sum_loss_dict(student(images, rgb_pseudo_targets)) * config.p1_pseudo_weight
            components.append(pseudo_loss)
            log["p1_pseudo_loss"] = pseudo_loss.item()
            log["pseudo_detection_loss"] = pseudo_loss.item()

    total_loss = sum(components) if components else images.sum() * 0.0
    log["p1_total_loss"] = total_loss.item()
    log["student_detection_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# Phase 2 — mixed [RGB | MID] loss
# ---------------------------------------------------------------------------

def compute_rgb_mid_loss(
    student: nn.Module,
    mixed_images: torch.Tensor,                         # student sees this (strong aug)
    gt_targets: List[Dict[str, torch.Tensor]],          # GT for ALL B images (both halves have GT)
    n_rgb: int,                                         # split index (RGB | MID)
    rgb_teacher: Optional[nn.Module] = None,
    ir_teacher: Optional[nn.Module] = None,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,      # teacher sees this (weak aug)
    rgb_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ir_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    skip_empty_pseudo: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 2 mixed-batch loss.

      L = p2_gt           · L_gt(whole batch)
        + p2_rgb_teacher  · L_pseudo(rgb_teacher on whole batch)
        + p2_ir_teacher   · L_pseudo(ir_teacher  on whole batch)

    Both halves of the batch have GT (MID inherits boxes from the RGB it was
    derived from), so the GT loss applies to all B images.

    When ``skip_empty_pseudo`` is True, images whose pseudo target has zero
    boxes are excluded from the pseudo detection loss.  If ALL pseudo targets
    are empty, the corresponding pseudo loss is 0 (no student forward).
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
        log["gt_detection_loss"] = loss.item()

    # --- rgb_teacher pseudo on whole batch ---
    if config.p2_rgb_teacher_weight > 0.0:
        if rgb_pseudo_targets is None and rgb_teacher is not None:
            with torch.no_grad():
                preds = rgb_teacher(t_images)
            rgb_pseudo_targets = filter_pseudo_labels(preds, conf_thresh)

        if rgb_pseudo_targets is not None:
            if skip_empty_pseudo:
                p_imgs, p_tgts, _ = select_nonempty_pseudo_targets(mixed_images, rgb_pseudo_targets)
                if p_imgs.shape[0] > 0:
                    loss = _sum_loss_dict(student(p_imgs, p_tgts)) * config.p2_rgb_teacher_weight
                    components.append(loss)
                    log["p2_rgb_teacher_loss"] = loss.item()
                else:
                    log["p2_rgb_teacher_loss"] = 0.0
            else:
                loss = _sum_loss_dict(student(mixed_images, rgb_pseudo_targets)) * config.p2_rgb_teacher_weight
                components.append(loss)
                log["p2_rgb_teacher_loss"] = loss.item()

    # --- ir_teacher pseudo on whole batch ---
    if config.p2_ir_teacher_weight > 0.0:
        if ir_pseudo_targets is None and ir_teacher is not None:
            with torch.no_grad():
                preds = ir_teacher(t_images)
            ir_pseudo_targets = filter_pseudo_labels(preds, conf_thresh)

        if ir_pseudo_targets is not None:
            if skip_empty_pseudo:
                p_imgs, p_tgts, _ = select_nonempty_pseudo_targets(mixed_images, ir_pseudo_targets)
                if p_imgs.shape[0] > 0:
                    loss = _sum_loss_dict(student(p_imgs, p_tgts)) * config.p2_ir_teacher_weight
                    components.append(loss)
                    log["p2_ir_teacher_loss"] = loss.item()
                else:
                    log["p2_ir_teacher_loss"] = 0.0
            else:
                loss = _sum_loss_dict(student(mixed_images, ir_pseudo_targets)) * config.p2_ir_teacher_weight
                components.append(loss)
                log["p2_ir_teacher_loss"] = loss.item()

    total_loss = sum(components) if components else mixed_images.sum() * 0.0
    log["p2_total_loss"] = total_loss.item()
    log["student_detection_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# Phase 3 — mixed [MID | IR] loss (domain-specialized)
# ---------------------------------------------------------------------------

def compute_mid_ir_loss(
    student: nn.Module,
    mixed_images: torch.Tensor,                          # student sees this (strong aug)
    mid_targets: List[Dict[str, torch.Tensor]],          # GT only for MID slice (length = n_mid)
    n_mid: int,                                          # split index (MID | IR)
    rgb_teacher: Optional[nn.Module] = None,
    ir_teacher: Optional[nn.Module] = None,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,       # teacher sees this (weak aug)
    rgb_pseudo_targets_mid: Optional[List[Dict[str, torch.Tensor]]] = None,
    ir_pseudo_targets_ir: Optional[List[Dict[str, torch.Tensor]]] = None,
    skip_empty_pseudo: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 3 mixed-batch loss — domain-specialized teachers.

      L = p3_gt           · L_gt(MID slice ONLY)              # IR has no GT
        + p3_rgb_teacher  · L_pseudo(rgb_teacher on MID only) # RGB teacher → MID
        + p3_ir_teacher   · L_pseudo(ir_teacher  on IR  only) # IR teacher  → IR

    The GT loss runs as a SEPARATE student forward on the MID slice — this
    avoids feeding empty targets for the IR slice into the detector.

    ``rgb_pseudo_targets_mid`` should be pre-generated from the MID slice
    (first ``n_mid`` images).  ``ir_pseudo_targets_ir`` should be pre-generated
    from the IR slice (remaining images).  If not provided, they will be
    generated from the corresponding slice of ``teacher_images``.
    """
    if config is None:
        config = LossConfig()

    B = mixed_images.shape[0]
    n_ir = B - n_mid
    t_images = teacher_images if teacher_images is not None else mixed_images

    components: List[torch.Tensor] = []
    log: Dict[str, float] = {"n_mid": float(n_mid), "n_ir": float(n_ir)}

    # --- GT loss on MID slice only ---
    if config.p3_gt_weight > 0.0 and n_mid > 0 and mid_targets:
        mid_slice = mixed_images[:n_mid]
        loss = _sum_loss_dict(student(mid_slice, mid_targets)) * config.p3_gt_weight
        components.append(loss)
        log["p3_gt_loss"] = loss.item()
        log["gt_detection_loss"] = loss.item()

    # --- rgb_teacher pseudo on MID slice only ---
    if config.p3_rgb_teacher_weight > 0.0 and n_mid > 0:
        mid_slice = mixed_images[:n_mid]
        if rgb_pseudo_targets_mid is None and rgb_teacher is not None:
            with torch.no_grad():
                preds = rgb_teacher(t_images[:n_mid])
            rgb_pseudo_targets_mid = filter_pseudo_labels(preds, conf_thresh)

        if rgb_pseudo_targets_mid is not None:
            if skip_empty_pseudo:
                p_imgs, p_tgts, _ = select_nonempty_pseudo_targets(mid_slice, rgb_pseudo_targets_mid)
                if p_imgs.shape[0] > 0:
                    loss = _sum_loss_dict(student(p_imgs, p_tgts)) * config.p3_rgb_teacher_weight
                    components.append(loss)
                    log["p3_rgb_teacher_loss"] = loss.item()
                else:
                    log["p3_rgb_teacher_loss"] = 0.0
            else:
                loss = _sum_loss_dict(student(mid_slice, rgb_pseudo_targets_mid)) * config.p3_rgb_teacher_weight
                components.append(loss)
                log["p3_rgb_teacher_loss"] = loss.item()

    # --- ir_teacher pseudo on IR slice only ---
    if config.p3_ir_teacher_weight > 0.0 and n_ir > 0:
        ir_slice = mixed_images[n_mid:]
        if ir_pseudo_targets_ir is None and ir_teacher is not None:
            with torch.no_grad():
                preds = ir_teacher(t_images[n_mid:])
            ir_pseudo_targets_ir = filter_pseudo_labels(preds, conf_thresh)

        if ir_pseudo_targets_ir is not None:
            if skip_empty_pseudo:
                p_imgs, p_tgts, _ = select_nonempty_pseudo_targets(ir_slice, ir_pseudo_targets_ir)
                if p_imgs.shape[0] > 0:
                    loss = _sum_loss_dict(student(p_imgs, p_tgts)) * config.p3_ir_teacher_weight
                    components.append(loss)
                    log["p3_ir_teacher_loss"] = loss.item()
                else:
                    log["p3_ir_teacher_loss"] = 0.0
            else:
                loss = _sum_loss_dict(student(ir_slice, ir_pseudo_targets_ir)) * config.p3_ir_teacher_weight
                components.append(loss)
                log["p3_ir_teacher_loss"] = loss.item()

    total_loss = sum(components) if components else mixed_images.sum() * 0.0
    log["p3_total_loss"] = total_loss.item()
    log["student_detection_loss"] = total_loss.item()
    return total_loss, log


# ---------------------------------------------------------------------------
# Phase 4 — IR focus loss
# ---------------------------------------------------------------------------

def compute_ir_loss(
    student: nn.Module,
    ir_images: torch.Tensor,
    ir_teacher: Optional[nn.Module] = None,
    config: Optional[LossConfig] = None,
    conf_thresh: ThreshType = 0.7,
    teacher_images: Optional[torch.Tensor] = None,
    ir_pseudo_targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    skip_empty_pseudo: bool = True,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Phase 4 IR unsupervised loss — ir_teacher pseudo-labels only.

    teacher_images: if provided, teacher infers on this (weak aug);
                    student trains on ir_images (strong aug).
                    If None, both use ir_images.

    When ``skip_empty_pseudo`` is True and ALL pseudo targets are empty,
    the loss is 0 and the student forward is not invoked.
    """
    if config is None:
        config = LossConfig()

    t_images = teacher_images if teacher_images is not None else ir_images
    log: Dict[str, float] = {}

    if ir_pseudo_targets is None and ir_teacher is not None:
        with torch.no_grad():
            ir_preds = ir_teacher(t_images)
        ir_pseudo_targets = filter_pseudo_labels(ir_preds, conf_thresh)

    if ir_pseudo_targets is None:
        total_loss = ir_images.sum() * 0.0
        log["p4_ir_teacher_loss"] = 0.0
        log["pseudo_detection_loss"] = 0.0
        log["p4_total_loss"] = 0.0
        log["student_detection_loss"] = 0.0
        return total_loss, log

    if skip_empty_pseudo:
        p_imgs, p_tgts, _ = select_nonempty_pseudo_targets(ir_images, ir_pseudo_targets)
        if p_imgs.shape[0] > 0:
            total_loss = _sum_loss_dict(student(p_imgs, p_tgts)) * config.p4_ir_teacher_weight
        else:
            total_loss = ir_images.sum() * 0.0
    else:
        total_loss = _sum_loss_dict(student(ir_images, ir_pseudo_targets)) * config.p4_ir_teacher_weight

    log["p4_ir_teacher_loss"] = total_loss.item()
    log["pseudo_detection_loss"] = total_loss.item()
    log["p4_total_loss"] = total_loss.item()
    log["student_detection_loss"] = total_loss.item()
    return total_loss, log
