"""EMA and AEMA teacher updates."""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def ema_update(
    teacher: nn.Module,
    student: nn.Module,
    alpha: float = 0.999,
    global_step: Optional[int] = None,
) -> None:
    """
    Update teacher parameters in-place with EMA from student.

    Args:
        teacher      : teacher model (requires_grad=False)
        student      : student model (being trained)
        alpha        : EMA decay. Higher = slower teacher change.
        global_step  : current iteration. If provided, applies warmup:
                       effective_alpha = min(alpha, (1 + step) / (10 + step))
                       This prevents the teacher from diverging too fast at
                       the very start when the student is still random.
    """
    if global_step is not None:
        # Warmup: alpha ramps from ~0.09 → target_alpha over first ~10k steps
        warmup_alpha = (1.0 + global_step) / (10.0 + global_step)
        alpha = min(alpha, warmup_alpha)

    with torch.no_grad():
        # Update learnable parameters
        t_params = dict(teacher.named_parameters())
        s_params = dict(student.named_parameters())

        for name, t_p in t_params.items():
            if name in s_params:
                t_p.data.mul_(alpha).add_(s_params[name].data, alpha=1.0 - alpha)

        # Update buffers (e.g. BatchNorm running_mean / running_var)
        t_bufs = dict(teacher.named_buffers())
        s_bufs = dict(student.named_buffers())

        for name, t_b in t_bufs.items():
            if name in s_bufs and t_b.is_floating_point():
                t_b.data.mul_(alpha).add_(s_bufs[name].data, alpha=1.0 - alpha)


def _strip_scores(targets: List[Dict[str, torch.Tensor]]) -> List[Dict[str, torch.Tensor]]:
    """Detector training APIs consume boxes/labels; scores are inference metadata."""
    stripped = []
    for target in targets:
        stripped.append({
            "boxes": target["boxes"].detach(),
            "labels": target["labels"].detach(),
        })
    return stripped


def _classification_loss(loss_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Return the classification term used to rank AEMA-responsive parameters.

    Torchvision FCOS exposes "classification"; Faster R-CNN exposes
    "loss_classifier". For other detector wrappers, fall back to summing all
    losses so AEMA remains usable.
    """
    for key in ("classification", "loss_classifier"):
        if key in loss_dict:
            return loss_dict[key]
    return sum(v for v in loss_dict.values())


def _set_batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        module.eval()


def _count_pseudo_boxes(pseudo_targets: List[Dict[str, torch.Tensor]]) -> int:
    """Total number of pseudo boxes across all targets in the batch."""
    return sum(t["boxes"].numel() // 4 for t in pseudo_targets)


def _select_nonempty_pseudo(
    images: torch.Tensor,
    pseudo_targets: List[Dict[str, torch.Tensor]],
) -> Tuple[torch.Tensor, List[Dict[str, torch.Tensor]], List[int]]:
    """
    Keep only images whose pseudo target has >= 1 box.

    Returns:
        filtered_images, filtered_targets, kept_indices
    """
    kept = [i for i, t in enumerate(pseudo_targets) if t["boxes"].numel() > 0]
    if not kept:
        return images[:0], [], kept
    idx = torch.tensor(kept, dtype=torch.long, device=images.device)
    return images.index_select(0, idx), [pseudo_targets[i] for i in kept], kept


def _box_iou_single(box1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU between one box [4] and a set of boxes [N, 4].  Returns [N].
    Pure-torch fallback (no torchvision dependency).
    """
    x1 = torch.max(box1[0], boxes2[:, 0])
    y1 = torch.max(box1[1], boxes2[:, 1])
    x2 = torch.min(box1[2], boxes2[:, 2])
    y2 = torch.min(box1[3], boxes2[:, 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    union = area1 + area2 - inter
    return inter / union.clamp(min=1e-6)


def merge_teacher_student_pseudo_labels(
    teacher_pseudo: List[Dict[str, torch.Tensor]],
    student_preds: List[Dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    student_conf_thresh: float = 0.7,
) -> List[Dict[str, torch.Tensor]]:
    """
    Merge teacher pseudo-labels with student detections (DDT-style).

    Teacher pseudo-labels are the primary source.  Student detections that
    do NOT overlap any teacher box of the same class (IoU < ``iou_threshold``)
    are appended.  This adds useful detections the teacher missed while
    avoiding duplicates.

    Must be called under ``torch.no_grad()``.

    Args:
        teacher_pseudo        : filtered teacher pseudo-labels (boxes/labels/scores)
        student_preds         : raw student inference output (boxes/labels/scores)
        iou_threshold         : IoU at or above which a student box is considered
                                a duplicate of a teacher box
        student_conf_thresh   : min confidence for a student detection to be
                                considered for merging

    Returns:
        merged list of dicts (same length as inputs), each with
        boxes/labels/scores on the same device as the teacher pseudo.
    """
    merged: List[Dict[str, torch.Tensor]] = []
    for t_pseudo, s_pred in zip(teacher_pseudo, student_preds):
        t_boxes = t_pseudo["boxes"]
        t_labels = t_pseudo["labels"]
        t_scores = t_pseudo.get("scores", torch.zeros(t_boxes.shape[0], device=t_boxes.device))

        # Filter student predictions by confidence
        s_scores = s_pred.get("scores", torch.zeros(0, device=t_boxes.device))
        s_boxes = s_pred.get("boxes", torch.zeros(0, 4, device=t_boxes.device))
        s_labels = s_pred.get("labels", torch.zeros(0, dtype=torch.long, device=t_boxes.device))

        if s_scores.numel() > 0:
            keep = s_scores >= student_conf_thresh
            s_boxes = s_boxes[keep]
            s_labels = s_labels[keep]
            s_scores = s_scores[keep]

        # No teacher boxes → use student detections only
        if t_boxes.numel() == 0:
            merged.append({
                "boxes": s_boxes,
                "labels": s_labels,
                "scores": s_scores,
            })
            continue

        # No student boxes → use teacher pseudo only
        if s_boxes.numel() == 0:
            merged.append({
                "boxes": t_boxes,
                "labels": t_labels,
                "scores": t_scores,
            })
            continue

        # Add student detections that don't overlap any teacher box of the same class
        new_boxes: List[torch.Tensor] = [t_boxes]
        new_labels: List[torch.Tensor] = [t_labels]
        new_scores: List[torch.Tensor] = [t_scores]

        for i in range(s_boxes.shape[0]):
            s_label = int(s_labels[i].item())
            same_class = t_labels == s_labels[i]
            if same_class.any():
                ious = _box_iou_single(s_boxes[i], t_boxes[same_class])
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


class AEMAUpdater:
    """
    DDT-style adaptive EMA updater.

    Every step accumulates abs(teacher gradient) from a target-domain
    classification loss. Every ``update_interval`` steps, parameters in the
    global top ``top_ratio`` of accumulated gradient magnitudes use fast_alpha;
    the rest use slow_alpha.
    """

    def __init__(
        self,
        fast_alpha: float = 0.997,
        slow_alpha: float = 0.9996,
        top_ratio: float = 0.10,
        update_interval: int = 2,
    ) -> None:
        self.fast_alpha = fast_alpha
        self.slow_alpha = slow_alpha
        self.top_ratio = top_ratio
        self.update_interval = update_interval
        self.steps_since_update = 0
        self.grad_accum: Dict[str, torch.Tensor] = {}

    def reset(self) -> None:
        self.steps_since_update = 0
        self.grad_accum = {}

    def state_dict(self) -> Dict:
        return {
            "fast_alpha": self.fast_alpha,
            "slow_alpha": self.slow_alpha,
            "top_ratio": self.top_ratio,
            "update_interval": self.update_interval,
            "steps_since_update": self.steps_since_update,
            "grad_accum": {k: v.detach().cpu() for k, v in self.grad_accum.items()},
        }

    def load_state_dict(self, state: Dict, device: torch.device) -> None:
        self.steps_since_update = int(state.get("steps_since_update", 0))
        self.grad_accum = {
            k: v.to(device=device)
            for k, v in state.get("grad_accum", {}).items()
            if isinstance(v, torch.Tensor)
        }

    def accumulate_and_maybe_update(
        self,
        teacher: nn.Module,
        student: nn.Module,
        images: torch.Tensor,
        pseudo_targets: List[Dict[str, torch.Tensor]],
        student_predictions: Optional[List[Dict[str, torch.Tensor]]] = None,
        merge_enabled: bool = False,
        merge_iou_threshold: float = 0.5,
        merge_student_conf_thresh: float = 0.7,
    ) -> Dict[str, float]:
        """
        Accumulate teacher gradients, then update teacher from student when due.

        If ``student_predictions`` is provided and ``merge_enabled`` is True,
        the teacher pseudo-labels are merged with student detections (DDT-style)
        before computing the teacher loss.  This makes the AEMA importance
        signal depend on both teacher and student predictions rather than
        only the teacher's self-consistent pseudo-labels.

        If all pseudo targets are empty (zero boxes across the batch), the
        method returns cleanly without running a teacher forward, backward,
        or accumulating gradient importance.
        """
        log: Dict[str, float] = {
            "aema_updated": 0.0,
            "aema_fast_fraction": 0.0,
            "aema_fast_count": 0.0,
            "aema_total_count": 0.0,
            "aema_importance_loss": 0.0,
        }

        # --- Guard: skip entirely if no pseudo boxes ---
        num_pseudo_boxes = _count_pseudo_boxes(pseudo_targets)
        if images.numel() == 0 or num_pseudo_boxes == 0:
            self.steps_since_update += 1
            if self.steps_since_update >= self.update_interval:
                log.update(self._apply_update(teacher, student))
            return log

        # --- Filter to non-empty pseudo images ---
        filt_images, filt_pseudo, _ = _select_nonempty_pseudo(images, pseudo_targets)
        if filt_images.shape[0] == 0:
            self.steps_since_update += 1
            if self.steps_since_update >= self.update_interval:
                log.update(self._apply_update(teacher, student))
            return log

        # --- Optionally merge teacher + student pseudo labels ---
        if merge_enabled and student_predictions is not None:
            # Align student predictions with the filtered (non-empty) images
            _, filt_student_preds, _ = _select_nonempty_pseudo(images, student_predictions)
            with torch.no_grad():
                merged_pseudo = merge_teacher_student_pseudo_labels(
                    filt_pseudo,
                    filt_student_preds,
                    iou_threshold=merge_iou_threshold,
                    student_conf_thresh=merge_student_conf_thresh,
                )
            filt_pseudo = merged_pseudo

        raw_teacher = _unwrap(teacher)
        raw_student = _unwrap(student)
        was_training = teacher.training
        requires_grad = [p.requires_grad for p in raw_teacher.parameters()]
        for p in raw_teacher.parameters():
            p.requires_grad_(True)
        teacher.train()
        raw_teacher.apply(_set_batchnorm_eval)
        raw_teacher.zero_grad(set_to_none=True)

        clean_targets = _strip_scores(filt_pseudo)
        loss_dict = teacher(filt_images, clean_targets)
        importance_loss = _classification_loss(loss_dict)
        importance_loss.backward()

        with torch.no_grad():
            for name, param in raw_teacher.named_parameters():
                if param.grad is None:
                    continue
                grad = param.grad.detach().abs()
                if name not in self.grad_accum:
                    self.grad_accum[name] = torch.zeros_like(grad)
                self.grad_accum[name].add_(grad)

        log["aema_importance_loss"] = float(importance_loss.detach().item())
        raw_teacher.zero_grad(set_to_none=True)
        for param, old_requires_grad in zip(raw_teacher.parameters(), requires_grad):
            param.requires_grad_(old_requires_grad)
        teacher.train(was_training)

        self.steps_since_update += 1
        if self.steps_since_update >= self.update_interval:
            log.update(self._apply_update(teacher, student))
        return log

    def _apply_update(self, teacher: nn.Module, student: nn.Module) -> Dict[str, float]:
        teacher = _unwrap(teacher)
        student = _unwrap(student)
        t_params = dict(teacher.named_parameters())
        s_params = dict(student.named_parameters())
        t_bufs = dict(teacher.named_buffers())
        s_bufs = dict(student.named_buffers())

        # --- Build a flat gradient tensor for exact global top-k ---
        eligible_names: List[str] = []
        flat_chunks: List[torch.Tensor] = []
        offsets: List[int] = []
        offset = 0
        for name, grad in self.grad_accum.items():
            if name in t_params and grad.numel() > 0:
                flat = grad.reshape(-1)
                flat_chunks.append(flat)
                eligible_names.append(name)
                offsets.append(offset)
                offset += flat.numel()

        total_eligible = offset
        if flat_chunks:
            all_grads = torch.cat(flat_chunks)
            k = max(1, int(total_eligible * self.top_ratio))
            # Exact top-k: select exactly k indices (ties broken by index)
            _, topk_idx = torch.topk(all_grads, k=k, largest=True)
            flat_fast_mask = torch.zeros(total_eligible, dtype=torch.bool, device=all_grads.device)
            flat_fast_mask[topk_idx] = True
        else:
            flat_fast_mask = None

        fast_count = 0
        total_count = 0
        with torch.no_grad():
            for name, t_param in t_params.items():
                if name not in s_params:
                    continue
                s_param = s_params[name]
                total_count += t_param.numel()
                grad = self.grad_accum.get(name)

                if grad is None or flat_fast_mask is None:
                    # No gradient accumulated for this param → slow update
                    t_param.data.mul_(self.slow_alpha).add_(
                        s_param.data, alpha=1.0 - self.slow_alpha
                    )
                    continue

                # Reshape the global flat mask slice back to param shape
                idx = eligible_names.index(name)
                start = offsets[idx]
                end = start + grad.numel()
                mask = flat_fast_mask[start:end].reshape_as(grad)

                fast_count += int(mask.sum().item())
                slow_updated = t_param.data * self.slow_alpha + s_param.data * (1.0 - self.slow_alpha)
                fast_updated = t_param.data * self.fast_alpha + s_param.data * (1.0 - self.fast_alpha)
                t_param.data.copy_(torch.where(mask, fast_updated, slow_updated))

            for name, t_buf in t_bufs.items():
                if name not in s_bufs:
                    continue
                s_buf = s_bufs[name]
                if t_buf.is_floating_point():
                    t_buf.data.mul_(self.slow_alpha).add_(s_buf.data, alpha=1.0 - self.slow_alpha)
                else:
                    t_buf.data.copy_(s_buf.data)

        self.reset()
        return {
            "aema_updated": 1.0,
            "aema_fast_fraction": (fast_count / total_count) if total_count else 0.0,
            "aema_fast_count": float(fast_count),
            "aema_total_count": float(total_count),
        }


def _unwrap(model: nn.Module) -> nn.Module:
    """Unwrap nn.DataParallel if present."""
    return model.module if isinstance(model, nn.DataParallel) else model


def copy_student_to_teacher(teacher: nn.Module, student: nn.Module) -> None:
    """Hard copy student weights into teacher (alpha=0). Used for initialization."""
    teacher = _unwrap(teacher)
    student = _unwrap(student)
    with torch.no_grad():
        for t_p, s_p in zip(teacher.parameters(), student.parameters()):
            t_p.data.copy_(s_p.data)
        for t_b, s_b in zip(teacher.buffers(), student.buffers()):
            if t_b.is_floating_point():
                t_b.data.copy_(s_b.data)
            else:
                t_b.data.copy_(s_b.data)
