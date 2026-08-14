"""
CurriculumDomainAdaptationTrainer (4-phase, mixed-batch).

Architecture:
  - 1 student   (trained by gradient descent, has optimizer)
  - 2 teachers:
      rgb_teacher  (EMA of student, updated in Phase 2)
      ir_teacher   (EMA of student, updated in Phase 3 and Phase 4)
  Teachers are ALWAYS in eval mode and require no grad.

Training flow per iteration:
  scheduler.get_next_step(global_step) → "rgb" | "rgb_mid" | "mid_ir" | "ir"
  dispatch to train_rgb_step / train_rgb_mid_step / train_mid_ir_step / train_ir_step

Phase 2 and Phase 3 build MIXED batches via in-batch split:
  Phase 2 :  [RGB | MID]     n_rgb = round(B * phase2_rgb_ratio)
  Phase 3 :  [MID | IR]      n_mid = round(B * phase3_mid_ratio)
"""

import logging
import os
import random
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from batch_types import IRBatch, MidIrBatch, RGBBatch, RgbMidBatch
from config import TrainingConfig
from discriminator import (
    DomainDiscriminator,
    GradientReversal,
    compute_adv_loss,
    grl_lambda_schedule,
)
from ema import AEMAUpdater, ema_update
from losses import (
    compute_ir_loss,
    compute_mid_ir_loss,
    compute_pseudo_health_stats,
    compute_rgb_loss,
    compute_rgb_mid_loss,
    filter_pseudo_labels,
    select_nonempty_pseudo_targets,
)
from saga import SemanticAwareGrayAugmentation
from scheduler import CurriculumScheduler, DomainStep, Phase

if TYPE_CHECKING:
    from adaptive_threshold import AdaptiveThresholdScheduler
    from evaluator import PhaseEvaluator

try:
    import torchvision.transforms as T
    import torchvision.transforms.functional as TF
    _HAS_TF = True
except ImportError:
    _HAS_TF = False

logger = logging.getLogger(__name__)


class CurriculumDomainAdaptationTrainer:
    """
    4-phase Curriculum Domain Adaptation Trainer with mixed batches.

    Args:
        student      : model being trained (must follow detector train/eval API)
        rgb_teacher  : EMA teacher for RGB domain
        ir_teacher   : EMA teacher for IR  domain
        optimizer    : optimizer attached to student parameters ONLY
        config       : full TrainingConfig
        rgb_loader   : DataLoader yielding (images, targets) — labeled RGB
        ir_loader    : DataLoader yielding images (or (images,)) — unlabeled IR
    """

    def __init__(
        self,
        student: nn.Module,
        rgb_teacher: nn.Module,
        ir_teacher: nn.Module,
        optimizer: Optimizer,
        config: TrainingConfig,
        rgb_loader: DataLoader,
        ir_loader: DataLoader,
        threshold_scheduler: Optional["AdaptiveThresholdScheduler"] = None,
        phase_evaluator: Optional["PhaseEvaluator"] = None,
        phase1_best_path: Optional[str] = None,
        phase2_best_path: Optional[str] = None,
        disc_rgb: Optional[nn.Module] = None,
        disc_ir:  Optional[nn.Module] = None,
        disc_optimizer: Optional[Optimizer] = None,
    ) -> None:
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)

        # Adversarial components (None = disabled)
        self.disc_rgb       = disc_rgb
        self.disc_ir        = disc_ir
        self.disc_optimizer = disc_optimizer
        self._grl           = GradientReversal(lambda_=1.0) if (disc_rgb or disc_ir) else None

        # Optional extensions
        self.threshold_scheduler = threshold_scheduler
        self.phase_evaluator     = phase_evaluator
        self.phase_best_paths: Dict[str, str] = {}
        if phase1_best_path:
            self.phase_best_paths["PHASE1_RGB_WARMUP"] = phase1_best_path
        if phase2_best_path:
            self.phase_best_paths["PHASE2_RGB_MID"] = phase2_best_path

        # Pseudo-label starvation tracking (diagnostic only)
        self._starvation_streak: int = 0
        self._starvation_warned: bool = False

        self._setup_models()

        self.saga      = SemanticAwareGrayAugmentation(apply_prob=config.saga.apply_prob)
        self.scheduler = CurriculumScheduler(config.curriculum)
        self.rgb_aema = AEMAUpdater(
            fast_alpha=config.ema.aema_fast_alpha,
            slow_alpha=config.ema.aema_slow_alpha,
            top_ratio=config.ema.aema_top_ratio,
            update_interval=config.ema.aema_update_interval,
        )
        self.ir_aema = AEMAUpdater(
            fast_alpha=config.ema.aema_fast_alpha,
            slow_alpha=config.ema.aema_slow_alpha,
            top_ratio=config.ema.aema_top_ratio,
            update_interval=config.ema.aema_update_interval,
        )

        # Infinite data iterators — never exhaust
        self._rgb_iter: Iterator = self._infinite(rgb_loader)
        self._ir_iter: Iterator  = self._infinite(ir_loader)

        self.global_step: int = 0
        self._last_phase: Optional[Phase] = None
        self._phase_step_count: int = 0   # steps taken in current phase (for early logging)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        """Move models to device; freeze & eval teachers; move discriminators."""
        for model in (self.student, self.rgb_teacher, self.ir_teacher):
            model.to(self.device)

        for teacher in (self.rgb_teacher, self.ir_teacher):
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()

        for disc in (self.disc_rgb, self.disc_ir):
            if disc is not None:
                disc.to(self.device)

    # ------------------------------------------------------------------
    # Phase transition handler
    # ------------------------------------------------------------------

    def _on_phase_transition(self, from_phase: Optional[Phase], to_phase: Phase) -> None:
        """
        Called exactly once when the curriculum phase changes.

        Phase 1 → Phase 2 :
          Hard-copy best Phase-1 student → BOTH teachers.
        Phase 2 → Phase 3 :
          Initialize IR teacher from best Phase-2 student (or current student
          if no best checkpoint exists).  Reset IR AEMA statistics.
          RGB teacher is NOT reset (continues from Phase 2).
        Other transitions : no-op.
        """
        if from_phase is None:
            return

        if from_phase == Phase.PHASE1_RGB_WARMUP and to_phase == Phase.PHASE2_RGB_MID:
            self._init_teachers_from_checkpoint(
                "PHASE1_RGB_WARMUP", teachers=["rgb", "ir"],
                fallback_msg="PHASE1→PHASE2: no best checkpoint, using current student",
            )
            self.rgb_aema.reset()
            self.ir_aema.reset()

        if from_phase == Phase.PHASE2_RGB_MID and to_phase == Phase.PHASE3_MID_IR:
            self._init_teachers_from_checkpoint(
                "PHASE2_RGB_MID", teachers=["ir"],
                fallback_msg="PHASE2→PHASE3: no best Phase-2 checkpoint, using current student",
            )
            self.ir_aema.reset()
            logger.info("[Phase Transition] Initialized IR teacher from Phase-2 student")
            logger.info("[Phase Transition] Reset IR AEMA statistics")

    def _init_teachers_from_checkpoint(
        self,
        phase_key: str,
        teachers: list,
        fallback_msg: str,
    ) -> None:
        """Load student weights from the best checkpoint of a phase into teachers."""
        from ema import copy_student_to_teacher
        best_path = self.phase_best_paths.get(phase_key)
        if best_path and os.path.exists(best_path):
            logger.info(f"[Phase transition] loading {best_path} → {teachers} teacher(s)")
            ckpt = torch.load(best_path, map_location=self.device, weights_only=False)
            state = ckpt["student"]
            if "rgb" in teachers:
                self.rgb_teacher.load_state_dict(state)
            if "ir" in teachers:
                self.ir_teacher.load_state_dict(state)
        else:
            logger.info(f"[Phase transition] {fallback_msg}")
            if "rgb" in teachers:
                copy_student_to_teacher(self.rgb_teacher, self.student)
            if "ir" in teachers:
                copy_student_to_teacher(self.ir_teacher,  self.student)

    @torch.no_grad()
    def _teacher_pseudo_labels(
        self,
        teacher: nn.Module,
        images: torch.Tensor,
        conf_thresh,
    ) -> List[Dict[str, torch.Tensor]]:
        was_training = teacher.training
        teacher.eval()
        preds = teacher(images)
        teacher.train(was_training)
        return filter_pseudo_labels(preds, conf_thresh)

    @torch.no_grad()
    def _teacher_pseudo_labels_with_raw(
        self,
        teacher: nn.Module,
        images: torch.Tensor,
        conf_thresh,
    ) -> Tuple[List[Dict[str, torch.Tensor]], List[Dict[str, torch.Tensor]]]:
        """Return (filtered_pseudo, raw_predictions) for health monitoring."""
        was_training = teacher.training
        teacher.eval()
        preds = teacher(images)
        teacher.train(was_training)
        pseudo = filter_pseudo_labels(preds, conf_thresh)
        return pseudo, preds

    def _log_pseudo_health(
        self,
        log: Dict[str, float],
        prefix: str,
        raw_preds: List[Dict[str, torch.Tensor]],
        filtered_pseudo: List[Dict[str, torch.Tensor]],
    ) -> None:
        """Compute and inject pseudo-label health metrics into the log dict."""
        stats = compute_pseudo_health_stats(raw_preds, filtered_pseudo)
        for key, val in stats.items():
            log[f"{prefix}_{key}"] = val

        # Check for pseudo-label starvation
        empty_ratio = stats.get("pseudo_empty_image_ratio", 0.0)
        if empty_ratio > 0.8:
            self._starvation_streak += 1
            if self._starvation_streak >= 10 and not self._starvation_warned:
                logger.warning(
                    f"WARNING: pseudo-label starvation detected: "
                    f"{empty_ratio*100:.0f}% of target images contain no accepted pseudo labels."
                )
                self._starvation_warned = True
        else:
            self._starvation_streak = 0
            self._starvation_warned = False

    def _teacher_update(
        self,
        teacher_name: str,
        teacher: nn.Module,
        updater: AEMAUpdater,
        images: Optional[torch.Tensor],
        pseudo_targets: Optional[List[Dict[str, torch.Tensor]]],
        log: Dict,
    ) -> None:
        # AEMA disabled → fall back to classic EMA
        effective_mode = "ema" if self.config.ablation.disable_aema else self.config.ema.mode
        prefix = f"{teacher_name}_{effective_mode}"

        if effective_mode == "ema":
            ema_update(
                teacher=teacher,
                student=self.student,
                alpha=self.config.ema.alpha,
                global_step=self.global_step if self.config.ema.use_warmup else None,
            )
            log[f"{prefix}_updated"] = 1.0
            return

        if images is None or pseudo_targets is None:
            log[f"{prefix}_updated"] = 0.0
            return

        # Generate student predictions for DDT-style teacher-student merge
        student_preds = None
        merge_enabled = not self.config.ablation.disable_teacher_student_merge
        if merge_enabled:
            num_pseudo_boxes = sum(
                t["boxes"].numel() // 4 for t in pseudo_targets
            )
            if num_pseudo_boxes > 0:
                with torch.no_grad():
                    was_training = self.student.training
                    self.student.eval()
                    student_preds = self.student(images)
                    self.student.train(was_training)

        aema_log = updater.accumulate_and_maybe_update(
            teacher=teacher,
            student=self.student,
            images=images,
            pseudo_targets=pseudo_targets,
            student_predictions=student_preds,
            merge_enabled=merge_enabled,
            merge_iou_threshold=self.config.ema.merge_iou_threshold,
            merge_student_conf_thresh=self.config.ema.merge_student_conf_thresh,
        )
        for key, value in aema_log.items():
            log[f"{teacher_name}_{key}"] = value

    # ------------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------------

    def _get_threshold(self, phase: Phase, teacher: str = "both"):
        """
        Return current confidence threshold for pseudo-label filtering.

        Phase 4 ir_teacher gets `steps_into_phase` for linear ramp-up
        (measured from the start of Phase 4 = phase3_end).
        """
        if self.threshold_scheduler is None:
            return self.config.pseudo_label_conf_thresh

        steps_into_phase = (
            max(0, self.global_step - self.config.curriculum.phase3_end)
            if phase == Phase.PHASE4_IR_FOCUS
            else 0
        )

        if teacher == "rgb":
            return self.threshold_scheduler.rgb_teacher(phase, steps_into_phase)
        if teacher == "ir":
            return self.threshold_scheduler.ir_teacher(phase, steps_into_phase)
        return self.threshold_scheduler.both(phase, steps_into_phase)

    def _get_grl_lambda(self, phase: Phase) -> float:
        """Return GRL lambda for the current step (fixed or DANN schedule)."""
        adv = self.config.adv
        if not adv.use_schedule:
            return adv.grl_lambda
        cur = self.config.curriculum
        if phase == Phase.PHASE2_RGB_MID:
            return grl_lambda_schedule(
                self.global_step, cur.phase1_end, cur.phase2_end, adv.grl_lambda
            )
        if phase == Phase.PHASE3_MID_IR:
            return grl_lambda_schedule(
                self.global_step, cur.phase2_end, cur.phase3_end, adv.grl_lambda
            )
        return adv.grl_lambda

    # ------------------------------------------------------------------
    # Infinite data utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _infinite(loader: DataLoader) -> Iterator:
        """Wrap a DataLoader in an infinite iterator."""
        while True:
            yield from loader

    def _next_rgb(self) -> RGBBatch:
        images, targets = next(self._rgb_iter)
        images = images.to(self.device)
        targets = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]
        return RGBBatch(images=images, targets=targets)

    def _next_ir(self) -> IRBatch:
        raw = next(self._ir_iter)
        images = raw[0] if isinstance(raw, (list, tuple)) else raw
        return IRBatch(images=images.to(self.device))

    # ------------------------------------------------------------------
    # Mixed-batch builders
    # ------------------------------------------------------------------

    @staticmethod
    def _split_count(batch_size: int, ratio: float) -> int:
        """
        Pick how many images go to the "earlier" half so both halves get
        at least 1 image when batch_size >= 2.
        """
        n = int(round(batch_size * ratio))
        if batch_size >= 2:
            n = max(1, min(batch_size - 1, n))
        else:
            n = max(0, min(batch_size, n))
        return n

    def _next_rgb_mid(self) -> RgbMidBatch:
        """
        Phase 2 mixed batch [RGB | MID].

        Pull a single RGB batch of size B; first n_rgb keep RGB pixels,
        last n_mid = B - n_rgb get SAGA → MID. Targets are kept for all B
        (SAGA does not change box coords).
        """
        rgb = self._next_rgb()
        B = rgb.images.shape[0]
        n_rgb = self._split_count(B, self.config.curriculum.phase2_rgb_ratio)
        n_mid = B - n_rgb

        rgb_part = rgb.images[:n_rgb]                           # [n_rgb, 3, H, W]
        if n_mid > 0:
            mid_source  = rgb.images[n_rgb:]                    # [n_mid, 3, H, W]
            mid_boxes   = [t["boxes"] for t in rgb.targets[n_rgb:]]
            mid_part    = self.saga.apply_to_batch(mid_source, mid_boxes)
            mixed       = torch.cat([rgb_part, mid_part], dim=0)
        else:
            mixed = rgb_part

        return RgbMidBatch(
            images=mixed,
            targets=rgb.targets,
            n_rgb=n_rgb,
            source_images=rgb.images,
        )

    def _next_mid_ir(self) -> MidIrBatch:
        """
        Phase 3 mixed batch [MID | IR] — parts kept separate pre-aug.

        - Pull RGB batch, take first n_mid images, apply SAGA → MID part (GT kept).
        - Pull IR  batch, take first n_ir  images                   (no GT).

        n_mid + n_ir = B = min(len(rgb_batch), len(ir_batch)). Extra images
        from the larger loader are dropped this iteration (they reappear on
        the next pull due to the infinite iterator).
        """
        rgb = self._next_rgb()
        ir  = self._next_ir()

        B = min(rgb.images.shape[0], ir.images.shape[0])
        n_mid = self._split_count(B, self.config.curriculum.phase3_mid_ratio)
        n_ir  = B - n_mid

        if n_mid > 0:
            mid_source  = rgb.images[:n_mid]
            mid_targets = rgb.targets[:n_mid]
            mid_boxes   = [t["boxes"] for t in mid_targets]
            mid_images  = self.saga.apply_to_batch(mid_source, mid_boxes)
        else:
            mid_images  = rgb.images[:0]
            mid_targets = []

        ir_images = ir.images[:n_ir] if n_ir > 0 else ir.images[:0]

        return MidIrBatch(
            mid_images=mid_images,
            ir_images=ir_images,
            mid_targets=mid_targets,
            n_mid=n_mid,
            n_ir=n_ir,
        )

    # ------------------------------------------------------------------
    # Geometric aug (weak — shared by teacher and student)
    # ------------------------------------------------------------------

    def _geometric_aug(
        self,
        images: torch.Tensor,
        targets: Optional[List[Dict]],
        aug,
    ):
        """
        Geometric aug shared between teacher and student:
          1. Random horizontal flip
          2. Random scale in [multiscale_min, multiscale_max] × original size
          3. Per-image random crop (if scaled > target) or zero-pad (if scaled < target)
        """
        B, C, H, W = images.shape

        if random.random() < aug.hflip_prob:
            images = torch.flip(images, dims=[-1])
            if targets is not None:
                targets = [self._flip_boxes(t, W) for t in targets]

        s = random.uniform(aug.multiscale_min, aug.multiscale_max)
        new_h = max(1, int(H * s))
        new_w = max(1, int(W * s))
        images = torch.nn.functional.interpolate(
            images, size=(new_h, new_w), mode="bilinear", align_corners=False
        )
        if targets is not None:
            targets = [self._scale_boxes(t, s) for t in targets]

        out_imgs, out_tgts = [], []
        for i in range(B):
            img, tgt = self._crop_or_pad(
                images[i],
                targets[i] if targets is not None else None,
                aug.multiscale_target_h,
                aug.multiscale_target_w,
            )
            out_imgs.append(img)
            out_tgts.append(tgt)

        return torch.stack(out_imgs), (out_tgts if targets is not None else None)

    @staticmethod
    def _flip_boxes(target: Dict, W: int) -> Dict:
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return target
        flipped = boxes.clone()
        flipped[:, 0] = W - boxes[:, 2]
        flipped[:, 2] = W - boxes[:, 0]
        return {**target, "boxes": flipped}

    @staticmethod
    def _scale_boxes(target: Dict, s: float) -> Dict:
        boxes = target.get("boxes")
        if boxes is None or boxes.numel() == 0:
            return target
        return {**target, "boxes": boxes * s}

    @staticmethod
    def _crop_or_pad(
        img: torch.Tensor,
        target: Optional[Dict],
        target_h: int,
        target_w: int,
    ):
        C, H, W = img.shape
        top  = random.randint(0, max(0, H - target_h))
        left = random.randint(0, max(0, W - target_w))

        img = img[:, top:top + min(H, target_h), left:left + min(W, target_w)]

        pad_b = target_h - img.shape[1]
        pad_r = target_w - img.shape[2]
        if pad_b > 0 or pad_r > 0:
            img = torch.nn.functional.pad(img, (0, pad_r, 0, pad_b), value=0.0)

        if target is not None:
            boxes = target.get("boxes")
            if boxes is not None and boxes.numel() > 0:
                offset = torch.tensor(
                    [left, top, left, top], dtype=boxes.dtype, device=boxes.device
                )
                boxes = boxes - offset
                boxes[:, 0::2].clamp_(0, target_w)
                boxes[:, 1::2].clamp_(0, target_h)
                keep = ((boxes[:, 2] - boxes[:, 0]) > 1) & ((boxes[:, 3] - boxes[:, 1]) > 1)
                target = {**target, "boxes": boxes[keep], "labels": target["labels"][keep]}

        return img, target

    # ------------------------------------------------------------------
    # Photometric aug (strong — student only)
    # ------------------------------------------------------------------

    def _rgb_photometric_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Strong photometric aug for RGB / MID images (3-channel, color-aware)."""
        if not _HAS_TF or images.numel() == 0:
            return images
        aug = self.config.rgb_aug

        if random.random() < aug.blur_prob:
            sigma = random.uniform(0.1, aug.blur_sigma_max)
            images = TF.gaussian_blur(images, kernel_size=[3, 3], sigma=sigma)

        if random.random() < aug.color_jitter_prob:
            jitter = T.ColorJitter(
                brightness=aug.cj_brightness,
                contrast=aug.cj_contrast,
                saturation=aug.cj_saturation,
                hue=aug.cj_hue,
            )
            images = torch.stack([jitter(img) for img in images])

        if random.random() < aug.random_erasing_prob:
            eraser = T.RandomErasing(
                p=1.0,
                scale=(aug.random_erasing_scale_min, aug.random_erasing_scale_max),
                ratio=(aug.random_erasing_ratio_min, aug.random_erasing_ratio_max),
                value=0,
            )
            images = torch.stack([eraser(img) for img in images])

        return images

    def _ir_photometric_aug(self, images: torch.Tensor) -> torch.Tensor:
        """Strong photometric aug for IR (thermal) images."""
        if images.numel() == 0:
            return images
        aug = self.config.ir_aug

        if random.random() < aug.intensity_shift_prob:
            shift = random.uniform(-aug.intensity_shift_mag, aug.intensity_shift_mag)
            images = torch.clamp(images + shift, 0.0, 1.0)

        if random.random() < aug.contrast_jitter_prob:
            factor = 1.0 + random.uniform(-aug.contrast_jitter_mag, aug.contrast_jitter_mag)
            mean = images.mean(dim=[-1, -2], keepdim=True)
            images = torch.clamp((images - mean) * factor + mean, 0.0, 1.0)

        if random.random() < aug.gamma_prob:
            gamma = random.uniform(aug.gamma_min, aug.gamma_max)
            images = torch.clamp(images, 1e-6, 1.0).pow(gamma)

        if random.random() < aug.gaussian_noise_prob:
            noise = torch.randn_like(images) * aug.gaussian_noise_std
            images = torch.clamp(images + noise, 0.0, 1.0)

        return images

    # ------------------------------------------------------------------
    # Gradient clip + optimizer step
    # ------------------------------------------------------------------

    def _clip_and_step(self) -> float:
        if self.config.grad_clip > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.student.parameters(), self.config.grad_clip
            ).item()
        else:
            grad_norm = 0.0
        self.optimizer.step()
        return grad_norm

    # ------------------------------------------------------------------
    # Domain steps
    # ------------------------------------------------------------------

    def train_rgb_step(self, phase: Phase = Phase.PHASE1_RGB_WARMUP) -> Dict:
        """
        Phase 1 RGB step — supervised on labeled source data.

        Loss:   p1_gt_weight · L_gt  +  optional p1_pseudo_weight · L_pseudo
        EMA:    none (warmup)
        """
        self.student.train()
        batch = self._next_rgb()
        self.optimizer.zero_grad()

        # Phase 1 = pure supervised warmup: geometric aug only, no teacher
        student_images, student_targets = self._geometric_aug(
            batch.images, batch.targets, self.config.rgb_aug
        )

        loss, log = compute_rgb_loss(
            student=self.student,
            images=student_images,
            gt_targets=student_targets,
            rgb_teacher=None,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher="rgb"),
            teacher_images=None,
        )

        loss.backward()
        log["grad_norm"] = self._clip_and_step()
        log["domain"] = "rgb"
        log["adversarial_loss"] = 0.0
        return log

    def train_rgb_mid_step(self, phase: Phase = Phase.PHASE2_RGB_MID) -> Dict:
        """
        Phase 2 mixed step — in-batch split [RGB | MID].

        Teacher (weak aug):  same geometric transform as student, NO photometric.
        Student (strong aug): teacher's geometric + RGB photometric.

        Loss:   p2_gt · L_gt(whole)
              + p2_rgb_teacher · L_pseudo(rgb_teacher, whole)
              + p2_ir_teacher  · L_pseudo(ir_teacher,  whole)
        EMA:    rgb_teacher only (Phase 2 specialist)
        """
        self.student.train()
        mixed = self._next_rgb_mid()
        self.optimizer.zero_grad()
        if self.disc_optimizer is not None:
            self.disc_optimizer.zero_grad()

        weak_images, aug_targets = self._geometric_aug(
            mixed.images, mixed.targets, self.config.rgb_aug
        )
        strong_images = self._rgb_photometric_aug(weak_images.clone())
        thresh_rgb = self._get_threshold(phase, teacher="rgb")
        thresh_ir = self._get_threshold(phase, teacher="ir")
        rgb_pseudo = None
        ir_pseudo = None
        rgb_raw_preds = None
        ir_raw_preds = None
        if self.config.loss.p2_rgb_teacher_weight > 0.0 or (
            not self.config.ablation.disable_aema and self.config.teacher_update.p2_update_rgb_teacher
        ):
            rgb_pseudo, rgb_raw_preds = self._teacher_pseudo_labels_with_raw(
                self.rgb_teacher, weak_images, thresh_rgb)
        if self.config.loss.p2_ir_teacher_weight > 0.0 or (
            not self.config.ablation.disable_aema and self.config.teacher_update.p2_update_ir_teacher
        ):
            ir_pseudo, ir_raw_preds = self._teacher_pseudo_labels_with_raw(
                self.ir_teacher, weak_images, thresh_ir)

        det_loss, log = compute_rgb_mid_loss(
            student=self.student,
            mixed_images=strong_images,
            gt_targets=aug_targets,
            n_rgb=mixed.n_rgb,
            rgb_teacher=self.rgb_teacher,
            ir_teacher=self.ir_teacher,
            config=self.config.loss,
            conf_thresh=thresh_rgb,
            teacher_images=weak_images,
            rgb_pseudo_targets=rgb_pseudo,
            ir_pseudo_targets=ir_pseudo,
            skip_empty_pseudo=self.config.ablation.skip_empty_pseudo,
        )

        # Pseudo-label health monitoring
        if rgb_pseudo is not None and rgb_raw_preds is not None:
            self._log_pseudo_health(log, "rgb", rgb_raw_preds, rgb_pseudo)
        if ir_pseudo is not None and ir_raw_preds is not None:
            self._log_pseudo_health(log, "ir", ir_raw_preds, ir_pseudo)

        # Adversarial alignment: disc_rgb distinguishes RGB (0) vs MID (1)
        total_loss = det_loss
        if (self.disc_rgb is not None and self.config.adv.p2_adv_weight > 0.0
                and not self.config.ablation.disable_adv):
            features  = self.student.get_backbone_features(strong_images)
            self._grl.set_lambda(self._get_grl_lambda(phase))
            adv_loss, adv_log = compute_adv_loss(
                features, self.disc_rgb, self._grl, n_a=mixed.n_rgb,
                label_smoothing=self.config.adv.label_smoothing,
            )
            total_loss = det_loss + self.config.adv.p2_adv_weight * adv_loss
            log["p2_adv_loss"]  = adv_log["adv_loss"]
            log["adversarial_loss"] = adv_log["adv_loss"]
            log["p2_disc_acc"]  = adv_log["disc_acc"]
            log["p2_grl_lambda"] = self._grl.lambda_
        else:
            log["adversarial_loss"] = 0.0

        total_loss.backward()
        log["grad_norm"] = self._clip_and_step()
        if self.disc_optimizer is not None:
            self.disc_optimizer.step()

        if self.config.teacher_update.p2_update_rgb_teacher:
            self._teacher_update(
                teacher_name="rgb",
                teacher=self.rgb_teacher,
                updater=self.rgb_aema,
                images=weak_images[mixed.n_rgb:],
                pseudo_targets=rgb_pseudo[mixed.n_rgb:] if rgb_pseudo is not None else None,
                log=log,
            )
        if self.config.teacher_update.p2_update_ir_teacher:
            self._teacher_update(
                teacher_name="ir",
                teacher=self.ir_teacher,
                updater=self.ir_aema,
                images=weak_images[mixed.n_rgb:],
                pseudo_targets=ir_pseudo[mixed.n_rgb:] if ir_pseudo is not None else None,
                log=log,
            )

        log["domain"] = "rgb_mid"
        return log

    def train_mid_ir_step(self, phase: Phase = Phase.PHASE3_MID_IR) -> Dict:
        """
        Phase 3 mixed step — in-batch split [MID | IR].

        Per-slice augmentation (raw H/W may differ — concat happens AFTER aug):
          MID slice : rgb_aug geometric  + RGB-style photometric
          IR  slice : ir_aug  geometric  + IR-style photometric

        rgb_aug.multiscale_target_h/w MUST equal ir_aug.multiscale_target_h/w
        so the two augmented sub-batches share a shape and can be concatenated.

        Loss:   p3_gt · L_gt(MID slice only)
              + p3_rgb_teacher · L_pseudo(rgb_teacher, whole)
              + p3_ir_teacher  · L_pseudo(ir_teacher,  whole)
        EMA:    ir_teacher only (Phase 3 specialist)
        """
        self.student.train()
        mixed = self._next_mid_ir()
        self.optimizer.zero_grad()
        if self.disc_optimizer is not None:
            self.disc_optimizer.zero_grad()

        # Per-slice geometric aug (raw → target shape)
        if mixed.n_mid > 0:
            weak_mid, aug_mid_targets = self._geometric_aug(
                mixed.mid_images, mixed.mid_targets, self.config.rgb_aug
            )
        else:
            weak_mid, aug_mid_targets = None, []

        if mixed.n_ir > 0:
            weak_ir, _ = self._geometric_aug(
                mixed.ir_images, None, self.config.ir_aug
            )
        else:
            weak_ir = None

        # Per-slice photometric aug
        strong_mid = self._rgb_photometric_aug(weak_mid.clone()) if weak_mid is not None else None
        strong_ir  = self._ir_photometric_aug(weak_ir.clone())   if weak_ir  is not None else None

        # Concat into the actual mixed batch (shapes match after geometric aug)
        if weak_mid is not None and weak_ir is not None:
            weak_images   = torch.cat([weak_mid,   weak_ir],   dim=0)
            strong_images = torch.cat([strong_mid, strong_ir], dim=0)
            n_mid_aug = weak_mid.shape[0]
        elif weak_mid is not None:
            weak_images, strong_images = weak_mid, strong_mid
            n_mid_aug = weak_mid.shape[0]
        else:
            weak_images, strong_images = weak_ir, strong_ir
            n_mid_aug = 0
        thresh_rgb = self._get_threshold(phase, teacher="rgb")
        thresh_ir = self._get_threshold(phase, teacher="ir")
        rgb_pseudo_mid = None
        ir_pseudo_ir = None
        rgb_raw_mid = None
        ir_raw_ir = None
        # RGB teacher → MID slice only
        if (self.config.loss.p3_rgb_teacher_weight > 0.0 or (
            not self.config.ablation.disable_aema and self.config.teacher_update.p3_update_rgb_teacher
        )) and n_mid_aug > 0:
            rgb_pseudo_mid, rgb_raw_mid = self._teacher_pseudo_labels_with_raw(
                self.rgb_teacher, weak_mid, thresh_rgb)
        # IR teacher → IR slice only
        if (self.config.loss.p3_ir_teacher_weight > 0.0 or (
            not self.config.ablation.disable_aema and self.config.teacher_update.p3_update_ir_teacher
        )) and (weak_ir is not None):
            ir_pseudo_ir, ir_raw_ir = self._teacher_pseudo_labels_with_raw(
                self.ir_teacher, weak_ir, thresh_ir)

        det_loss, log = compute_mid_ir_loss(
            student=self.student,
            mixed_images=strong_images,
            mid_targets=aug_mid_targets,
            n_mid=n_mid_aug,
            rgb_teacher=self.rgb_teacher,
            ir_teacher=self.ir_teacher,
            config=self.config.loss,
            conf_thresh=thresh_rgb,
            teacher_images=weak_images,
            rgb_pseudo_targets_mid=rgb_pseudo_mid,
            ir_pseudo_targets_ir=ir_pseudo_ir,
            skip_empty_pseudo=self.config.ablation.skip_empty_pseudo,
        )

        # Pseudo-label health monitoring (domain-specialized)
        if rgb_pseudo_mid is not None and rgb_raw_mid is not None:
            self._log_pseudo_health(log, "rgb_mid", rgb_raw_mid, rgb_pseudo_mid)
        if ir_pseudo_ir is not None and ir_raw_ir is not None:
            self._log_pseudo_health(log, "ir_ir", ir_raw_ir, ir_pseudo_ir)

        # Adversarial alignment: disc_ir distinguishes MID (0) vs IR (1)
        total_loss = det_loss
        if (self.disc_ir is not None and self.config.adv.p3_adv_weight > 0.0
                and not self.config.ablation.disable_adv):
            features  = self.student.get_backbone_features(strong_images)
            self._grl.set_lambda(self._get_grl_lambda(phase))
            adv_loss, adv_log = compute_adv_loss(
                features, self.disc_ir, self._grl, n_a=n_mid_aug,
                label_smoothing=self.config.adv.label_smoothing,
            )
            total_loss = det_loss + self.config.adv.p3_adv_weight * adv_loss
            log["p3_adv_loss"]   = adv_log["adv_loss"]
            log["adversarial_loss"] = adv_log["adv_loss"]
            log["p3_disc_acc"]   = adv_log["disc_acc"]
            log["p3_grl_lambda"] = self._grl.lambda_
        else:
            log["adversarial_loss"] = 0.0

        total_loss.backward()
        log["grad_norm"] = self._clip_and_step()
        if self.disc_optimizer is not None:
            self.disc_optimizer.step()

        # Phase 3: only ir_teacher is updated (domain specialist for IR)
        if self.config.teacher_update.p3_update_ir_teacher:
            self._teacher_update(
                teacher_name="ir",
                teacher=self.ir_teacher,
                updater=self.ir_aema,
                images=weak_ir if weak_ir is not None else None,
                pseudo_targets=ir_pseudo_ir,
                log=log,
            )

        log["domain"] = "mid_ir"
        return log

    def train_ir_step(self, phase: Phase = Phase.PHASE4_IR_FOCUS) -> Dict:
        """
        Phase 4 IR step — unsupervised on unlabeled IR data, ir_teacher only.

        Loss:   p4_ir_teacher_weight · L_pseudo(ir_teacher, IR)
        EMA:    ir_teacher
        """
        self.student.train()
        batch = self._next_ir()
        self.optimizer.zero_grad()

        weak_images, _ = self._geometric_aug(batch.images, None, self.config.ir_aug)
        strong_images = self._ir_photometric_aug(weak_images.clone())
        thresh = self._get_threshold(phase, teacher="ir")
        ir_pseudo = None
        ir_raw_preds = None
        if self.config.loss.p4_ir_teacher_weight > 0.0 or (
            not self.config.ablation.disable_aema and self.config.teacher_update.p4_update_ir_teacher
        ):
            ir_pseudo, ir_raw_preds = self._teacher_pseudo_labels_with_raw(
                self.ir_teacher, weak_images, thresh)

        loss, log = compute_ir_loss(
            student=self.student,
            ir_images=strong_images,
            ir_teacher=self.ir_teacher,
            config=self.config.loss,
            conf_thresh=thresh,
            teacher_images=weak_images,
            ir_pseudo_targets=ir_pseudo,
            skip_empty_pseudo=self.config.ablation.skip_empty_pseudo,
        )

        # Pseudo-label health monitoring
        if ir_pseudo is not None and ir_raw_preds is not None:
            self._log_pseudo_health(log, "ir", ir_raw_preds, ir_pseudo)
        log["adversarial_loss"] = 0.0

        loss.backward()
        log["grad_norm"] = self._clip_and_step()

        if self.config.teacher_update.p4_update_ir_teacher:
            self._teacher_update(
                teacher_name="ir",
                teacher=self.ir_teacher,
                updater=self.ir_aema,
                images=weak_images,
                pseudo_targets=ir_pseudo,
                log=log,
            )

        log["domain"] = "ir"
        return log

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def train_one_iteration(self) -> Dict:
        step:  DomainStep = self.scheduler.get_next_step(self.global_step)
        phase: Phase      = self.scheduler.get_phase(self.global_step)

        if phase != self._last_phase:
            self._on_phase_transition(from_phase=self._last_phase, to_phase=phase)
            self._phase_step_count = 0
        self._last_phase = phase

        if step == "rgb":
            log = self.train_rgb_step(phase=phase)
        elif step == "rgb_mid":
            log = self.train_rgb_mid_step(phase=phase)
        elif step == "mid_ir":
            log = self.train_mid_ir_step(phase=phase)
        elif step == "ir":
            log = self.train_ir_step(phase=phase)
        else:
            raise ValueError(f"Unknown domain step: {step!r}")  # pragma: no cover

        log["phase"]       = phase.name
        log["global_step"] = self.global_step

        if self.threshold_scheduler is not None:
            rt = self._get_threshold(phase, teacher="rgb")
            it = self._get_threshold(phase, teacher="ir")
            log["thresh_rgb"] = min(rt.values()) if isinstance(rt, dict) else rt
            log["thresh_ir"]  = min(it.values()) if isinstance(it, dict) else it

        if self.phase_evaluator is not None:
            self.phase_evaluator.step(
                model=self.student,
                global_step=self.global_step,
                current_phase=phase,
            )

        if self.global_step % self.config.log_interval == 0 or self._phase_step_count < 10:
            self._log(log)

        self._phase_step_count += 1
        self.global_step += 1
        return log

    def train_one_epoch(self, steps_per_epoch: int) -> List[Dict]:
        return [self.train_one_iteration() for _ in range(steps_per_epoch)]

    def train(self, total_iterations: int) -> None:
        logger.info(
            f"Starting Curriculum DA training  total_iters={total_iterations}"
            f"  scheduler={self.scheduler}"
        )
        for _ in range(total_iterations):
            self.train_one_iteration()
        logger.info("Training complete.")

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str, extra_state: Optional[Dict] = None) -> None:
        ckpt = {
            "global_step": self.global_step,
            "student":     self.student.state_dict(),
            "rgb_teacher": self.rgb_teacher.state_dict(),
            "ir_teacher":  self.ir_teacher.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
            "ema_mode":    self.config.ema.mode,
            "rgb_aema":    self.rgb_aema.state_dict(),
            "ir_aema":     self.ir_aema.state_dict(),
        }
        if self.disc_rgb is not None:
            ckpt["disc_rgb"] = self.disc_rgb.state_dict()
        if self.disc_ir is not None:
            ckpt["disc_ir"] = self.disc_ir.state_dict()
        if self.disc_optimizer is not None:
            ckpt["disc_optimizer"] = self.disc_optimizer.state_dict()
        if extra_state is not None:
            ckpt["extra_state"] = extra_state
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved → {path}  (step {self.global_step})")

    def load_checkpoint(self, path: str) -> Dict:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = ckpt["global_step"]
        self.student.load_state_dict(ckpt["student"])
        self.rgb_teacher.load_state_dict(ckpt["rgb_teacher"])
        self.ir_teacher.load_state_dict(ckpt["ir_teacher"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        if self.config.ema.mode == "aema":
            if ckpt.get("ema_mode", "ema") == "aema":
                if "rgb_aema" in ckpt:
                    self.rgb_aema.load_state_dict(ckpt["rgb_aema"], self.device)
                if "ir_aema" in ckpt:
                    self.ir_aema.load_state_dict(ckpt["ir_aema"], self.device)
            else:
                logger.warning(
                    "Checkpoint was saved with classic EMA; starting fresh AEMA accumulators."
                )
        elif ckpt.get("ema_mode", "ema") != self.config.ema.mode:
            logger.warning(
                f"Checkpoint ema_mode={ckpt.get('ema_mode')} differs from current "
                f"ema_mode={self.config.ema.mode}; ignoring updater state."
            )
        if self.disc_rgb is not None and "disc_rgb" in ckpt:
            self.disc_rgb.load_state_dict(ckpt["disc_rgb"])
        if self.disc_ir is not None and "disc_ir" in ckpt:
            self.disc_ir.load_state_dict(ckpt["disc_ir"])
        if self.disc_optimizer is not None and "disc_optimizer" in ckpt:
            self.disc_optimizer.load_state_dict(ckpt["disc_optimizer"])
        logger.info(f"Checkpoint loaded ← {path}  (step {self.global_step})")
        return ckpt

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, val_loader: DataLoader) -> Dict:
        self.student.eval()
        all_predictions = []
        for batch in val_loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(self.device)
            preds = self.student(images)
            all_predictions.extend(preds)
        self.student.train()
        return {"num_samples": len(all_predictions)}

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------

    def _log(self, log: Dict) -> None:
        step   = log.get("global_step", self.global_step)
        phase  = log.get("phase",  "?")
        domain = log.get("domain", "?")
        losses = "  ".join(
            f"{k}={v:.4f}"
            for k, v in sorted(log.items())
            if isinstance(v, float)
        )
        logger.info(f"[{step:07d}]  phase={phase:<22s}  domain={domain:<8s}  {losses}")
