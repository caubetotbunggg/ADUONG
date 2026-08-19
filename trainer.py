"""
CurriculumDomainAdaptationTrainer (3-phase, mixed-batch).

Architecture:
  - 1 student     (trained by gradient descent từ Phase 2 trở đi)
  - 2 teachers:
      rgb_teacher  (train supervised trên RGB trong Phase 1, EMA update từ Phase 2)
      ir_teacher   (train supervised trên MID/GAN trong Phase 1, EMA update từ Phase 3)
  Phase 1: TEACHERS tự train độc lập — student KHÔNG train.
  Phase 1→2 transition: student được khởi tạo từ rgb_teacher weights.
  Phase 2+: teachers freeze grad, chỉ update qua EMA.

MID domain bridge:
  GAN translator (frozen CycleGAN generator, RGB → grayscale 3-channel).

Training flow per iteration:
  scheduler.get_next_step(global_step) → "rgb" | "rgb_mid" | "mid_ir"
  Phase 1 "rgb" step → train_teachers_step() (cả 2 teacher train song song)
  Phase 2+ → train_rgb_mid_step / train_mid_ir_step (student)

Phase 2 and Phase 3 build MIXED batches via in-batch split:
  Phase 2 :  [RGB | MID]     n_rgb = round(B * phase2_rgb_ratio)
  Phase 3 :  [MID | IR]      n_mid = round(B * phase3_mid_ratio)
"""

import logging
import random
from typing import TYPE_CHECKING, Dict, Iterator, List, Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

from batch_types import IRBatch, MidIrBatch, RGBBatch, RgbMidBatch
from config import TrainingConfig
from ema import ema_update, AEMAUpdater
from gan_translator import GANTranslator
from losses import (
    compute_mid_ir_loss,
    compute_rgb_mid_loss,
    filter_pseudo_labels,
)
from aat import AATPseudoLabelGenerator
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
        gan_translator: GANTranslator,
        teacher_optimizer: Optimizer,
        mid_loader: DataLoader,
        threshold_scheduler: Optional["AdaptiveThresholdScheduler"] = None,
        phase_evaluator: Optional["PhaseEvaluator"] = None,
    ) -> None:
        self.student = student
        self.rgb_teacher = rgb_teacher
        self.ir_teacher = ir_teacher
        self.optimizer = optimizer
        self.config = config
        self.device = torch.device(config.device)

        # GAN translator: RGB → MID (frozen, no grad)
        self.gan_translator = gan_translator

        # Teacher optimizer — dùng trong Phase 1, bị bỏ qua từ Phase 2
        self.teacher_optimizer = teacher_optimizer

        # Optional extensions
        self.threshold_scheduler = threshold_scheduler
        self.phase_evaluator     = phase_evaluator

        self._setup_models()

        # --- AEMA / AAT (ablation toggles; default off = classic EMA, no AAT) ---
        if config.ema.mode == "aema":
            self.aema_rgb = AEMAUpdater(
                fast_alpha=config.ema.aema_fast_alpha,
                slow_alpha=config.ema.aema_slow_alpha,
                top_ratio=config.ema.aema_top_ratio,
                update_interval=config.ema.aema_update_interval,
            )
            self.aema_ir = AEMAUpdater(
                fast_alpha=config.ema.aema_fast_alpha,
                slow_alpha=config.ema.aema_slow_alpha,
                top_ratio=config.ema.aema_top_ratio,
                update_interval=config.ema.aema_update_interval,
            )
        else:
            self.aema_rgb = None
            self.aema_ir = None

        if config.aat.enabled:
            self.aat_generator = AATPseudoLabelGenerator(
                epsilon=config.aat.epsilon,
                merge_iou_threshold=config.aat.merge_iou,
                mode=config.aat.mode,
            )
        else:
            self.aat_generator = None

        self.scheduler = CurriculumScheduler(config.curriculum)

        # Infinite data iterators — never exhaust
        self._rgb_iter: Iterator = self._infinite(rgb_loader)
        self._ir_iter: Iterator  = self._infinite(ir_loader)
        self._mid_iter: Iterator = self._infinite(mid_loader)

        self.global_step: int = 0
        self._last_phase: Optional[Phase] = None
        self._phase_step_count: int = 0   # steps taken in current phase (for early logging)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup_models(self) -> None:
        """Move models to device. Teachers bắt đầu ở train mode (Phase 1)."""
        for model in (self.student, self.rgb_teacher, self.ir_teacher):
            model.to(self.device)
        # Student freeze cho đến Phase 2
        self.student.eval()
        for p in self.student.parameters():
            p.requires_grad = False
        # Teachers train mode trong Phase 1
        for teacher in (self.rgb_teacher, self.ir_teacher):
            teacher.train()
            for p in teacher.parameters():
                p.requires_grad = True

    def _freeze_teachers(self) -> None:
        """Freeze teachers sau Phase 1 — gọi tại transition Phase 1→2."""
        for teacher in (self.rgb_teacher, self.ir_teacher):
            for p in teacher.parameters():
                p.requires_grad = False
            teacher.eval()

    def _unfreeze_student(self) -> None:
        """Unfreeze student và init từ rgb_teacher tại Phase 1→2."""
        from ema import copy_student_to_teacher
        copy_student_to_teacher(self.student, self.rgb_teacher)
        self.student.train()
        for p in self.student.parameters():
            p.requires_grad = True
        logger.info("[Phase transition] student initialized from rgb_teacher weights")

    # ------------------------------------------------------------------
    # Phase transition handler
    # ------------------------------------------------------------------

    def _teacher_pseudo(self, teacher, images, conf_thresh):
        """Pseudo-labels for a teacher (AAT-merged clean+attacked when enabled)."""
        if self.aat_generator is not None:
            merged, _ = self.aat_generator.generate(teacher, images, conf_thresh)
            return merged
        with torch.no_grad():
            preds = teacher(images)
        return filter_pseudo_labels(preds, conf_thresh)

    def _update_teacher(self, teacher, images, phase):
        """Update a teacher via classic EMA (default) or AEMA (when enabled)."""
        conf = self._get_threshold(phase, teacher="both")
        if self.config.ema.mode == "aema":
            pseudo = self._teacher_pseudo(teacher, images, conf)
            updater = self.aema_rgb if teacher is self.rgb_teacher else self.aema_ir
            return updater.accumulate_and_maybe_update(
                teacher, self.student, images=images, pseudo_targets=pseudo,
            )
        ema_update(
            teacher=teacher, student=self.student,
            alpha=self.config.ema.alpha,
            global_step=self.global_step if self.config.ema.use_warmup else None,
        )
        return {}

    def _on_phase_transition(self, from_phase: Optional[Phase], to_phase: Phase) -> None:
        """
        Phase 1 → Phase 2:
          - Freeze cả 2 teachers (chuyển sang EMA mode).
          - Init student từ rgb_teacher weights.
        """
        # --- Resume (--resume) enters with from_phase=None ---
        if from_phase is None:
            if to_phase == Phase.PHASE1_RGB_WARMUP:
                # Fresh start at step 0: nothing to do (teachers train, student frozen).
                return
            # Resume into Phase 2/3: ensure teachers are frozen+eval and student
            # is trainable (model mode / requires_grad are NOT saved in checkpoints).
            self._freeze_teachers()
            if to_phase == Phase.PHASE2_RGB_MID and self.global_step == self.config.curriculum.phase1_end:
                # Resumed exactly at the Phase1->2 boundary: the checkpoint student is
                # still the untrained Phase-1 student -> init it from rgb_teacher, exactly
                # like a fresh Phase1->2 transition.
                logger.info("[Resume at PHASE1->2] freezing teachers, init student from rgb_teacher")
                self._unfreeze_student()
            else:
                # Resumed mid-Phase2 / Phase3: student is already trained -> keep its
                # weights, only flip requires_grad + train mode.
                logger.info(f"[Resume into {to_phase.name}] freeze teachers, unfreeze student (keep weights)")
                for _p in self.student.parameters():
                    _p.requires_grad = True
                self.student.train()
            return
        if from_phase == Phase.PHASE1_RGB_WARMUP and to_phase == Phase.PHASE2_RGB_MID:
            logger.info("[Phase transition] PHASE1 → PHASE2: freezing teachers, init student")
            self._freeze_teachers()
            self._unfreeze_student()

    # ------------------------------------------------------------------
    # Adaptive threshold
    # ------------------------------------------------------------------

    def _get_threshold(self, phase: Phase, teacher: str = "both"):
        """
        Return current confidence threshold for pseudo-label filtering.
        """
        if self.threshold_scheduler is None:
            return self.config.pseudo_label_conf_thresh

        if teacher == "rgb":
            return self.threshold_scheduler.rgb_teacher(phase)
        if teacher == "ir":
            return self.threshold_scheduler.ir_teacher(phase)
        return self.threshold_scheduler.both(phase)

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

    def _next_mid(self) -> RGBBatch:
        """Lấy batch MID (GAN-translated) có GT — dùng trong Phase 1 để train ir_teacher."""
        images, targets = next(self._mid_iter)
        images = images.to(self.device)
        targets = [
            {k: v.to(self.device) if isinstance(v, torch.Tensor) else v
             for k, v in t.items()}
            for t in targets
        ]
        return RGBBatch(images=images, targets=targets)

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
            mid_part    = self.gan_translator.apply_to_batch(mid_source)
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
            mid_images  = self.gan_translator.apply_to_batch(mid_source)
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

    def train_teachers_step(self, phase: Phase = Phase.PHASE1_RGB_WARMUP) -> Dict:
        """
        Phase 1: train 2 teachers độc lập, student KHÔNG train.

        rgb_teacher ← supervised trên RGB batch (GT boxes)
        ir_teacher  ← supervised trên MID batch (GAN-translated, cùng GT boxes)

        Student bị frozen trong phase này.
        """
        self.rgb_teacher.train()
        self.ir_teacher.train()
        self.teacher_optimizer.zero_grad()

        # rgb_teacher train trên RGB
        rgb_batch = self._next_rgb()
        rgb_images, rgb_targets = self._geometric_aug(
            rgb_batch.images, rgb_batch.targets, self.config.rgb_aug
        )
        rgb_loss_dict = self.rgb_teacher(rgb_images, rgb_targets)
        rgb_loss = sum(v for v in rgb_loss_dict.values()) * self.config.loss.p1_gt_weight

        # ir_teacher train trên MID
        mid_batch = self._next_mid()
        mid_images, mid_targets = self._geometric_aug(
            mid_batch.images, mid_batch.targets, self.config.rgb_aug
        )
        mid_loss_dict = self.ir_teacher(mid_images, mid_targets)
        mid_loss = sum(v for v in mid_loss_dict.values()) * self.config.loss.p1_gt_weight

        total_loss = rgb_loss + mid_loss
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.rgb_teacher.parameters()) + list(self.ir_teacher.parameters()),
            self.config.grad_clip if self.config.grad_clip > 0 else float("inf"),
        )
        self.teacher_optimizer.step()

        log: Dict = {
            "p1_rgb_teacher_loss": rgb_loss.item(),
            "p1_ir_teacher_loss":  mid_loss.item(),
            "p1_total_loss":       total_loss.item(),
            "domain": "rgb+mid",
        }
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

        weak_images, aug_targets = self._geometric_aug(
            mixed.images, mixed.targets, self.config.rgb_aug
        )
        strong_images = self._rgb_photometric_aug(weak_images.clone())

        det_loss, log = compute_rgb_mid_loss(
            student=self.student,
            mixed_images=strong_images,
            gt_targets=aug_targets,
            n_rgb=mixed.n_rgb,
            rgb_teacher=self.rgb_teacher,
            ir_teacher=self.ir_teacher,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher="both"),
            teacher_images=weak_images,
            aat_generator=self.aat_generator,
        )

        det_loss.backward()
        log["grad_norm"] = self._clip_and_step()

        if self.config.teacher_update.p2_update_rgb_teacher:
            log.update(self._update_teacher(self.rgb_teacher, weak_images, phase))
        if self.config.teacher_update.p2_update_ir_teacher:
            log.update(self._update_teacher(self.ir_teacher, weak_images, phase))

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

        det_loss, log = compute_mid_ir_loss(
            student=self.student,
            mixed_images=strong_images,
            mid_targets=aug_mid_targets,
            n_mid=n_mid_aug,
            rgb_teacher=self.rgb_teacher,
            ir_teacher=self.ir_teacher,
            config=self.config.loss,
            conf_thresh=self._get_threshold(phase, teacher="both"),
            teacher_images=weak_images,
            aat_generator=self.aat_generator,
        )

        det_loss.backward()
        log["grad_norm"] = self._clip_and_step()

        if self.config.teacher_update.p3_update_rgb_teacher:
            log.update(self._update_teacher(self.rgb_teacher, weak_images, phase))
        if self.config.teacher_update.p3_update_ir_teacher:
            log.update(self._update_teacher(self.ir_teacher, weak_images, phase))
        # AAT: refresh IR reference statistics with real IR images
        if self.aat_generator is not None and weak_ir is not None:
            self.aat_generator.update_ir_stats(weak_ir, momentum=self.config.aat.ir_stats_momentum)

        log["domain"] = "mid_ir"
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
            log = self.train_teachers_step(phase=phase)
        elif step == "rgb_mid":
            log = self.train_rgb_mid_step(phase=phase)
        elif step == "mid_ir":
            log = self.train_mid_ir_step(phase=phase)
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

    def save_checkpoint(self, path: str) -> None:
        ckpt = {
            "global_step": self.global_step,
            "student":     self.student.state_dict(),
            "rgb_teacher": self.rgb_teacher.state_dict(),
            "ir_teacher":  self.ir_teacher.state_dict(),
            "optimizer":   self.optimizer.state_dict(),
        }
        torch.save(ckpt, path)
        logger.info(f"Checkpoint saved → {path}  (step {self.global_step})")

    def load_checkpoint(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.global_step = ckpt["global_step"]
        self.student.load_state_dict(ckpt["student"])
        self.rgb_teacher.load_state_dict(ckpt["rgb_teacher"])
        self.ir_teacher.load_state_dict(ckpt["ir_teacher"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        logger.info(f"Checkpoint loaded ← {path}  (step {self.global_step})")

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
