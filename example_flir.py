"""
example_flir.py — Curriculum DA training on FLIR ADAS Aligned dataset.

Dataset structure (align/ directory):
  align/
  ├── JPEGImages/
  │   ├── FLIR_XXXXX_PreviewData.jpeg   ← IR (thermal) images
  │   ├── FLIR_XXXXX_RGB.jpg            ← RGB images (paired, spatially aligned)
  │   └── ...
  ├── Annotations/
  │   ├── FLIR_XXXXX_PreviewData.xml    ← VOC XML annotations (for IR)
  │   └── ...
  └── ImageSets/Main/
      ├── align_train.txt               ← 4129 training stems
      └── align_validation.txt          ← 1013 validation stems

Classes (FCOS 0-indexed):
  0=person  1=car  2=bicycle

Run (từ trong thư mục DAOD-RGB2IR/):
  python example_flir.py --data_root /path/to/align --device mps

Run (từ thư mục cha DomainAdaptation/):
  python DAOD-RGB2IR/example_flir.py --data_root /path/to/align --device mps
"""

import argparse
import json
import logging
import os
import random
import sys
from pathlib import Path

# Đảm bảo thư mục của script luôn nằm trong sys.path,
# cho dù chạy từ thư mục nào.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from adaptive_threshold import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdScheduler,
    TeacherThresholds,
    ThreshRampConfig,
)
from config import (
    AATConfig,
    AblationConfig,
    AdvConfig,
    CurriculumConfig,
    EMAConfig,
    IRAugConfig,
    LossConfig,
    RGBAugConfig,
    SAGAConfig,
    TeacherUpdateConfig,
    TrainingConfig,
)
from discriminator import DomainDiscriminator
from datasets import (
    FLIR_CLASSES,
    FLIR_TO_COCO_IDX,
    NUM_CLASSES,
    FLIRIRDataset,
    FLIRIRValDataset,
    FLIRRGBDataset,
    ir_collate,
    ir_val_collate,
    rgb_collate,
)
from ema import copy_student_to_teacher
from evaluator import DetectionEvaluator, PhaseEvaluator
from faster_rcnn_wrapper import build_faster_rcnn_trio
from fcos_wrapper import build_fcos_trio
from scheduler import Phase
from trainer import CurriculumDomainAdaptationTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def configure_logging(log_file: str = None) -> None:
    if not log_file:
        return
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    log_path = os.path.abspath(log_file)
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and handler.baseFilename == log_path:
            return
    file_handler = logging.FileHandler(log_path, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s  %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(file_handler)


def _looks_like_flir_align(path: Path) -> bool:
    has_split_files = (
        (path / "align_train.txt").exists()
        and (path / "align_validation.txt").exists()
    ) or (
        (path / "ImageSets" / "Main" / "align_train.txt").exists()
        and (path / "ImageSets" / "Main" / "align_validation.txt").exists()
    )
    return (
        has_split_files
        and (path / "JPEGImages").is_dir()
        and (path / "Annotations").is_dir()
    )


def resolve_data_root(data_root: str = None) -> str:
    candidates = []
    if data_root:
        candidates.append(Path(data_root))
    for base in (Path("/kaggle/input/flir-aligned"), Path("/kaggle/input")):
        if base.exists():
            candidates.append(base)
            candidates.extend(p for p in base.rglob("*") if p.is_dir())

    seen = set()
    checked = []
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        if len(checked) < 20:
            checked.append(str(candidate))
        if _looks_like_flir_align(candidate):
            return str(candidate)
    raise FileNotFoundError(
        "Could not find FLIR aligned data root. Pass --data_root pointing to a "
        "directory containing JPEGImages/, Annotations/, and either root-level "
        "align_train.txt/align_validation.txt or ImageSets/Main split files. "
        f"Checked candidates: {checked}"
    )


def capture_rng_state() -> dict:
    state = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def restore_rng_state(state: dict) -> None:
    if not state:
        return
    try:
        if "python" in state:
            random.setstate(state["python"])
    except Exception as exc:
        logger.warning(f"Skipping Python RNG restore: {exc}")

    try:
        torch_state = state.get("torch")
        if torch_state is not None:
            if not isinstance(torch_state, torch.Tensor):
                torch_state = torch.ByteTensor(torch_state)
            torch.set_rng_state(torch_state.cpu().to(torch.uint8))
    except Exception as exc:
        logger.warning(f"Skipping torch RNG restore: {exc}")

    try:
        cuda_state = state.get("cuda")
        if torch.cuda.is_available() and cuda_state is not None:
            cuda_state = [
                s if isinstance(s, torch.Tensor) else torch.ByteTensor(s)
                for s in cuda_state
            ]
            torch.cuda.set_rng_state_all(cuda_state)
    except Exception as exc:
        logger.warning(f"Skipping CUDA RNG restore: {exc}")


def write_metrics(path: str, phase_eval) -> None:
    if phase_eval is None or not path:
        return
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(phase_eval.history, f, indent=2)


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def make_training_config(
    device: str,
    target_h: int = 512,
    target_w: int = 640,
    ema_mode: str = "aema",
    ema_alpha: float = 0.9996,
    aema_fast_alpha: float = 0.997,
    aema_slow_alpha: float = 0.9996,
    aema_top_ratio: float = 0.10,
    aema_update_interval: int = 2,
    phase1_end: int = 15_000,
    phase2_end: int = 20_000,
    phase3_end: int = 25_000,
    skip_empty_pseudo: bool = True,
    disable_aema: bool = False,
    disable_adv: bool = False,
    disable_teacher_student_merge: bool = False,
    merge_iou_threshold: float = 0.5,
    merge_student_conf_thresh: float = 0.7,
    aat_enabled: bool = False,
    aat_epsilon: float = 0.02,
    aat_merge_iou: float = 0.5,
) -> TrainingConfig:
    return TrainingConfig(
        ema=EMAConfig(
            mode=ema_mode,
            alpha=ema_alpha,
            use_warmup=True,
            aema_fast_alpha=aema_fast_alpha,
            aema_slow_alpha=aema_slow_alpha,
            aema_top_ratio=aema_top_ratio,
            aema_update_interval=aema_update_interval,
            merge_iou_threshold=merge_iou_threshold,
            merge_student_conf_thresh=merge_student_conf_thresh,
        ),
        saga=SAGAConfig(apply_prob=1.0),   # SAGA applied 100% in MID phase
        rgb_aug=RGBAugConfig(
            hflip_prob=0.5,
            multiscale_min=0.5,
            multiscale_max=1.5,
            multiscale_target_h=target_h,
            multiscale_target_w=target_w,
            blur_prob=0.5,
            blur_sigma_max=1.0,
            color_jitter_prob=0.5,
            cj_brightness=0.2,
            cj_contrast=0.2,
            cj_saturation=0.3,
            cj_hue=0.05,
            random_erasing_prob=0.3,
            random_erasing_scale_min=0.02,
            random_erasing_scale_max=0.10,
        ),
        ir_aug=IRAugConfig(
            hflip_prob=0.5,
            multiscale_min=0.5,
            multiscale_max=1.5,
            multiscale_target_h=target_h,
            multiscale_target_w=target_w,
            intensity_shift_prob=0.5,
            intensity_shift_mag=0.1,
            contrast_jitter_prob=0.5,
            contrast_jitter_mag=0.2,
            gamma_prob=0.3,
            gamma_min=0.7,
            gamma_max=1.3,
            gaussian_noise_prob=0.3,
            gaussian_noise_std=0.02,
        ),
        curriculum=CurriculumConfig(
            phase1_end=phase1_end,    # RGB warmup
            phase2_end=phase2_end,    # mixed [RGB | MID]
            phase3_end=phase3_end,    # mixed [MID | IR]
            # Phase 4: IR focus until total_iters
            phase2_rgb_ratio=0.5, # 50% RGB + 50% MID per Phase-2 batch
            phase3_mid_ratio=0.5, # 50% MID + 50% IR per Phase-3 batch
        ),
        loss=LossConfig(
            # Phase 1 — RGB warmup
            p1_gt_weight=1.0,
            p1_pseudo_weight=0.0,
            # Phase 2 — mixed [RGB | MID]
            p2_gt_weight=1.0,
            p2_rgb_teacher_weight=0.4,
            p2_ir_teacher_weight=0.1,
            # Phase 3 — mixed [MID | IR]
            p3_gt_weight=1.0,
            p3_rgb_teacher_weight=0.1,
            p3_ir_teacher_weight=0.4,
            # Phase 4 — IR focus
            p4_ir_teacher_weight=1.0,
        ),
        pseudo_label_conf_thresh=0.7,
        device=device,
        log_interval=100,
        ablation=AblationConfig(
            skip_empty_pseudo=skip_empty_pseudo,
            disable_aema=disable_aema,
            disable_adv=disable_adv,
            disable_teacher_student_merge=disable_teacher_student_merge,
        ),
        aat=AATConfig(
            enabled=aat_enabled,
            epsilon=aat_epsilon,
            merge_iou=aat_merge_iou,
        ),
    )


def make_adaptive_threshold(
    phase4_ramp_enabled: bool = False,
    phase4_thresh_person: float = 0.55,
    phase4_thresh_car: float = 0.65,
    phase4_thresh_bicycle: float = 0.50,
    phase2_thresh_person: float = 0.55,
    phase2_thresh_car: float = 0.60,
    phase2_thresh_bicycle: float = 0.50,
    phase3_thresh_person: float = 0.55,
    phase3_thresh_car: float = 0.60,
    phase3_thresh_bicycle: float = 0.50,
) -> AdaptiveThresholdScheduler:
    # FLIR classes: 0=person  1=car  2=bicycle
    # person: harder to detect in IR → lower threshold
    # car: most distinct in IR → higher threshold
    # bicycle: small, rare → lower threshold
    phase4_base = {0: phase4_thresh_person, 1: phase4_thresh_car, 2: phase4_thresh_bicycle}
    phase23_base = {0: phase2_thresh_person, 1: phase2_thresh_car, 2: phase2_thresh_bicycle}
    phase3_base = {0: phase3_thresh_person, 1: phase3_thresh_car, 2: phase3_thresh_bicycle}
    return AdaptiveThresholdScheduler(AdaptiveThresholdConfig(
        rgb_teacher=TeacherThresholds(
            phase1={0: 0.70, 1: 0.70, 2: 0.65},
            phase2=phase23_base,
            phase3=phase3_base,
            phase4=phase4_base,
        ),
        ir_teacher=TeacherThresholds(
            phase1={0: 0.70, 1: 0.70, 2: 0.65},
            phase2=phase23_base,
            phase3=phase3_base,
            phase4=phase4_base,
        ),
        phase4_ir_ramp=ThreshRampConfig(
            enabled=phase4_ramp_enabled,
            # per-class: person(0) / car(1) / bicycle(2)
            start={0: phase4_thresh_person, 1: phase4_thresh_car, 2: phase4_thresh_bicycle},
            end  ={0: 0.85, 1: 0.90, 2: 0.80},
            ramp_steps=10_000,
        ),
    ))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    configure_logging(args.log_file)
    device_str = args.device
    device     = torch.device(device_str)
    data_root  = resolve_data_root(args.data_root)
    latest_path = os.path.join(args.output_dir, "latest.pt")
    metrics_path = args.metrics_file or os.path.join(args.output_dir, "metrics_history.json")

    logger.info("=== Curriculum DA — FLIR ADAS Aligned ===")
    logger.info(f"Data root : {data_root}")
    logger.info(f"Device    : {device_str}")
    logger.info(f"Output dir: {args.output_dir}")
    if args.log_file:
        logger.info(f"Log file  : {os.path.abspath(args.log_file)}")
    logger.info(f"Classes   : {FLIR_CLASSES}  (num_classes={NUM_CLASSES})")

    # --- Datasets ---
    logger.info("Loading datasets ...")
    rgb_train = FLIRRGBDataset(data_root, split="train")
    ir_train  = FLIRIRDataset( data_root, split="train")
    ir_val    = FLIRIRValDataset(data_root, split="validation")
    rgb_val   = FLIRRGBDataset( data_root, split="validation")

    logger.info(
        f"  RGB train     : {len(rgb_train):>5} images  (labeled, source domain)\n"
        f"  IR  train     : {len(ir_train):>5} images  (unlabeled, target domain)\n"
        f"  IR  val       : {len(ir_val):>5} images  (labeled, for mAP)\n"
        f"  RGB val       : {len(rgb_val):>5} images  (labeled, Phase 1 eval)"
    )

    # --- DataLoaders ---
    rgb_loader = DataLoader(
        rgb_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=rgb_collate, num_workers=args.workers, drop_last=True,
        pin_memory=(device_str == "cuda"),
    )
    ir_loader = DataLoader(
        ir_train, batch_size=args.batch_size, shuffle=True,
        collate_fn=ir_collate, num_workers=args.workers, drop_last=True,
        pin_memory=(device_str == "cuda"),
    )
    ir_val_loader = DataLoader(
        ir_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=ir_val_collate, num_workers=args.workers,
    )
    rgb_val_loader = DataLoader(
        rgb_val, batch_size=args.batch_size, shuffle=False,
        collate_fn=rgb_collate, num_workers=args.workers,
    )

    # --- Models ---
    logger.info(f"Building {args.model.upper()} trio ...")
    _trio_kwargs = dict(
        num_classes=NUM_CLASSES,
        pretrained_backbone=not args.no_pretrained_backbone,
        trainable_backbone_layers=3,
        min_size=args.min_size,
        max_size=args.max_size,
        ir_to_rgb=True,
        from_coco=args.from_coco,
        coco_src_indices=FLIR_TO_COCO_IDX if args.from_coco else None,
    )
    if args.model == "faster_rcnn":
        student, rgb_teacher, ir_teacher = build_faster_rcnn_trio(
            **_trio_kwargs,
            focal_gamma=args.focal_gamma,
        )
    else:
        student, rgb_teacher, ir_teacher = build_fcos_trio(**_trio_kwargs)
    copy_student_to_teacher(rgb_teacher, student)
    copy_student_to_teacher(ir_teacher,  student)

    # --- Multi-GPU: wrap with DataParallel if >1 GPU ---
    n_gpus = torch.cuda.device_count() if device_str == "cuda" else 0
    if n_gpus > 1:
        logger.info(f"Multi-GPU: wrapping models with DataParallel ({n_gpus} GPUs)")
        student    = nn.DataParallel(student)
        rgb_teacher = nn.DataParallel(rgb_teacher)
        ir_teacher  = nn.DataParallel(ir_teacher)

    # --- Optimizer ---
    optimizer = torch.optim.SGD([
        {"params": [p for n, p in student.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": args.lr_backbone},
        {"params": [p for n, p in student.named_parameters()
                    if "backbone" not in n and p.requires_grad],
         "lr": args.lr_head},
    ], momentum=0.9, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iters,
    )

    # --- Adaptive threshold ---
    thresh = make_adaptive_threshold(
        phase4_ramp_enabled=args.phase4_threshold_ramp,
        phase4_thresh_person=args.phase4_thresh_person,
        phase4_thresh_car=args.phase4_thresh_car,
        phase4_thresh_bicycle=args.phase4_thresh_bicycle,
        phase2_thresh_person=args.phase2_thresh_person,
        phase2_thresh_car=args.phase2_thresh_car,
        phase2_thresh_bicycle=args.phase2_thresh_bicycle,
        phase3_thresh_person=args.phase3_thresh_person,
        phase3_thresh_car=args.phase3_thresh_car,
        phase3_thresh_bicycle=args.phase3_thresh_bicycle,
    )
    logger.info("\n" + thresh.summary())

    # --- Evaluator ---
    evaluator = DetectionEvaluator(
        num_classes=NUM_CLASSES,
        class_names=FLIR_CLASSES,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    _cfg = make_training_config(
        device_str,
        target_h=args.min_size,
        target_w=args.max_size,
        ema_mode=args.ema_mode,
        ema_alpha=args.ema_alpha,
        aema_fast_alpha=args.aema_fast_alpha,
        aema_slow_alpha=args.aema_slow_alpha,
        aema_top_ratio=args.aema_top_ratio,
        aema_update_interval=args.aema_update_interval,
        phase1_end=args.phase1_end,
        phase2_end=args.phase2_end,
        phase3_end=args.phase3_end,
        skip_empty_pseudo=not args.no_skip_empty_pseudo,
        disable_aema=args.disable_aema,
        disable_adv=args.disable_adv or args.adv_weight == 0.0,
        disable_teacher_student_merge=args.disable_teacher_student_merge,
        merge_iou_threshold=args.merge_iou_threshold,
        merge_student_conf_thresh=args.merge_student_conf_thresh,
        aat_enabled=args.enable_aat,
        aat_epsilon=args.aat_epsilon,
        aat_merge_iou=args.aat_merge_iou,
    )
    logger.info(
        f"Teacher update: mode={_cfg.ema.mode}  ema_alpha={_cfg.ema.alpha}  "
        f"aema_fast={_cfg.ema.aema_fast_alpha}  aema_slow={_cfg.ema.aema_slow_alpha}  "
        f"aema_top_ratio={_cfg.ema.aema_top_ratio}  "
        f"aema_update_interval={_cfg.ema.aema_update_interval}"
    )
    phase_eval = None
    if not args.skip_eval:
        phase_eval = PhaseEvaluator(
            evaluator=evaluator,
            ir_val_loader=ir_val_loader,
            device=device,
            eval_every_n=args.eval_every,
            rgb_val_loader=rgb_val_loader,
            vis_dir=os.path.join(args.output_dir, "vis"),
            vis_every_n=args.vis_every,
            vis_num_samples=16,
            class_names=FLIR_CLASSES,
            thresh_scheduler=thresh,
            phase3_end=_cfg.curriculum.phase3_end,
            rgb_teacher=rgb_teacher,
            ir_teacher=ir_teacher,
        )

    # --- Config (reuse _cfg built above for PhaseEvaluator) ---
    config = _cfg
    config.adv = AdvConfig(
        p2_adv_weight=args.adv_weight,
        p3_adv_weight=args.adv_weight,
        disc_hidden=1024,
        backbone_dim=2048,
        disc_lr=args.disc_lr,
        grl_lambda=args.grl_lambda,
        use_schedule=not args.no_grl_schedule,
    )

    # --- Adversarial discriminators (None when adv_weight=0 or --disable_adv) ---
    disc_rgb = disc_ir = disc_optimizer = None
    adv_effectively_disabled = args.disable_adv or args.adv_weight == 0.0
    if args.adv_weight > 0.0 and not args.disable_adv:
        disc_rgb = DomainDiscriminator(
            in_features=config.adv.backbone_dim,
            hidden=config.adv.disc_hidden,
        )
        disc_ir = DomainDiscriminator(
            in_features=config.adv.backbone_dim,
            hidden=config.adv.disc_hidden,
        )
        disc_optimizer = torch.optim.AdamW(
            list(disc_rgb.parameters()) + list(disc_ir.parameters()),
            lr=config.adv.disc_lr,
            weight_decay=1e-4,
        )
        logger.info(
            f"Adversarial training ON  "
            f"adv_weight={args.adv_weight}  "
            f"grl_lambda={args.grl_lambda}  "
            f"schedule={'DANN' if not args.no_grl_schedule else 'fixed'}"
        )
    elif args.disable_adv:
        logger.info("Adversarial training OFF  (--disable_adv)")
    else:
        logger.info("Adversarial training OFF  (--adv_weight 0)")

    # --- Trainer ---
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_loader,
        ir_loader=ir_loader,
        threshold_scheduler=thresh,
        phase_evaluator=phase_eval,
        phase1_best_path=os.path.join(args.output_dir, "best_PHASE1_RGB_WARMUP.pt"),
        phase2_best_path=os.path.join(args.output_dir, "best_PHASE2_RGB_MID.pt"),
        disc_rgb=disc_rgb,
        disc_ir=disc_ir,
        disc_optimizer=disc_optimizer,
    )

    # --- Best checkpoint callbacks ---
    os.makedirs(args.output_dir, exist_ok=True)

    def save_global_best(results):
        step  = results["global_step"]
        phase = results["phase"]
        map50 = results["mAP@0.5"]
        path  = f"{args.output_dir}/best.pt"
        save_training_checkpoint(path)
        logger.info(f"[Global Best] mAP@0.5={map50:.4f}  phase={phase}  step={step}  → {path}")

    def save_phase_best(results):
        step  = results["global_step"]
        phase = results["phase"]
        # Phase 1 best tracked by RGB mAP; all other phases by IR mAP
        if phase == "PHASE1_RGB_WARMUP" and "rgb_mAP@0.5" in results:
            map50       = results["rgb_mAP@0.5"]
            metric_name = "rgb_mAP@0.5"
        else:
            map50       = results["mAP@0.5"]
            metric_name = "mAP@0.5"
        path = f"{args.output_dir}/best_{phase}.pt"
        save_training_checkpoint(path)
        logger.info(f"[Phase Best] {phase}  {metric_name}={map50:.4f}  step={step}  → {path}")

    if phase_eval is not None:
        phase_eval.register_best_fn(save_global_best)
        phase_eval.register_phase_best_fn(save_phase_best)

    def checkpoint_extra_state():
        return {
            "lr_scheduler": lr_scheduler.state_dict(),
            "rng": capture_rng_state(),
            "phase_eval": phase_eval.state_dict() if phase_eval is not None else None,
            "args": vars(args),
        }

    def save_training_checkpoint(path):
        trainer.save_checkpoint(path, extra_state=checkpoint_extra_state())

    # --- Resume ---
    resume_path = args.resume
    if args.auto_resume and resume_path is None and os.path.exists(latest_path):
        resume_path = latest_path

    if resume_path:
        logger.info(f"Resuming from: {resume_path}")
        ckpt = trainer.load_checkpoint(resume_path)
        extra_state = ckpt.get("extra_state", {})
        if "lr_scheduler" in extra_state:
            lr_scheduler.load_state_dict(extra_state["lr_scheduler"])
        else:
            # Backward-compatible resume for older checkpoints.
            optimizer.param_groups[0]["lr"] = args.lr_backbone
            optimizer.param_groups[1]["lr"] = args.lr_head
            lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=args.total_iters, last_epoch=-1,
            )
            for _ in range(trainer.global_step):
                lr_scheduler.step()
        restore_rng_state(extra_state.get("rng", {}))
        if phase_eval is not None and extra_state.get("phase_eval") is not None:
            phase_eval.load_state_dict(extra_state["phase_eval"])
        logger.info(
            f"Resumed at global_step={trainer.global_step}  "
            f"remaining={args.total_iters - trainer.global_step} iters  "
            f"lr={optimizer.param_groups[-1]['lr']:.2e}"
        )
    elif phase_eval is not None:
        # --- Baseline eval (only on fresh run) ---
        logger.info("\nBaseline evaluation (before training) ...")
        phase_eval.evaluate(student, global_step=0,
                            current_phase=Phase.PHASE1_RGB_WARMUP,
                            trigger_reason="baseline")
    else:
        logger.info("\nBaseline evaluation skipped (--skip_eval).")

    # --- Training loop ---
    train_until = args.total_iters
    if args.stop_at is not None:
        train_until = min(args.total_iters, args.stop_at)
    remaining_iters = max(0, train_until - trainer.global_step)
    logger.info(
        f"\nStarting training: {remaining_iters} remaining iterations "
        f"(global_step {trainer.global_step} → {train_until}; total target {args.total_iters}) ..."
    )
    for _ in range(remaining_iters):
        log  = trainer.train_one_iteration()
        lr_scheduler.step()
        step = trainer.global_step  # incremented at end of train_one_iteration

        # Verbose log every 500 iters
        if step % 500 == 0:
            phase  = log.get("phase", "?")
            domain = log.get("domain", "?")
            t_rgb  = log.get("thresh_rgb", "?")
            t_ir   = log.get("thresh_ir",  "?")
            lr     = optimizer.param_groups[-1]["lr"]
            thresh_str = f"({t_rgb:.2f}/{t_ir:.2f})" if isinstance(t_rgb, float) else ""
            logger.info(
                f"[{step:07d}/{args.total_iters}]  "
                f"phase={phase:<22}  domain={domain}  "
                f"thresh={thresh_str}  lr={lr:.2e}"
            )

        # Checkpoint
        if step % args.save_every == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            save_training_checkpoint(f"{args.output_dir}/ckpt_{step:07d}.pt")
            save_training_checkpoint(latest_path)
            write_metrics(metrics_path, phase_eval)

    # --- Final eval + summary ---
    if phase_eval is not None:
        logger.info("\nFinal evaluation ...")
        final_phase = trainer.scheduler.get_phase(max(0, trainer.global_step - 1))
        phase_eval.evaluate(student, global_step=trainer.global_step,
                            current_phase=final_phase,
                            trigger_reason="final")
        phase_eval.print_history()
        write_metrics(metrics_path, phase_eval)
    else:
        logger.info("\nFinal evaluation skipped (--skip_eval).")

    os.makedirs(args.output_dir, exist_ok=True)
    save_training_checkpoint(latest_path)
    if trainer.global_step >= args.total_iters:
        save_training_checkpoint(f"{args.output_dir}/final.pt")
    else:
        logger.info(
            f"Chunk complete at global_step={trainer.global_step}; "
            f"full target remains {args.total_iters}."
        )
    logger.info(f"Done.  global_step={trainer.global_step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   default=None,
                   help="Path to align/ directory. If omitted, common Kaggle input paths are auto-detected.")
    p.add_argument("--output_dir",  default="./output")
    p.add_argument("--total_iters", type=int,   default=35_000)
    p.add_argument("--stop_at", "--stop-at", type=int, default=None,
                   help="Absolute global_step to stop at for chunked Kaggle runs")
    p.add_argument("--log_file", "--log-file", default=None,
                   help="Append Python logger output to this file in addition to stdout")
    p.add_argument("--metrics_file", "--metrics-file", default=None,
                   help="Write PhaseEvaluator history JSON here (default: output_dir/metrics_history.json)")
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--lr_head",     type=float, default=5e-4)
    p.add_argument("--min_size",    type=int,   default=512)
    p.add_argument("--max_size",    type=int,   default=640)
    p.add_argument("--eval_every",  type=int,   default=2_000)
    p.add_argument("--vis_every",   type=int,   default=500)
    p.add_argument("--save_every",  type=int,   default=5_000)
    p.add_argument("--phase1_end", "--phase1-end", type=int, default=15_000,
                   help="Iteration where Phase 1 ends")
    p.add_argument("--phase2_end", "--phase2-end", type=int, default=20_000,
                   help="Iteration where Phase 2 ends")
    p.add_argument("--phase3_end", "--phase3-end", type=int, default=25_000,
                   help="Iteration where Phase 3 ends")
    p.add_argument("--model",       default="fcos",
                   choices=["fcos", "faster_rcnn"],
                   help="Detector backbone (default: fcos)")
    p.add_argument("--from_coco",   action="store_true",
                   help="Init head from COCO pretrained weights (91-class → replace head)")
    p.add_argument("--no_pretrained_backbone", action="store_true",
                   help="Do not download/load ImageNet pretrained ResNet50-FPN backbone weights")
    p.add_argument("--focal_gamma", type=float, default=2.0,
                   help="Focal loss gamma for faster_rcnn classifier (default 2.0, 0=cross-entropy)")
    p.add_argument("--adv_weight",      type=float, default=0.2,
                   help="Adversarial alignment loss weight for Phase 2/3 (default 0.2, 0=disabled)")
    p.add_argument("--disc_lr",         type=float, default=1e-4,
                   help="Discriminator optimizer LR (default 1e-4)")
    p.add_argument("--grl_lambda",      type=float, default=1.0,
                   help="Max GRL lambda (default 1.0)")
    p.add_argument("--no_grl_schedule", action="store_true",
                   help="Use fixed GRL lambda instead of DANN progressive schedule")
    p.add_argument("--ema_mode", "--ema-mode", default="aema", choices=["aema", "ema"],
                   help="Teacher update mode: DDT-style AEMA or classic EMA")
    p.add_argument("--ema_alpha", "--ema-alpha", type=float, default=0.9996,
                   help="Classic EMA alpha when --ema_mode ema")
    p.add_argument("--aema_fast_alpha", "--aema-fast-alpha", type=float, default=0.997,
                   help="AEMA alpha for high-gradient teacher parameters")
    p.add_argument("--aema_slow_alpha", "--aema-slow-alpha", type=float, default=0.9996,
                   help="AEMA alpha for low-gradient teacher parameters")
    p.add_argument("--aema_top_ratio", "--aema-top-ratio", type=float, default=0.10,
                   help="Fraction of globally highest accumulated teacher gradients using fast alpha")
    p.add_argument("--aema_update_interval", "--aema-update-interval", type=int, default=2,
                   help="AEMA gradient accumulation steps before applying a teacher update")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from (e.g. output/best_PHASE1_RGB_WARMUP.pt)")
    p.add_argument("--auto_resume", "--auto-resume", action="store_true",
                   help="Resume output_dir/latest.pt automatically when it exists")
    p.add_argument("--skip_eval", "--skip-eval", action="store_true",
                   help="Skip baseline/final/periodic validation for fast smoke runs")
    # --- AAT (Adversarial Attacked Teacher) ---
    p.add_argument("--enable_aat", "--enable-aat", action="store_true",
                   help="Enable statistical domain-match adversarial attacked teacher")
    p.add_argument("--aat_epsilon", "--aat-epsilon", type=float, default=0.02,
                   help="AAT perturbation budget (default 0.02)")
    p.add_argument("--aat_merge_iou", "--aat-merge-iou", type=float, default=0.5,
                   help="IoU threshold for merging clean + attacked pseudo (default 0.5)")
    p.add_argument("--device",      default="cuda",
                   choices=["cuda", "cpu", "mps"])
    # --- Ablation / debug flags ---
    p.add_argument("--disable_aema", "--disable-aema", action="store_true",
                   help="Fall back to classic EMA instead of AEMA")
    p.add_argument("--disable_adv", "--disable-adv", action="store_true",
                   help="Disable adversarial domain alignment (GRL)")
    p.add_argument("--disable_teacher_student_merge", "--disable-teacher-student-merge",
                   action="store_true",
                   help="Use teacher-only pseudo for AEMA importance (no DDT merge)")
    p.add_argument("--no_skip_empty_pseudo", "--no-skip-empty-pseudo", action="store_true",
                   help="Do NOT skip all-empty pseudo batches (default: skip is enabled)")
    p.add_argument("--phase4_threshold_ramp", "--phase4-threshold-ramp", action="store_true",
                   help="Enable Phase-4 linear threshold ramp-up (disabled by default)")
    p.add_argument("--phase4_thresh_person", "--phase4-thresh-person", type=float, default=0.55,
                   help="Phase-4 IR confidence threshold for person (default 0.55)")
    p.add_argument("--phase4_thresh_car", "--phase4-thresh-car", type=float, default=0.65,
                   help="Phase-4 IR confidence threshold for car (default 0.65)")
    p.add_argument("--phase4_thresh_bicycle", "--phase4-thresh-bicycle", type=float, default=0.50,
                   help="Phase-4 IR confidence threshold for bicycle (default 0.50)")
    p.add_argument("--phase2_thresh_person", "--phase2-thresh-person", type=float, default=0.55,
                   help="Phase-2 confidence threshold for person (default 0.55)")
    p.add_argument("--phase2_thresh_car", "--phase2-thresh-car", type=float, default=0.60,
                   help="Phase-2 confidence threshold for car (default 0.60)")
    p.add_argument("--phase2_thresh_bicycle", "--phase2-thresh-bicycle", type=float, default=0.50,
                   help="Phase-2 confidence threshold for bicycle (default 0.50)")
    p.add_argument("--phase3_thresh_person", "--phase3-thresh-person", type=float, default=0.55,
                   help="Phase-3 confidence threshold for person (default 0.55)")
    p.add_argument("--phase3_thresh_car", "--phase3-thresh-car", type=float, default=0.60,
                   help="Phase-3 confidence threshold for car (default 0.60)")
    p.add_argument("--phase3_thresh_bicycle", "--phase3-thresh-bicycle", type=float, default=0.50,
                   help="Phase-3 confidence threshold for bicycle (default 0.50)")
    p.add_argument("--merge_iou_threshold", "--merge-iou-threshold", type=float, default=0.5,
                   help="IoU threshold for teacher-student pseudo merge (default 0.5)")
    p.add_argument("--merge_student_conf_thresh", "--merge-student-conf-thresh",
                   type=float, default=0.7,
                   help="Min confidence for student detections in merge (default 0.7)")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
