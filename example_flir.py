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
  python example_flir.py --data_root /path/to/align --device cuda

Run (từ thư mục cha DomainAdaptation/):
  python DAOD-RGB2IR/example_flir.py --data_root /path/to/align --device cuda
"""

import argparse
import json
import logging
import os
import sys
import torch

# Đảm bảo thư mục của script luôn nằm trong sys.path,
# cho dù chạy từ thư mục nào.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Optional

from torch.utils.data import DataLoader

from adaptive_threshold import (
    AdaptiveThresholdConfig,
    AdaptiveThresholdScheduler,
    TeacherThresholds,
)
from config import (
    AATConfig,
    CurriculumConfig,
    EMAConfig,
    GANConfig,
    IRAugConfig,
    LossConfig,
    RGBAugConfig,
    TeacherUpdateConfig,
    TrainingConfig,
)
from gan_translator import GANTranslator
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


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

def make_training_config(device: str, target_h: int = 512, target_w: int = 640,
                         gan_checkpoint: str = "",
                         ema_mode: str = "ema", ema_alpha: float = 0.9996,
                         aema_fast_alpha: float = 0.997, aema_slow_alpha: float = 0.9996,
                         aema_top_ratio: float = 0.10, aema_update_interval: int = 2,
                         aat_enabled: bool = False, aat_mode: str = "statistical",
                         aat_epsilon: float = 0.02, aat_merge_iou: float = 0.5,
                         phase1_end: int = 10_000, phase2_end: int = 18_000, phase3_end: int = 30_000) -> TrainingConfig:
    return TrainingConfig(
        ema=EMAConfig(alpha=ema_alpha, use_warmup=True, mode=ema_mode,
                      aema_fast_alpha=aema_fast_alpha, aema_slow_alpha=aema_slow_alpha,
                      aema_top_ratio=aema_top_ratio, aema_update_interval=aema_update_interval),
        gan=GANConfig(
            checkpoint_path=gan_checkpoint,
            input_nc=3,
            output_nc=1,
            ngf=64,
            n_blocks=9,
        ),
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
        ),
        aat=AATConfig(enabled=aat_enabled, mode=aat_mode,
                      epsilon=aat_epsilon, merge_iou=aat_merge_iou),
        pseudo_label_conf_thresh=0.7,
        device=device,
        log_interval=100,
    )


def make_adaptive_threshold() -> AdaptiveThresholdScheduler:
    # FLIR classes: 0=person  1=car  2=bicycle
    # person: harder to detect in IR → lower threshold
    # car: most distinct in IR → higher threshold
    # bicycle: small, rare → lower threshold
    return AdaptiveThresholdScheduler(AdaptiveThresholdConfig(
        rgb_teacher=TeacherThresholds(
            phase1={0: 0.70, 1: 0.70, 2: 0.65},
            phase2={0: 0.70, 1: 0.75, 2: 0.65},
            phase3={0: 0.75, 1: 0.80, 2: 0.70},
        ),
        ir_teacher=TeacherThresholds(
            phase1={0: 0.70, 1: 0.70, 2: 0.65},
            phase2={0: 0.70, 1: 0.75, 2: 0.65},
            phase3={0: 0.75, 1: 0.80, 2: 0.70},
        ),
    ))


# ---------------------------------------------------------------------------
# Teacher pretraining
# ---------------------------------------------------------------------------

def pretrain_teacher(
    model: torch.nn.Module,
    loader: DataLoader,
    total_iters: int,
    device: torch.device,
    lr_backbone: float,
    lr_head: float,
    label: str,
    save_path: str,
    ir_val_loader: Optional[DataLoader] = None,
    evaluator: Optional["DetectionEvaluator"] = None,
) -> None:
    """
    Supervised pretraining cho một teacher trên loader cho trước.

    Dùng cho:
      - rgb_teacher  : loader = rgb_loader  (labeled RGB images + GT boxes)
      - ir_teacher   : loader = mid_loader  (MID images do GAN sinh + GT boxes từ RGB)

    Nếu ir_val_loader và evaluator được cung cấp, tại mỗi log checkpoint sẽ
    eval mAP teacher trên IR val set và log kết quả.

    Args:
        model          : teacher model (FCOSDetector hoặc FasterRCNNDetector)
        loader         : DataLoader trả (images [B,3,H,W], targets List[Dict])
        total_iters    : số iteration pretrain
        device         : cuda/cpu
        lr_backbone, lr_head : learning rates
        label          : tên log (ví dụ "rgb_teacher", "ir_teacher")
        save_path      : đường dẫn lưu checkpoint sau pretrain
        ir_val_loader  : IR val DataLoader để eval mAP trong burn-in
        evaluator      : DetectionEvaluator instance (dùng chung, reset trước mỗi lần eval)
    """
    model.to(device)
    model.train()
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.SGD([
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" in n and p.requires_grad],
         "lr": lr_backbone},
        {"params": [p for n, p in model.named_parameters()
                    if "backbone" not in n and p.requires_grad],
         "lr": lr_head},
    ], momentum=0.9, weight_decay=1e-4)

    lr_sched = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_iters)

    data_iter = iter(loader)
    log_every = max(1, total_iters // 20)
    do_ir_eval = ir_val_loader is not None and evaluator is not None

    logger.info(f"[Pretrain {label}] starting {total_iters} iters ...")
    for step in range(1, total_iters + 1):
        try:
            images, targets = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            images, targets = next(data_iter)

        images  = images.to(device)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                    for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        loss_dict = model(images, targets)
        loss = sum(v for v in loss_dict.values())
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        optimizer.step()
        lr_sched.step()

        if step % log_every == 0 or step == total_iters:
            logger.info(
                f"[Pretrain {label}] [{step:06d}/{total_iters}]  "
                f"loss={loss.item():.4f}  lr={optimizer.param_groups[-1]['lr']:.2e}"
            )
            if do_ir_eval:
                model.eval()
                evaluator.reset()
                with torch.no_grad():
                    for ir_images, ir_targets in ir_val_loader:
                        ir_images = ir_images.to(device)
                        preds = model(ir_images)
                        evaluator.update(preds, [{k: v for k, v in t.items()} for t in ir_targets])
                ir_results = evaluator.compute()
                map50   = ir_results.get("mAP@0.5", 0.0)
                per_cls = "  ".join(
                    f"{k.split('/')[-1]}={v:.4f}"
                    for k, v in sorted(ir_results.items()) if k.startswith("AP@0.5/")
                )
                logger.info(
                    f"[Pretrain {label}] [IR val @ {step:06d}]  "
                    f"mAP@0.5={map50:.4f}  {per_cls}"
                )
                model.train()

    # Freeze và eval sau pretrain
    for p in model.parameters():
        p.requires_grad = False
    model.eval()

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    torch.save({"teacher": model.state_dict()}, save_path)
    logger.info(f"[Pretrain {label}] done — saved to {save_path}")


class MIDDataset(torch.utils.data.Dataset):
    """
    Wrapper dataset: lấy ảnh RGB + GT từ FLIRRGBDataset,
    dùng GAN translator để sinh ảnh MID on-the-fly.

    Dùng để pretrain ir_teacher trên MID domain.
    """

    def __init__(self, rgb_dataset, gan_translator: "GANTranslator"):
        self.rgb_dataset    = rgb_dataset
        self.gan_translator = gan_translator

    def __len__(self):
        return len(self.rgb_dataset)

    def __getitem__(self, idx):
        image, target = self.rgb_dataset[idx]         # [3, H, W], dict
        mid = self.gan_translator.apply_to_batch(
            image.unsqueeze(0)                        # [1, 3, H, W]
        ).squeeze(0)                                  # [3, H, W]
        return mid, target

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    if getattr(args, "log_file", None):
        os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
        _fh = logging.FileHandler(args.log_file, mode="a")
        _fh.setLevel(logging.INFO)
        _fh.setFormatter(logging.Formatter("%(asctime)s  %(message)s", datefmt="%H:%M:%S"))
        logging.getLogger().addHandler(_fh)
    device_str = args.device
    device     = torch.device(device_str)
    data_root  = args.data_root

    logger.info("=== Curriculum DA — FLIR ADAS Aligned ===")
    logger.info(f"Data root : {data_root}")
    logger.info(f"Device    : {device_str}")
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

    # --- GAN Translator (cần trước khi build MIDDataset) ---
    gan_device = device
    if args.gan_checkpoint:
        logger.info(f"Loading GAN translator from: {args.gan_checkpoint}")
        gan_translator = GANTranslator.from_checkpoint(
            checkpoint_path=args.gan_checkpoint,
            input_nc=3,
            output_nc=1,
            ngf=64,
            n_blocks=9,
            device=gan_device,
            amp=args.gan_amp,
        )
    else:
        logger.warning(
            "No --gan_checkpoint provided — MID images will be identical to RGB (passthrough). "
            "Pass --gan_checkpoint to enable GAN-based RGB→MID translation."
        )
        gan_translator = GANTranslator(generator=None, device=gan_device)

    # --- Models ---
    logger.info(f"Building {args.model.upper()} trio ...")
    _trio_kwargs = dict(
        num_classes=NUM_CLASSES,
        pretrained_backbone=True,
        trainable_backbone_layers=3,
        min_size=args.min_size,
        max_size=args.max_size,
        ir_to_rgb=True,
        from_coco=args.from_coco,
        coco_src_indices=FLIR_TO_COCO_IDX if args.from_coco else None,
    )
    if args.model == "faster_rcnn":
        student, rgb_teacher, ir_teacher = build_faster_rcnn_trio(
            **_trio_kwargs, focal_gamma=args.focal_gamma
        )
    else:
        student, rgb_teacher, ir_teacher = build_fcos_trio(**_trio_kwargs)

    # --- MID DataLoader (GAN-translated RGB, dùng trong Phase 1 cho ir_teacher) ---
    mid_dataset = MIDDataset(rgb_train, gan_translator)
    mid_loader  = DataLoader(
        mid_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=rgb_collate, num_workers=0, drop_last=True,
        pin_memory=(device_str == "cuda"),
    )

    # --- Optimizer cho student (Phase 2+) ---
    # Lúc này student chưa có requires_grad=True (trainer._setup_models sẽ freeze student),
    # nên truyền tất cả params — trainer sẽ unfreeze đúng lúc.
    optimizer = torch.optim.SGD([
        {"params": [p for n, p in student.named_parameters() if "backbone" in n],
         "lr": args.lr_backbone},
        {"params": [p for n, p in student.named_parameters() if "backbone" not in n],
         "lr": args.lr_head},
    ], momentum=0.9, weight_decay=1e-4)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.total_iters,
    )

    # --- Optimizer cho teachers (Phase 1) ---
    teacher_optimizer = torch.optim.SGD(
        [
            {"params": [p for n, p in rgb_teacher.named_parameters() if "backbone" in n],
             "lr": args.lr_backbone},
            {"params": [p for n, p in rgb_teacher.named_parameters() if "backbone" not in n],
             "lr": args.lr_head},
            {"params": [p for n, p in ir_teacher.named_parameters() if "backbone" in n],
             "lr": args.lr_backbone},
            {"params": [p for n, p in ir_teacher.named_parameters() if "backbone" not in n],
             "lr": args.lr_head},
        ],
        momentum=0.9, weight_decay=1e-4,
    )
    teacher_lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        teacher_optimizer, T_max=args.total_iters,
    )

    # --- Adaptive threshold ---
    thresh = make_adaptive_threshold()
    logger.info("\n" + thresh.summary())

    # --- Evaluator ---
    evaluator = DetectionEvaluator(
        num_classes=NUM_CLASSES,
        class_names=FLIR_CLASSES,
        iou_thresholds=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
    )
    _cfg = make_training_config(device_str, target_h=args.min_size, target_w=args.max_size,
                                gan_checkpoint=args.gan_checkpoint,
                                ema_mode=args.ema_mode, ema_alpha=args.ema_alpha,
                                aema_fast_alpha=args.aema_fast_alpha, aema_slow_alpha=args.aema_slow_alpha,
                                aema_top_ratio=args.aema_top_ratio, aema_update_interval=args.aema_update_interval,
                                aat_enabled=args.enable_aat, aat_mode=args.aat_mode,
                                aat_epsilon=args.aat_epsilon, aat_merge_iou=args.aat_merge_iou,
                                phase1_end=args.phase1_end, phase2_end=args.phase2_end, phase3_end=args.phase3_end)
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

    # --- Trainer ---
    trainer = CurriculumDomainAdaptationTrainer(
        student=student,
        rgb_teacher=rgb_teacher,
        ir_teacher=ir_teacher,
        optimizer=optimizer,
        config=config,
        rgb_loader=rgb_loader,
        ir_loader=ir_loader,
        gan_translator=gan_translator,
        teacher_optimizer=teacher_optimizer,
        mid_loader=mid_loader,
        threshold_scheduler=thresh,
        phase_evaluator=phase_eval,
    )

    # --- Best checkpoint callbacks ---
    os.makedirs(args.output_dir, exist_ok=True)

    def save_global_best(results):
        step  = results["global_step"]
        phase = results["phase"]
        map50 = results["mAP@0.5"]
        path  = f"{args.output_dir}/best.pt"
        trainer.save_checkpoint(path)
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
        trainer.save_checkpoint(path)
        logger.info(f"[Phase Best] {phase}  {metric_name}={map50:.4f}  step={step}  → {path}")

    phase_eval.register_best_fn(save_global_best)
    phase_eval.register_phase_best_fn(save_phase_best)

    # --- Resume ---
    if args.resume:
        logger.info(f"Resuming from: {args.resume}")
        trainer.load_checkpoint(args.resume)
        # Reset optimizer LRs to base values so CosineAnnealingLR reads correct base_lrs.
        optimizer.param_groups[0]["lr"] = args.lr_backbone
        optimizer.param_groups[1]["lr"] = args.lr_head
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.total_iters, last_epoch=-1,
        )
        for _ in range(trainer.global_step):
            lr_scheduler.step()
        logger.info(
            f"Resumed at global_step={trainer.global_step}  "
            f"remaining={args.total_iters - trainer.global_step} iters  "
            f"lr={optimizer.param_groups[-1]['lr']:.2e}"
        )
    else:
        # --- Baseline eval (only on fresh run) ---
        logger.info("\nBaseline evaluation (before training) ...")
        phase_eval.evaluate(student, global_step=0,
                            current_phase=Phase.PHASE1_RGB_WARMUP,
                            trigger_reason="baseline")

    # --- Training loop ---
    remaining_iters = args.total_iters - trainer.global_step
    logger.info(
        f"\nStarting training: {remaining_iters} remaining iterations "
        f"(global_step {trainer.global_step} → {args.total_iters}) ..."
    )
    for _ in range(remaining_iters):
        log  = trainer.train_one_iteration()
        step = trainer.global_step  # incremented at end of train_one_iteration

        # Phase 1: step teacher lr; Phase 2+: step student lr
        if log.get("phase") == "PHASE1_RGB_WARMUP":
            teacher_lr_scheduler.step()
        else:
            lr_scheduler.step()

        # Verbose log every 500 iters
        if step % 500 == 0:
            phase  = log.get("phase", "?")
            domain = log.get("domain", "?")
            t_rgb  = log.get("thresh_rgb", "?")
            t_ir   = log.get("thresh_ir",  "?")
            if phase == "PHASE1_RGB_WARMUP":
                lr = teacher_optimizer.param_groups[-1]["lr"]
            else:
                lr = optimizer.param_groups[-1]["lr"]
            thresh_str = f"({t_rgb:.2f}/{t_ir:.2f})" if isinstance(t_rgb, float) else ""
            logger.info(
                f"[{step:07d}/{args.total_iters}]  "
                f"phase={phase:<22}  domain={domain}  "
                f"thresh={thresh_str}  lr={lr:.2e}"
            )

        # Checkpoint
        if step % args.save_every == 0:
            os.makedirs(args.output_dir, exist_ok=True)
            trainer.save_checkpoint(f"{args.output_dir}/ckpt_{step:07d}.pt")

    # --- Final eval + summary ---
    logger.info("\nFinal evaluation ...")
    phase_eval.evaluate(student, global_step=trainer.global_step,
                        current_phase=Phase.PHASE3_MID_IR,
                        trigger_reason="final")
    phase_eval.print_history()

    if getattr(args, "metrics_file", None):
        os.makedirs(os.path.dirname(args.metrics_file) or ".", exist_ok=True)
        with open(args.metrics_file, "w") as _mf:
            json.dump(phase_eval.history, _mf, indent=2)
        logger.info(f"Metrics history saved -> {args.metrics_file}")

    os.makedirs(args.output_dir, exist_ok=True)
    trainer.save_checkpoint(f"{args.output_dir}/final.pt")
    logger.info(f"Done.  global_step={trainer.global_step}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",   required=True,
                   help="Path to align/ directory (contains JPEGImages/, Annotations/, ImageSets/)")
    p.add_argument("--output_dir",  default="./output")
    p.add_argument("--total_iters", type=int,   default=30_000)
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--workers",     type=int,   default=4)
    p.add_argument("--lr_backbone", type=float, default=5e-5)
    p.add_argument("--lr_head",     type=float, default=5e-4)
    p.add_argument("--min_size",    type=int,   default=512)
    p.add_argument("--max_size",    type=int,   default=640)
    p.add_argument("--eval_every",  type=int,   default=1_000)
    p.add_argument("--vis_every",   type=int,   default=1_000)
    p.add_argument("--save_every",  type=int,   default=5_000)
    p.add_argument("--model",       default="fcos",
                   choices=["fcos", "faster_rcnn"],
                   help="Detector backbone (default: fcos)")
    p.add_argument("--from_coco",   action="store_true",
                   help="Init head from COCO pretrained weights (91-class → replace head)")
    p.add_argument("--focal_gamma", type=float, default=0.0,
                   help="Focal loss gamma for faster_rcnn classifier (default 2.0, 0=cross-entropy)")
    p.add_argument("--gan_checkpoint", default="",
                   help="Path to GAN generator checkpoint .pth (e.g. gan_mid/latest_net_G_A.pth). "
                        "If empty, MID = RGB passthrough (for debugging).")
    p.add_argument("--gan_amp",     action="store_true",
                   help="Use torch.autocast during GAN inference (saves VRAM)")
    p.add_argument("--ema_mode", "--ema-mode", default="ema", choices=["ema", "aema"],
                   help="Teacher update: 'ema' classic (default, = repo gốc) or 'aema' adaptive")
    p.add_argument("--ema_alpha", "--ema-alpha", type=float, default=0.9996,
                   help="Classic EMA alpha when --ema_mode ema (default 0.9996)")
    p.add_argument("--aema_fast_alpha", "--aema-fast-alpha", type=float, default=0.997,
                   help="AEMA alpha for top-ratio params (default 0.997)")
    p.add_argument("--aema_slow_alpha", "--aema-slow-alpha", type=float, default=0.9996,
                   help="AEMA alpha for the rest (default 0.9996)")
    p.add_argument("--aema_top_ratio", "--aema-top-ratio", type=float, default=0.10,
                   help="Fraction of params using fast alpha (default 0.10)")
    p.add_argument("--aema_update_interval", "--aema-update-interval", type=int, default=2,
                   help="AEMA gradient accumulation steps per update (default 2)")
    p.add_argument("--enable_aat", "--enable-aat", action="store_true",
                   help="Enable AAT statistical domain-match attacked teacher")
    p.add_argument("--aat_mode", "--aat-mode", default="statistical", choices=["statistical", "discriminator"],
                   help="AAT attack mode (default statistical, no discriminator needed)")
    p.add_argument("--aat_epsilon", "--aat-epsilon", type=float, default=0.02,
                   help="AAT perturbation budget (default 0.02)")
    p.add_argument("--aat_merge_iou", "--aat-merge-iou", type=float, default=0.5,
                   help="IoU threshold for clean+attacked pseudo merge (default 0.5)")
    p.add_argument("--metrics_file", "--metrics-file", default=None,
                   help="Write PhaseEvaluator history JSON here (default: none)")
    p.add_argument("--log_file", "--log-file", default=None,
                   help="Append training log to this file (default: none)")
    p.add_argument("--phase1_end", "--phase1-end", type=int, default=10_000,
                   help="End of Phase 1 RGB warmup (default 10000, = repo gốc)")
    p.add_argument("--phase2_end", "--phase2-end", type=int, default=18_000,
                   help="End of Phase 2 mixed [RGB|MID] (default 18000, = repo gốc)")
    p.add_argument("--phase3_end", "--phase3-end", type=int, default=30_000,
                   help="End of Phase 3 mixed [MID|IR] (default 30000, = repo gốc)")
    p.add_argument("--resume",      default=None,
                   help="Path to checkpoint to resume from (e.g. output/best_PHASE1_RGB_WARMUP.pt)")
    p.add_argument("--device",      default="cuda",
                   choices=["cuda", "cpu", "mps"])
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
