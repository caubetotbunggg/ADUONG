"""
Curriculum Domain Adaptation: RGB → MID(SAGA) → IR
"""

from batch_types import IRBatch, MidBatch, RGBBatch
from config import (
    CurriculumConfig,
    EMAConfig,
    IRAugConfig,
    LossConfig,
    RGBAugConfig,
    SAGAConfig,
    TeacherUpdateConfig,
    TrainingConfig,
)
from ema import copy_student_to_teacher, ema_update
from losses import compute_ir_loss, compute_mid_loss, compute_rgb_loss
from saga import SemanticAwareGrayAugmentation
from scheduler import CurriculumScheduler, DomainStep, Phase
from trainer import CurriculumDomainAdaptationTrainer
from fcos_wrapper import FCOSDetector, build_fcos_trio
from adaptive_threshold import AdaptiveThresholdScheduler, AdaptiveThresholdConfig, TeacherThresholds
from evaluator import DetectionEvaluator, PhaseEvaluator
from visualize import visualize_eval_samples

__all__ = [
    # Trainer (main entry point)
    "CurriculumDomainAdaptationTrainer",
    # Config
    "TrainingConfig",
    "EMAConfig",
    "SAGAConfig",
    "RGBAugConfig",
    "IRAugConfig",
    "CurriculumConfig",
    "LossConfig",
    "TeacherUpdateConfig",
    # SAGA
    "SemanticAwareGrayAugmentation",
    # Scheduler
    "CurriculumScheduler",
    "Phase",
    "DomainStep",
    # EMA
    "ema_update",
    "copy_student_to_teacher",
    # Losses
    "compute_rgb_loss",
    "compute_mid_loss",
    "compute_ir_loss",
    # Batch types
    "RGBBatch",
    "MidBatch",
    "IRBatch",
    # FCOS
    "FCOSDetector",
    "build_fcos_trio",
    # Adaptive threshold
    "AdaptiveThresholdScheduler",
    "AdaptiveThresholdConfig",
    "TeacherThresholds",
    # Evaluation
    "DetectionEvaluator",
    "PhaseEvaluator",
    # Visualization
    "visualize_eval_samples",
]
