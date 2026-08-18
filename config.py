"""
Configuration dataclasses for Curriculum Domain Adaptation framework.

3-phase training flow with **mixed-batch** Phase 2 and Phase 3:

    Phase 1  → RGB only            (supervised warmup)
    Phase 2  → mixed batch [RGB | MID]   (rgb_part keeps RGB, mid_part = GAN)
    Phase 3  → mixed batch [MID | IR]    (mid_part = GAN, ir_part = unlabeled IR)
"""

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Sub-configs
# ---------------------------------------------------------------------------

@dataclass
class EMAConfig:
    """Teacher update settings: classic EMA or adaptive EMA (AEMA).

    AEMA (DDT-style) splits teacher params into a fast group (top fraction by
    accumulated target-domain gradient magnitude) and a slow group (the rest),
    updating each group with a different EMA decay.  ``mode="ema"`` keeps the
    original classic EMA behaviour (used as the ablation baseline).
    """
    alpha: float = 0.999          # classic EMA decay factor (higher = slower)
    use_warmup: bool = True       # ramp classic EMA alpha up during early training

    # --- AEMA (adaptive EMA) ---
    mode: str = "ema"             # "ema" (classic) or "aema" (adaptive)
    aema_fast_alpha: float = 0.997   # alpha for high-gradient (top) params
    aema_slow_alpha: float = 0.9996  # alpha for the remaining params
    aema_top_ratio: float = 0.10     # fraction of params using fast alpha
    aema_update_interval: int = 2    # gradient accumulation steps per update
    merge_iou_threshold: float = 0.50        # DDT teacher-student pseudo merge IoU
    merge_student_conf_thresh: float = 0.70  # min student conf in merge


@dataclass
class GANConfig:
    """GAN translator settings — RGB → MID via pretrained GAN generator."""
    checkpoint_path: str   = ""       # path tới file .pth của GAN generator
    input_nc:        int   = 3        # channels input GAN (RGB)
    output_nc:       int   = 1        # channels output GAN (grayscale)
    ngf:             int   = 64       # base filter count
    n_blocks:        int   = 9        # số ResNet blocks
    state_dict_key:  str   = ""       # key trong checkpoint dict ("" = state_dict thẳng)
    amp:             bool  = False    # dùng autocast khi GAN inference


@dataclass
class RGBAugConfig:
    """Augmentation for RGB and MID (SAGA) images."""
    # --- Geometric (weak aug — applied to BOTH teacher and student) ---
    hflip_prob:                  float = 0.5

    # Multi-scale: resize to [scale_min, scale_max] × original, then crop/pad to fixed size
    multiscale_min:              float = 0.5
    multiscale_max:              float = 1.5
    multiscale_target_h:         int   = 512
    multiscale_target_w:         int   = 640

    # --- Photometric (strong aug — applied to student only) ---
    blur_prob:                   float = 0.5
    blur_sigma_max:              float = 1.0

    color_jitter_prob:           float = 0.5
    cj_brightness:               float = 0.2
    cj_contrast:                 float = 0.2
    cj_saturation:               float = 0.3
    cj_hue:                      float = 0.05

    random_erasing_prob:         float = 0.3
    random_erasing_scale_min:    float = 0.02
    random_erasing_scale_max:    float = 0.10
    random_erasing_ratio_min:    float = 0.3
    random_erasing_ratio_max:    float = 3.3


@dataclass
class IRAugConfig:
    """Augmentation for IR (thermal) images."""
    # --- Geometric (weak aug — applied to BOTH teacher and student) ---
    hflip_prob:                  float = 0.5

    # Multi-scale
    multiscale_min:              float = 0.5
    multiscale_max:              float = 1.5
    multiscale_target_h:         int   = 512
    multiscale_target_w:         int   = 640

    # --- Photometric (strong aug — applied to student only) ---
    intensity_shift_prob:        float = 0.5
    intensity_shift_mag:         float = 0.1

    contrast_jitter_prob:        float = 0.5
    contrast_jitter_mag:         float = 0.2

    gamma_prob:                  float = 0.3
    gamma_min:                   float = 0.7
    gamma_max:                   float = 1.3

    gaussian_noise_prob:         float = 0.3
    gaussian_noise_std:          float = 0.02


@dataclass
class CurriculumConfig:
    """
    Phase boundaries (in global iterations) for the 3-phase curriculum.

    Phase 1: [0,          phase1_end)  → RGB only           (supervised warmup)
    Phase 2: [phase1_end, phase2_end)  → mixed [RGB | MID]  (in-batch split)
    Phase 3: [phase2_end, phase3_end)  → mixed [MID | IR]   (in-batch split)

    In-batch split ratios:
      phase2_rgb_ratio : fraction of each Phase-2 batch that's RGB (rest is MID).
      phase3_mid_ratio : fraction of each Phase-3 batch that's MID (rest is IR).

    Example (batch_size=8, phase2_rgb_ratio=0.5):
      batch layout = [RGB, RGB, RGB, RGB | MID, MID, MID, MID]
    """
    phase1_end: int = 3_000
    phase2_end: int = 10_000
    phase3_end: int = 17_000

    phase2_rgb_ratio: float = 0.5
    phase3_mid_ratio: float = 0.5


@dataclass
class LossConfig:
    """Loss weights per phase."""
    # Phase 1 — RGB warmup (no teacher pseudo in pure warmup)
    p1_gt_weight:           float = 1.0
    p1_pseudo_weight:       float = 0.0   # optional rgb_teacher pseudo on RGB (kept 0 in warmup)

    # Phase 2 — mixed [RGB | MID]; GT applies to whole batch (both halves have GT)
    p2_gt_weight:           float = 1.0
    p2_rgb_teacher_weight:  float = 0.5
    p2_ir_teacher_weight:   float = 0.5

    # Phase 3 — mixed [MID | IR]; GT applies to MID slice ONLY (IR has no GT)
    p3_gt_weight:           float = 1.0
    p3_rgb_teacher_weight:  float = 0.5
    p3_ir_teacher_weight:   float = 0.5


@dataclass
class TeacherUpdateConfig:
    """
    Which teachers to EMA-update in each phase step.

    Selected policy: only the "specialist" teacher updates per phase.
      Phase 1 (rgb step)      : no EMA update (warmup)
      Phase 2 (rgb_mid step)  : rgb_teacher only
      Phase 3 (mid_ir step)   : ir_teacher  only
    """
    p2_update_rgb_teacher: bool = True
    p2_update_ir_teacher:  bool = False

    p3_update_rgb_teacher: bool = False
    p3_update_ir_teacher:  bool = True


# ---------------------------------------------------------------------------
# AAT — Adversarial Attacked Teacher (statistical domain match)
# ---------------------------------------------------------------------------

@dataclass
class AATConfig:
    """Statistical domain-match adversarial attacked teacher.

    Perturbs teacher inputs toward the IR domain's per-channel mean/std to
    generate extra pseudo-labels the clean teacher may miss.  Clean + attacked
    predictions are merged via IoU dedup.  No discriminator needed (works with
    the original 3-phase GAN-bridge pipeline, no DANN changes).
    """
    enabled: bool = False
    mode: str = "statistical"            # "statistical" (no disc) or "discriminator"
    epsilon: float = 0.02               # perturbation budget
    merge_iou: float = 0.5               # IoU threshold for clean+attacked merge
    ir_stats_momentum: float = 0.9       # EMA momentum for IR reference stats


# ---------------------------------------------------------------------------
# Top-level config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfig:
    """Master config for CurriculumDomainAdaptationTrainer."""
    ema: EMAConfig = field(default_factory=EMAConfig)
    gan: GANConfig = field(default_factory=GANConfig)
    rgb_aug: RGBAugConfig = field(default_factory=RGBAugConfig)
    ir_aug: IRAugConfig = field(default_factory=IRAugConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    teacher_update: TeacherUpdateConfig = field(default_factory=TeacherUpdateConfig)
    aat: AATConfig = field(default_factory=AATConfig)

    pseudo_label_conf_thresh: float = 0.7   # min score to keep a pseudo-label box
    grad_clip: float = 10.0                 # max gradient norm (0 = disabled)
    device: str = "cuda"
    log_interval: int = 50                  # log every N iterations
