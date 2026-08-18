"""
AdaptiveThresholdScheduler — curriculum-based pseudo-label confidence threshold.

3-phase strategy:
  Phase 1 (RGB warmup)        → high threshold (teachers not trained yet)
  Phase 2 (mixed RGB + MID)   → medium (rgb_teacher specializes)
  Phase 3 (mixed MID + IR)    → medium (ir_teacher specializes)

Each teacher can have an independent threshold per phase.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Union

from scheduler import Phase

ThreshType = Union[float, Dict[int, float]]


# ---------------------------------------------------------------------------
# Ramp config
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Per-teacher threshold config
# ---------------------------------------------------------------------------

@dataclass
class TeacherThresholds:
    """
    Confidence thresholds per phase.

    Each value can be:
      float            — global threshold for all classes
      Dict[int, float] — per-class threshold; missing classes fall back to 0.7
    """
    phase1: ThreshType = 0.90
    phase2: ThreshType = 0.70
    phase3: ThreshType = 0.60

    def get(self, phase: Phase) -> ThreshType:
        mapping = {
            Phase.PHASE1_RGB_WARMUP: self.phase1,
            Phase.PHASE2_RGB_MID:    self.phase2,
            Phase.PHASE3_MID_IR:     self.phase3,
        }
        return mapping.get(phase, 0.7)


@dataclass
class AdaptiveThresholdConfig:
    """
    Configuration for adaptive confidence thresholds.
    """
    rgb_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.85, phase2=0.65, phase3=0.55,
    ))
    ir_teacher: TeacherThresholds = field(default_factory=lambda: TeacherThresholds(
        phase1=0.95, phase2=0.75, phase3=0.65,
    ))


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class AdaptiveThresholdScheduler:
    """
    Returns per-teacher confidence thresholds based on phase.

    Usage:
        phase = curriculum_scheduler.get_phase(global_step)
        rgb_thresh = thresh_sched.rgb_teacher(phase)
        ir_thresh  = thresh_sched.ir_teacher(phase)
    """

    def __init__(self, config: Optional[AdaptiveThresholdConfig] = None) -> None:
        self.config = config or AdaptiveThresholdConfig()

    def rgb_teacher(self, phase: Phase) -> ThreshType:
        """Threshold for rgb_teacher."""
        return self.config.rgb_teacher.get(phase)

    def ir_teacher(self, phase: Phase) -> ThreshType:
        """Threshold for ir_teacher."""
        return self.config.ir_teacher.get(phase)

    def both(self, phase: Phase) -> ThreshType:
        """Shared threshold when both teachers are used (mixed-batch steps)."""
        return self.config.rgb_teacher.get(phase)

    def summary(self) -> str:
        def _fmt(v: ThreshType) -> str:
            if isinstance(v, dict):
                return "{" + ", ".join(f"{k}:{t:.2f}" for k, t in sorted(v.items())) + "}"
            return f"{v:.2f}"

        lines = ["AdaptiveThresholdScheduler:"]
        for p in Phase:
            rt = self.rgb_teacher(p)
            it = self.ir_teacher(p)
            lines.append(f"  {p.name:<25s}  rgb={_fmt(rt)}  ir={_fmt(it)}")

        return "\n".join(lines)
