"""
CurriculumScheduler — decides which domain step to execute at each iteration.

3-phase curriculum (mixed batches in Phase 2 and Phase 3):

  Phase 1  [0,           phase1_end)  → "rgb"      (RGB warmup, supervised)
  Phase 2  [phase1_end,  phase2_end)  → "rgb_mid"  (mixed batch [RGB | MID])
  Phase 3  [phase2_end,  phase3_end)  → "mid_ir"   (mixed batch [MID | IR])

There is exactly ONE step type per phase — no within-phase alternation
because the RGB/MID and MID/IR mixing now happens **inside each batch**
via in-batch split (see CurriculumConfig.phase2_rgb_ratio / phase3_mid_ratio).
"""

from enum import Enum
from typing import Literal

from config import CurriculumConfig


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

class Phase(Enum):
    PHASE1_RGB_WARMUP = 1
    PHASE2_RGB_MID    = 2   # mixed [RGB | MID]
    PHASE3_MID_IR     = 3   # mixed [MID | IR]


DomainStep = Literal["rgb", "rgb_mid", "mid_ir"]


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class CurriculumScheduler:
    """
    Curriculum scheduler — one step type per phase.

    Call `get_next_step(global_step)` exactly once per iteration in order.
    """

    def __init__(self, config: CurriculumConfig) -> None:
        self.config = config

    def get_phase(self, global_step: int) -> Phase:
        """Return the curriculum phase for a given global_step."""
        c = self.config
        if global_step < c.phase1_end:
            return Phase.PHASE1_RGB_WARMUP
        if global_step < c.phase2_end:
            return Phase.PHASE2_RGB_MID
        return Phase.PHASE3_MID_IR

    def get_next_step(self, global_step: int) -> DomainStep:
        """Determine the next domain step to execute."""
        phase = self.get_phase(global_step)
        if phase == Phase.PHASE1_RGB_WARMUP:
            return "rgb"
        if phase == Phase.PHASE2_RGB_MID:
            return "rgb_mid"
        return "mid_ir"

    def __repr__(self) -> str:
        c = self.config
        return (
            f"CurriculumScheduler("
            f"phase1_end={c.phase1_end}, "
            f"phase2_end={c.phase2_end}, "
            f"phase3_end={c.phase3_end}, "
            f"phase2_rgb_ratio={c.phase2_rgb_ratio}, "
            f"phase3_mid_ratio={c.phase3_mid_ratio})"
        )
