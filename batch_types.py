"""
Typed batch containers for each domain step.

Domain hierarchy:   RGB  →  MID (SAGA)  →  IR
Data availability:  labeled   semi-labeled   unlabeled

Mixed-batch types (Phase 2 / Phase 3) use an in-batch split: the first
`n_xxx` images come from the "earlier" domain, the rest from the "later" one.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class RGBBatch:
    """
    Source domain batch — fully labeled.

    images  : [B, 3, H, W]  float32, values in model's expected range
    targets : list of B dicts, each with
                "boxes"  : [N, 4]  float32  xyxy
                "labels" : [N]     int64
    """
    images: torch.Tensor
    targets: List[Dict[str, torch.Tensor]]


@dataclass
class IRBatch:
    """
    Target domain batch — unlabeled.

    images : [B, C, H, W]  float32  (C=1 for thermal or C=3 for pseudo-RGB)
    """
    images: torch.Tensor


@dataclass
class RgbMidBatch:
    """
    Phase 2 mixed batch — in-batch split [RGB | MID].

    Layout:
      images[0       : n_rgb]   → original RGB images (RGB part)
      images[n_rgb   : B   ]   → SAGA-transformed RGB images (MID part)

    Both halves come from the same RGB loader pull and BOTH have ground truth,
    since SAGA only changes pixels inside boxes, never the boxes themselves.

    images          : [B, 3, H, W]
    targets         : list of B GT dicts (RGB targets for whole batch)
    n_rgb           : index where MID part begins (i.e. len(rgb_part))
    source_images   : optional copy of the pre-SAGA RGB (for debugging/vis)
    """
    images: torch.Tensor
    targets: List[Dict[str, torch.Tensor]]
    n_rgb: int
    source_images: Optional[torch.Tensor] = None


@dataclass
class MidIrBatch:
    """
    Phase 3 mixed batch — MID and IR parts kept SEPARATE pre-aug.

    Their raw H/W may differ (RGB loader vs IR loader can yield different
    sizes), so we defer concatenation to after the geometric aug normalizes
    both parts to (multiscale_target_h, multiscale_target_w). The trainer
    concat's them itself once the shapes match.

    mid_images   : [n_mid, 3, H_rgb, W_rgb]   SAGA-transformed RGB — HAS GT
    ir_images    : [n_ir,  C, H_ir,  W_ir ]   unlabeled IR         — NO GT
    mid_targets  : list of n_mid GT dicts (only for MID slice)
    n_mid, n_ir  : sub-batch counts (n_mid + n_ir = effective B)
    """
    mid_images: torch.Tensor
    ir_images: torch.Tensor
    mid_targets: List[Dict[str, torch.Tensor]]
    n_mid: int
    n_ir: int
