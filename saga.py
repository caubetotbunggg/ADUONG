"""
SAGA — Semantic-Aware Grayscale Augmentation

Purpose: create an intermediate domain that bridges RGB and IR by converting
*only object instances* (inside bounding boxes) to grayscale while keeping the
background in RGB.

Domain role:
  RGB (color + texture)  →  MID/SAGA (gray objects, RGB background)  →  IR (grayscale)

This is NOT a simple augmentation — it is the domain bridge that enables the
curriculum learning strategy: RGB → MID → IR.
"""

import random
from typing import List, Optional

import torch
from typing import Literal

MidLevel = Literal["mid_near_rgb", "mid_intermediate", "mid_near_ir"]


class SemanticAwareGrayAugmentation:
    """
    Convert object bounding-box regions to grayscale; keep background as RGB.

    Args:
        apply_prob: probability of applying the transform to each image.
                    Set to 1.0 to always apply (deterministic).
    """

    # ITU-R BT.601 luma coefficients (standard for digital video)
    _LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])

    def __init__(self, apply_prob: float = 0.5) -> None:
        if not (0.0 <= apply_prob <= 1.0):
            raise ValueError(f"apply_prob must be in [0, 1], got {apply_prob}")
        self.apply_prob = apply_prob

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        image: torch.Tensor,          # [3, H, W]
        boxes: Optional[torch.Tensor],  # [N, 4]  xyxy, can be empty
    ) -> torch.Tensor:
        """
        Apply SAGA to a single image with stochastic gating.

        Returns the (possibly transformed) image — same shape as input.
        """
        if random.random() > self.apply_prob:
            return image
        return self.apply(image, boxes)

    def apply(
        self,
        image: torch.Tensor,           # [3, H, W]
        boxes: Optional[torch.Tensor], # [N, 4]  xyxy
    ) -> torch.Tensor:
        """
        Deterministic SAGA transform (no probability gating).

        - If boxes is None or empty → return image unchanged.
        - Otherwise: object regions → grayscale, background → RGB.
        """
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(
                f"Expected image shape [3, H, W], got {tuple(image.shape)}"
            )

        if boxes is None or boxes.numel() == 0:
            return image.clone()

        _, H, W = image.shape
        device = image.device

        # Build per-pixel object mask from all bounding boxes
        obj_mask = self._build_object_mask_static(boxes, H, W, device)  # [H, W] bool

        if not obj_mask.any():
            return image.clone()

        # Compute grayscale version of the whole image
        gray = self._to_grayscale(image)          # [H, W]
        gray_3ch = gray.unsqueeze(0).expand(3, H, W)  # [3, H, W]

        # Blend: object mask → gray, background → original RGB
        mask_3ch = obj_mask.unsqueeze(0).expand(3, H, W)
        result = torch.where(mask_3ch, gray_3ch, image)

        return result

    def apply_to_batch(
        self,
        images: torch.Tensor,            # [B, 3, H, W]
        batch_boxes: List[Optional[torch.Tensor]],  # list of [N_i, 4] or None
    ) -> torch.Tensor:
        """
        Apply SAGA to a whole batch.

        Each image is processed independently with its own boxes.
        The stochastic gating (apply_prob) is evaluated per image.
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(
                f"Expected images shape [B, 3, H, W], got {tuple(images.shape)}"
            )
        if len(images) != len(batch_boxes):
            raise ValueError(
                f"images batch size {len(images)} != len(batch_boxes) {len(batch_boxes)}"
            )

        results = [
            self(img, boxes)
            for img, boxes in zip(images, batch_boxes)
        ]
        return torch.stack(results, dim=0)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _to_grayscale(self, image: torch.Tensor) -> torch.Tensor:
        """
        Convert [3, H, W] RGB to [H, W] grayscale using ITU-R BT.601 luma.
        The result is replicated to 3 channels by the caller.
        """
        weights = self._LUMA_WEIGHTS.to(device=image.device, dtype=image.dtype)
        return (image * weights.view(3, 1, 1)).sum(dim=0)

    @staticmethod
    def _build_object_mask_static(
        boxes: torch.Tensor,   # [N, 4]  xyxy float
        H: int,
        W: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Create a binary mask [H, W] that is True inside any bounding box.

        Out-of-bound coordinates are clamped to image boundaries.
        Zero-area boxes (x1 >= x2 or y1 >= y2 after clamping) are skipped.
        """
        mask = torch.zeros(H, W, dtype=torch.bool, device=device)

        for box in boxes:
            x1 = max(0, int(box[0].item()))
            y1 = max(0, int(box[1].item()))
            x2 = min(W, int(box[2].item()))
            y2 = min(H, int(box[3].item()))

            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True

        return mask


# ---------------------------------------------------------------------------
# SoftSAGA — soft interpolation between RGB and grayscale in object regions
# ---------------------------------------------------------------------------

class SoftSAGA:
    """
    Soft Semantic-Aware Grayscale Augmentation.

    Unlike hard SAGA (binary object→gray or bg→gray), SoftSAGA blends
    object regions between RGB and grayscale using alpha:

        pixel_out = alpha * rgb_pixel + (1 - alpha) * gray_pixel

    Background stays unchanged (original RGB).

    3 levels for curriculum bridge RGB → MID → IR:
        near_rgb     : alpha ~ 0.7  (objects mostly RGB, slightly desaturated)
        intermediate : alpha ~ 0.5  (half RGB, half gray)
        near_ir      : alpha ~ 0.25 (objects mostly gray, approaching IR)
    """

    _LUMA_WEIGHTS = torch.tensor([0.299, 0.587, 0.114])

    def apply(
        self,
        image: torch.Tensor,            # [3, H, W]
        boxes: Optional[torch.Tensor],  # [N, 4] xyxy
        alpha: float,                   # 1.0=pure RGB, 0.0=pure gray
    ) -> torch.Tensor:
        """Deterministic soft blend on object regions. Background unchanged."""
        if image.dim() != 3 or image.shape[0] != 3:
            raise ValueError(f"Expected [3,H,W], got {tuple(image.shape)}")
        if boxes is None or boxes.numel() == 0:
            return image.clone()

        _, H, W = image.shape
        device = image.device

        # Grayscale version of whole image (luma)
        weights = self._LUMA_WEIGHTS.to(device=device, dtype=image.dtype)
        gray = (image * weights.view(3, 1, 1)).sum(dim=0)   # [H, W]
        gray_3ch = gray.unsqueeze(0).expand(3, H, W)         # [3, H, W]

        # Build object mask
        obj_mask = SemanticAwareGrayAugmentation._build_object_mask_static(
            boxes, H, W, device
        )
        if not obj_mask.any():
            return image.clone()

        # Soft blend in object regions: alpha*rgb + (1-alpha)*gray
        blended = alpha * image + (1.0 - alpha) * gray_3ch  # [3, H, W]
        mask_3ch = obj_mask.unsqueeze(0).expand(3, H, W)
        return torch.where(mask_3ch, blended, image)

    def apply_to_batch(
        self,
        images: torch.Tensor,                        # [B, 3, H, W]
        batch_boxes: List[Optional[torch.Tensor]],
        alpha: float,
    ) -> torch.Tensor:
        return torch.stack([
            self.apply(img, boxes, alpha)
            for img, boxes in zip(images, batch_boxes)
        ])
