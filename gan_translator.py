"""
GANTranslator — wrapper inference-only cho GAN generator RGB → MID.

GAN (CycleGAN ResNet, input_nc=3, output_nc=1) nhận RGB [B,3,H,W]
và sinh ra ảnh grayscale 1-channel [B,1,H,W] rồi expand lên [B,3,H,W]
— giống cách FLIRIRDataset load ảnh IR qua Grayscale(num_output_channels=3).

Pipeline:
  RGB [B,3,H,W] → normalize [-1,1] → GAN → Tanh [-1,1] → rescale [0,1]
  → expand 3ch [B,3,H,W]  (tương đương ảnh IR trong dataset)

Cách dùng:
    translator = GANTranslator.from_checkpoint(
        "gan_mid/latest_net_G_A.pth",
        device=torch.device("cuda"),
    )
    mid_images = translator.apply_to_batch(rgb_images)  # [B, 3, H, W]
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CycleGAN ResNet Generator
# ---------------------------------------------------------------------------

class ResnetBlock(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, bias=True),
            nn.InstanceNorm2d(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.conv_block(x)


class ResnetGenerator(nn.Module):
    """
    CycleGAN ResNet generator.
    Mặc định: input_nc=3 (RGB), output_nc=1 (grayscale MID), ngf=64, n_blocks=9.
    Khớp với checkpoint latest_net_G_A.pth trong gan_mid/.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 1,
        ngf: int = 64,
        n_blocks: int = 9,
    ) -> None:
        super().__init__()

        layers: List[nn.Module] = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf,     ngf * 2, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 4),
            nn.ReLU(inplace=True),
        ]
        for _ in range(n_blocks):
            layers.append(ResnetBlock(ngf * 4))
        layers += [
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(ngf * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2,
                               padding=1, output_padding=1, bias=True),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7),
            nn.Tanh(),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ---------------------------------------------------------------------------
# GANTranslator
# ---------------------------------------------------------------------------

class GANTranslator:
    """
    Wrapper inference-only: RGB [B,3,H,W] → MID [B,3,H,W].

    - Input : float32 [0,1] (chuẩn torchvision ToTensor)
    - Output: float32 [0,1], 3 channel (grayscale repeat × 3, giống IR dataset)
    - GAN không có gradient — hoàn toàn frozen khi training.

    Args:
        generator : ResnetGenerator đã load weights, ở eval mode.
                    Nếu None, apply_to_batch() trả nguyên input (passthrough).
        device    : device để chạy GAN inference.
        amp       : dùng torch.autocast khi inference (tiết kiệm VRAM).
    """

    def __init__(
        self,
        generator: Optional[nn.Module] = None,
        device: torch.device = torch.device("cpu"),
        amp: bool = False,
    ) -> None:
        self.device = device
        self.amp    = amp
        self._gen   = generator

        if self._gen is not None:
            self._gen.to(self.device)
            self._gen.eval()
            for p in self._gen.parameters():
                p.requires_grad = False
            logger.info(
                "GANTranslator ready — %d params, device=%s, amp=%s",
                sum(p.numel() for p in self._gen.parameters()), device, amp,
            )
        else:
            logger.warning(
                "GANTranslator: no generator provided — "
                "apply_to_batch() returns original RGB unchanged."
            )

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        input_nc: int = 3,
        output_nc: int = 1,
        ngf: int = 64,
        n_blocks: int = 9,
        state_dict_key: Optional[str] = None,
        device: torch.device = torch.device("cpu"),
        amp: bool = False,
    ) -> "GANTranslator":
        """
        Load ResnetGenerator từ checkpoint (state_dict).

        Args:
            checkpoint_path : đường dẫn .pt / .pth
            input_nc        : channels input (mặc định 3)
            output_nc       : channels output (mặc định 1)
            ngf, n_blocks   : kiến trúc generator
            state_dict_key  : key trong checkpoint dict để lấy state_dict.
                              None = file là state_dict thẳng.
            device, amp     : xem __init__
        """
        gen = ResnetGenerator(input_nc=input_nc, output_nc=output_nc,
                              ngf=ngf, n_blocks=n_blocks)
        ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        state = ckpt[state_dict_key] if state_dict_key is not None else ckpt
        gen.load_state_dict(state)
        logger.info("GANTranslator: loaded %s", checkpoint_path)
        return cls(generator=gen, device=device, amp=amp)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        [B,3,H,W] float [0,1] → [B,3,H,W] float [0,1] (grayscale × 3).

        Bước:
          1. [0,1] → [-1,1]  (CycleGAN convention)
          2. GAN forward → [B, out_nc, H, W]  Tanh → [-1,1]
          3. [-1,1] → [0,1]
          4. Nếu out_nc==1 → expand 3 channel (contiguous)
        """
        if self._gen is None:
            return images

        x = images.to(self.device) * 2.0 - 1.0  # [0,1] → [-1,1]

        if self.amp:
            with torch.autocast(device_type=self.device.type):
                out = self._gen(x)
        else:
            out = self._gen(x)

        out = (out + 1.0) / 2.0          # [-1,1] → [0,1]
        out = out.clamp(0.0, 1.0)

        if out.shape[1] == 1:
            out = out.expand(-1, 3, -1, -1).contiguous()

        return out.to(dtype=images.dtype, device=images.device)

    # ------------------------------------------------------------------
    # Public interface (tương thích với SAGA.apply_to_batch)
    # ------------------------------------------------------------------

    def apply_to_batch(
        self,
        images: torch.Tensor,
        batch_boxes: Optional[List] = None,  # không dùng — GAN dịch toàn ảnh
    ) -> torch.Tensor:
        """
        RGB → MID qua GAN cho cả batch.

        Args:
            images      : [B, 3, H, W] float32, [0, 1]
            batch_boxes : không dùng (tương thích interface SAGA)

        Returns:
            [B, 3, H, W] float32, [0, 1]  — grayscale 3-channel (như IR dataset)
        """
        if images.dim() != 4 or images.shape[1] != 3:
            raise ValueError(f"Expected [B, 3, H, W], got {tuple(images.shape)}")
        return self._forward(images)

    def __call__(
        self,
        images: torch.Tensor,
        batch_boxes: Optional[List] = None,
    ) -> torch.Tensor:
        return self.apply_to_batch(images, batch_boxes)
