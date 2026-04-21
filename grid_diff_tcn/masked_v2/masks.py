# -*- coding: utf-8 -*-
"""
Random-center masking for masked image modeling.
The mask patch is placed at a random position within the image center area,
covering ~20% of the image by default.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np


class CenterMask:
    """
    Random-center masking: places a square patch at a random position near the
    image center, covering a configurable fraction of the image area.

    Each call (each forward pass) draws a new random position, so the model
    sees different mask locations over time.

    Args:
        mask_ratio: fraction of image area to mask (default 0.2 = 20%)
        mask_shape: "square" (only square is supported for random positioning)
        image_size: size of input images (assuming square)
    """

    def __init__(
        self,
        mask_ratio: float = 0.2,
        mask_shape: str = "square",
        image_size: int = 224,
    ) -> None:
        self.mask_ratio = float(mask_ratio)
        self.mask_shape = str(mask_shape)
        self.image_size = int(image_size)

    def __call__(
        self,
        images: torch.Tensor,
        return_mask: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Args:
            images: (B, C, H, W) or (B*N, C, H, W)
            return_mask: if True, also return boolean mask

        Returns:
            masked_images: same shape as input
            mask: (B, H, W) or (B*N, H, W), True = masked region
        """
        if images.ndim == 4:
            return self._mask_4d(images, return_mask)
        else:
            raise ValueError(f"Expected 4D tensor, got {images.ndim}D")

    def _create_mask(self, shape: torch.Size, device: torch.device) -> torch.Tensor:
        """Create randomly-positioned center mask for a single image."""
        h, w = shape[-2], shape[-1]
        rng = np.random.default_rng()

        # Square side so that side^2 / (h*w) = mask_ratio
        side = int(round((self.mask_ratio ** 0.5) * h))
        half_side = side // 2

        # Random center within [half_margin, h-half_margin]
        margin = half_side
        center_y = rng.integers(margin, h - margin + 1)
        center_x = rng.integers(margin, w - margin + 1)

        y_coords = torch.arange(h, device=device)
        x_coords = torch.arange(w, device=device)
        yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

        mask = torch.ones(h, w, dtype=torch.bool, device=device)
        y0, y1 = center_y - half_side, center_y + half_side
        x0, x1 = center_x - half_side, center_x + half_side
        mask[y0:y1, x0:x1] = False

        return mask

    def _mask_4d(
        self,
        images: torch.Tensor,
        return_mask: bool,
    ) -> tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """Handle (B, C, H, W) input."""
        B, C, H, W = images.shape
        device = images.device

        masks = []
        for _ in range(B):
            masks.append(self._create_mask(torch.Size([H, W]), device))
        mask_2d = torch.stack(masks, dim=0)  # (B, H, W), True = visible

        masked = images.clone()
        masked[mask_2d.unsqueeze(1).expand(-1, C, H, W)] = 0.0

        # Invert: True = masked (region where loss is computed)
        mask_2d = ~mask_2d

        if return_mask:
            return masked, mask_2d
        else:
            return masked


class MaskedImageModelingLoss(nn.Module):
    """
    Loss for masked image modeling - reconstruct masked pixels.
    """
    
    def __init__(
        self,
        loss_type: str = "l1",
        patch_size: int = 16,
    ) -> None:
        super().__init__()
        self.loss_type = str(loss_type)
        self.patch_size = int(patch_size)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            pred: (B, C, H, W) or (B, T, F, C, H, W) predicted pixels
            target: (B, C, H, W) or (B, T, F, C, H, W) ground truth pixels
            mask: (B, H, W) or (B, T, F, H, W), True = masked region to compute loss
        
        Returns:
            loss: scalar
        """
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")
        
        if mask.ndim == 3:
            mask_expanded = mask.unsqueeze(1)
        elif mask.ndim == 5:
            mask_expanded = mask.unsqueeze(3)
        else:
            raise ValueError(f"Unexpected mask ndim {mask.ndim}")
        
        valid_count = mask.sum()
        if valid_count == 0:
            return pred.sum() * 0
        
        if self.loss_type == "l1":
            loss = (pred - target).abs()
        elif self.loss_type == "l2":
            loss = (pred - target) ** 2
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")
        
        loss = (loss * mask_expanded.float()).sum() / valid_count.clamp(min=1)
        return loss