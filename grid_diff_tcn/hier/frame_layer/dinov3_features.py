# -*- coding: utf-8 -*-
"""
DINOv3 feature extractor for grid_diff_tcn.
Wraps the pretrained DINOv3 ViT backbone for frame-level feature extraction.
Each ROI image (H x W x 3) is fed through DINOv3 and the CLS token is returned
as the per-frame feature (768-dim for ViT-B).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DinoV3FeatureExtractor", "DINOV3_MODELS", "DINOV3_DEFAULT_MODEL", "DINOV3_FEAT_DIMS"]

# Model name -> (backbone function name in dinov3.hub.backbones, embedding dim)
DINOV3_MODELS = {
    "vit_small":      ("dinov3_vits16",      384),
    "vit_small_plus":  ("dinov3_vits16plus",  384),
    "vit_base":        ("dinov3_vitb16",       768),
    "vit_large":       ("dinov3_vitl16",       1024),
    "vit_large_plus":  ("dinov3_vitl16plus",  1024),
    "vit_huge_plus":   ("dinov3_vith16plus",  1280),
}

DINOV3_DEFAULT_MODEL = "vit_base"
DINOV3_FEAT_DIMS = {name: info[1] for name, info in DINOV3_MODELS.items()}

# ImageNet normalization constants
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD  = (0.229, 0.224, 0.225)
_PATCH_SIZE = 16


class DinoV3FeatureExtractor(nn.Module):
    """
    Wraps a pretrained DINOv3 ViT for efficient frame-level feature extraction.

    Args:
        model_name: one of DINOV3_MODELS keys, e.g. "vit_base"
        pretrained: load pretrained weights (default True)
        pool_strategy: how to pool patch tokens
            - "cls": return CLS token only (recommended, dim = embed_dim)
            - "mean": return mean of all patch tokens
            - "both": return (cls_token, patch_tokens_mean)
        image_size: resize input images to this size before feeding to ViT (default 224)
        device: "cuda" or "cpu"
    """

    def __init__(
        self,
        model_name: str = DINOV3_DEFAULT_MODEL,
        pretrained: bool = True,
        pool_strategy: Literal["cls", "mean", "both"] = "cls",
        image_size: int = 224,
        device: str = "cuda",
        weights: str | None = None,
    ):
        super().__init__()
        if model_name not in DINOV3_MODELS:
            raise ValueError(
                f"Unknown model_name={model_name!r}. "
                f"Available: {list(DINOV3_MODELS.keys())}"
            )
        backbone_fn_name, self.embed_dim = DINOV3_MODELS[model_name]
        self.pool_strategy = pool_strategy
        self.image_size = int(image_size)

        # Register normalization buffers (on CPU, moved to device in forward)
        self.register_buffer(
            "_mean",
            torch.tensor(_IMAGENET_MEAN).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "_std",
            torch.tensor(_IMAGENET_STD).view(1, 3, 1, 1),
        )
        self._device = torch.device(device)

        # --- resolve weights (enum vs local file path) --------------------------------
        import dinov3.hub.backbones as _backbones
        from dinov3.hub.backbones import Weights as _Weights
        from glob import glob as _glob

        backbone_fn = getattr(_backbones, backbone_fn_name)

        # Resolve local file path if available, otherwise use enum
        _local_path: Path | None = None
        if weights is None:
            _pkg_root = Path(__file__).resolve().parents[3]  # .../dinov3-main/
            _pattern = str(_pkg_root / "dinov3" / "checkpointer" / "*.pth")
            for _p in sorted(_glob(_pattern)):
                _bn = os.path.basename(_p)
                # match e.g. "dinov3_vits16_pretrain_lvd1689m-08c60483.pth"
                _p_path = Path(_p)
                if _p_path.stat().st_size == 0:
                    continue  # skip empty/corrupted file
                if _bn.startswith(backbone_fn_name + "_") or _bn.startswith(backbone_fn_name + "="):
                    _local_path = _p_path
                    break
            if _local_path is None:
                weights = _Weights.LVD1689M  # fallback: let torch.hub download

        # Load from local file directly via torch.load (bypasses torch.hub URL logic)
        if _local_path is not None and _local_path.is_file() and _local_path.stat().st_size > 0:
            _size = _local_path.stat().st_size
            if _size == 0:
                raise RuntimeError(
                    f"Weight file is empty (0 bytes): {_local_path}. "
                    "The file was not downloaded correctly. "
                    "Please delete it and re-download."
                )
            print(f"[DinoV3FeatureExtractor] Loading weights from local file: {_local_path.name} ({_size/1024**2:.1f} MB)")
            _backbone_no_load = backbone_fn(pretrained=False)  # build structure only
            _state = torch.load(_local_path, map_location="cpu", weights_only=False)
            _backbone_no_load.load_state_dict(_state, strict=True)
            self.backbone = _backbone_no_load
            self.backbone.eval()
        else:
            # torch-hub path (enum or network URL)
            kwargs_for_backbone: dict = {}
            if isinstance(weights, str) and weights.startswith("file://"):
                kwargs_for_backbone["check_hash"] = False
            self.backbone = backbone_fn(pretrained=pretrained, weights=weights, **kwargs_for_backbone)
            self.backbone.eval()

    @property
    def device(self) -> torch.device:
        return self._device

    @property
    def feat_dim(self) -> int:
        if self.pool_strategy == "both":
            return self.embed_dim * 2
        return self.embed_dim

    def _preprocess(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: (B, 3, H, W) float tensor in [0, 1]
        returns: (B, 3, image_size, image_size) normalized tensor
        """
        mean = self._mean.to(images.device, images.dtype)
        std  = self._std.to(images.device, images.dtype)
        x = (images - mean) / std

        # Resize if needed (make sure H, W are divisible by patch_size)
        if x.shape[-1] != self.image_size or x.shape[-2] != self.image_size:
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bicubic",
                align_corners=False,
                antialias=True,
            )
        return x

    def forward(
        self,
        images: torch.Tensor,
        return_patch_tokens: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Extract DINOv3 features from a batch of ROI images.

        Args:
            images: (B, 3, H, W) float tensor in [0, 1]
            return_patch_tokens: if True and pool_strategy="cls", also return
                the raw patch token tensor (B, num_patches, embed_dim)

        Returns:
            - pool_strategy="cls":       (B, embed_dim) CLS token
            - pool_strategy="mean":      (B, embed_dim) mean of patch tokens
            - pool_strategy="both":      tuple of (cls, mean_patch)
            If return_patch_tokens=True with pool_strategy="cls":
                also returns patch_tokens (B, num_patches, embed_dim)
        """
        x = self._preprocess(images)

        with torch.inference_mode():
            feat_dict = self.backbone.forward_features(x)

        cls_token: torch.Tensor = feat_dict["x_norm_clstoken"]        # (B, D)
        patch_tokens: torch.Tensor = feat_dict["x_norm_patchtokens"]  # (B, N, D)

        if self.pool_strategy == "cls":
            if return_patch_tokens:
                return cls_token, patch_tokens
            return cls_token

        elif self.pool_strategy == "mean":
            pooled = patch_tokens.mean(dim=1)  # (B, D)
            return pooled

        elif self.pool_strategy == "both":
            pooled = patch_tokens.mean(dim=1)  # (B, D)
            return cls_token, pooled

        raise AssertionError(f"Unknown pool_strategy={self.pool_strategy!r}")

    def extract_single(self, roi: torch.Tensor) -> torch.Tensor:
        """
        Convenience method for extracting a single ROI feature.

        Args:
            roi: (3, H, W) or (1, 3, H, W) float tensor in [0, 1]
        Returns:
            (embed_dim,) 1-D tensor
        """
        if roi.ndim == 3:
            roi = roi.unsqueeze(0)  # (1, 3, H, W)
        out = self.forward(roi)
        if isinstance(out, tuple):
            return torch.cat(out, dim=-1).squeeze(0)  # (embed_dim*2,)
        return out.squeeze(0)

    @torch.no_grad()
    def extract_batch(self, roi_batch) -> torch.Tensor:
        """
        Extract features from a list or batch of ROI images.

        Args:
            roi_batch: list of (H, W, 3) numpy arrays or a single batched tensor
        Returns:
            (N, embed_dim) tensor
        """
        if isinstance(roi_batch, list):
            # list of numpy arrays (H, W, 3) in [0, 1]
            imgs = torch.stack(
                [torch.from_numpy(r).permute(2, 0, 1).float() for r in roi_batch]
            )
        else:
            imgs = torch.atleast_4d(roi_batch).float()

        imgs = imgs.to(self._device)
        feat = self.forward(imgs)
        if isinstance(feat, tuple):
            feat = torch.cat(feat, dim=-1)
        return feat.cpu()
