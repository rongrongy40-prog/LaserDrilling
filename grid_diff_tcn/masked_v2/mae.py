# -*- coding: utf-8 -*-
"""
Standard MAE for self-supervised pre-training with frozen pretrained encoder.

Architecture:
  1. Full image → frozen encoder → all patch tokens (RoPE handles positions)
  2. Replace masked patches with encoder's native [MASK] token
  3. MAE decoder: masked tokens → per-patch pixel reconstruction
  4. L1 loss between predicted and original masked pixels

Key design:
  - Encoder is FROZEN: we use its pretrained patch tokens directly
  - DINOv3's native [MASK] token is used for masked positions
  - MAE decoder learns to reconstruct masked pixels from visible patch features
  - This gives encoder richer domain features while preserving pretrained quality
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MAEDecoder(nn.Module):
    """
    Lightweight Transformer decoder for MAE pixel reconstruction.

    Input: (encoder_visible_feat OR mask_token) + position embeddings
    Output: per-patch pixel values
    """

    def __init__(
        self,
        encoder_dim: int = 384,
        decoder_dim: int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
        num_patches: int = 196,
    ) -> None:
        super().__init__()
        self.num_patches = num_patches

        # Project encoder dim → decoder dim
        self.enc_to_dec = nn.Linear(encoder_dim, decoder_dim)

        # Learnable mask tokens (shared across all masked positions)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)

        # Decoder position embeddings (learnable)
        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_dim))
        nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        # Transformer decoder layers
        dec_layer = nn.TransformerEncoderLayer(
            d_model=decoder_dim,
            nhead=decoder_heads,
            dim_feedforward=decoder_dim * 4,
            dropout=0.0,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(dec_layer, num_layers=decoder_depth)

        # Prediction head: decoder dim → pixel values per patch
        self.pixel_head = nn.Linear(decoder_dim, 16 * 16 * 3)

    def forward(
        self,
        enc_tokens: torch.Tensor,  # (B, N, D_enc) with masked positions already replaced
    ) -> torch.Tensor:              # (B, N, ps*ps*3) predictions
        """
        Args:
            enc_tokens: (B, N, D_enc) encoder features with mask tokens already filled

        Returns:
            patch_preds: (B, N, ps*ps*3)
        """
        D_dec = self.enc_to_dec.out_features

        # Project to decoder dim
        dec_tokens = self.enc_to_dec(enc_tokens)  # (B, N, D_dec)

        # Add decoder position embeddings
        dec_tokens = dec_tokens + self.decoder_pos_embed  # (B, N, D_dec)

        # Transformer decoder
        dec_out = self.transformer(dec_tokens)  # (B, N, D_dec)

        # Predict pixel values per patch
        patch_preds = self.pixel_head(dec_out)  # (B, N, 768)

        return patch_preds


class StandardMAEPreTrainer(nn.Module):
    """
    Standard MAE pre-training using a frozen pretrained encoder.

    The encoder is NOT fine-tuned. The MAE decoder learns to reconstruct
    masked patch pixels from the encoder's patch token features.
    """

    def __init__(
        self,
        encoder: nn.Module,
        encoder_dim: int = 384,
        patch_size: int = 16,
        image_size: int = 224,
        mask_ratio: float = 0.75,
        decoder_dim: int = 256,
        decoder_depth: int = 4,
        decoder_heads: int = 8,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.mask_ratio = mask_ratio

        self.num_patches_per_side = image_size // patch_size
        self.num_patches = self.num_patches_per_side ** 2  # 196 for 224/16

        # Freeze encoder entirely
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder.eval()

        # MAE decoder
        self.decoder = MAEDecoder(
            encoder_dim=encoder_dim,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            decoder_heads=decoder_heads,
            num_patches=self.num_patches,
        )

    def _get_mask_and_visible(self, B: int, N: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate per-image random visible/masked patch indices.
        Returns:
            visible: (B, Nv) per-image visible patch indices (0..N-1)
            mask:   (B, N) bool, True = masked
        """
        Nv = int(N * (1 - self.mask_ratio))

        visible_list = []
        for _ in range(B):
            perm = torch.randperm(N, device=device)
            visible_list.append(perm[:Nv])

        visible = torch.stack(visible_list, dim=0)  # (B, Nv)

        # mask[b, i] = True means patch i in image b is masked
        mask = torch.ones(B, N, dtype=torch.bool, device=device)
        for b in range(B):
            mask[b, visible[b].long()] = False

        return visible, mask

    @torch.no_grad()
    def _encode_with_mask(
        self,
        images: torch.Tensor,
        visible: torch.Tensor,  # (B, Nv)
        mask_bool: torch.Tensor,  # (B, N) bool, True = masked
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward full image through frozen encoder, then replace masked patches
        with the encoder's native [MASK] token.

        Returns:
            all_tokens: (B, N, D_enc) encoder features with masked positions = [MASK] token
            visible:    (B, Nv) visible indices
        """
        B, C, H, W = images.shape
        N = self.num_patches
        expected_H = expected_W = self.image_size

        # Always resize to encoder's expected size (bilinear, no antialias)
        if H != expected_H or W != expected_W:
            images = F.interpolate(images, size=(expected_H, expected_W),
                                  mode="bilinear", align_corners=False)

        # Full forward through frozen encoder
        feat_dict = self.encoder.forward_features(images)
        patch_tokens = feat_dict["x_norm_patchtokens"]  # (B, N, D_enc)
        encoder_mask_token = self.encoder.mask_token    # (1, D_enc) or (D_enc,)

        # Ensure encoder_mask_token is (1, D)
        if encoder_mask_token.ndim == 1:
            encoder_mask_token = encoder_mask_token.unsqueeze(0)  # (1, D)

        # Replace masked positions with encoder's [MASK] token
        all_tokens = patch_tokens.clone()
        for b in range(B):
            masked_positions = torch.where(mask_bool[b])[0]  # (Nm,)
            all_tokens[b, masked_positions.long()] = encoder_mask_token

        return all_tokens, visible, images

    def forward(self, images: torch.Tensor, return_loss: bool = True) -> dict:
        """
        Args:
            images: (B, 3, H, W) float32 in [0, 1]
            NOTE: B must be divisible by num_patches for MAE to work.
                  Incomplete batches should be skipped by the caller.

        Returns:
            dict with:
              - loss: scalar MAE loss
              - pred: (B, 3, H, W) reconstructed image
              - target: (B, 3, H, W) original image
              - mask_img: (B, H, W) float, 1=masked, 0=visible
        """
        B, C, H, W = images.shape
        N = self.num_patches
        ps = self.patch_size

        # Validate: B must be a multiple of N for unflatten to work
        assert B % N == 0, \
            f"B={B} not divisible by num_patches={N}. Skip incomplete batches in the caller."

        # Random visible/masked indices
        visible, mask_bool = self._get_mask_and_visible(B, N, images.device)
        # Encode full image + replace masked patches with [MASK] token
        all_tokens, visible, images = self._encode_with_mask(images, visible, mask_bool)
        # all_tokens: (B, N, D), images: (B, 3, H, W) resized to self.image_size

        # Re-read dimensions AFTER resize (H/W may have changed)
        _, _, H, W = images.shape
        h = w = H // ps  # actual patch grid from actual image size
        # Decode: all tokens (visible feats + [MASK]) → pixel predictions
        patch_preds = self.decoder(all_tokens)  # (B, N, ps*ps*3)

        # Reshape to image: (B, N, ps*ps*3) → (B, 3, H, W)
        pred_patches = patch_preds.reshape(B, h, w, ps * ps * 3)  # (B, h, w, ps^2*3)
        pred_patches = pred_patches.permute(0, 3, 1, 2)           # (B, ps^2*3, h, w)
        pred = F.pixel_shuffle(pred_patches, upscale_factor=ps)  # (B, 3, H, W)

        result: dict = {"pred": pred, "target": images}

        if return_loss:
            # mask_img: (B, H, W), 1=masked, 0=visible
            mask_img = mask_bool.unflatten(1, (h, w))      # (B, h, w)
            mask_img = mask_img.repeat_interleave(ps, dim=1)  # (B, H, w)
            mask_img = mask_img.repeat_interleave(ps, dim=2)  # (B, H, W)
            mask_img = mask_img.unsqueeze(1)                  # (B, 1, H, W) for broadcast

            valid_count = mask_img.sum().clamp(min=1)
            loss = (pred - images).abs() * mask_img
            loss = loss.sum() / valid_count

            result["loss"] = loss
            result["mask_img"] = mask_img

        return result


def create_mae_pretrainer(
    encoder: nn.Module,
    encoder_dim: int = 384,
    patch_size: int = 16,
    image_size: int = 224,
    mask_ratio: float = 0.75,
    decoder_dim: int = 256,
    decoder_depth: int = 4,
    decoder_heads: int = 8,
) -> StandardMAEPreTrainer:
    """Factory function to create an MAE pre-trainer."""
    return StandardMAEPreTrainer(
        encoder=encoder,
        encoder_dim=encoder_dim,
        patch_size=patch_size,
        image_size=image_size,
        mask_ratio=mask_ratio,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_heads=decoder_heads,
    )
