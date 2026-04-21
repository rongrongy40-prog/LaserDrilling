# -*- coding: utf-8 -*-
"""
Masked Image Modeling for drilling hole detection (v2 - trainable encoder).

Two-stage training:
  1. Self-supervised: train encoder + decoder to reconstruct masked pixels (encoder UNFROZEN)
  2. Supervised: fine-tune with classification head (encoder frozen or fine-tuned)

Key difference from masked/ (v1):
  - Encoder is TRAINABLE during Stage 1 MIM, learning domain-specific features
  - Stage 2 can freeze or fine-tune the learned encoder
"""

from __future__ import annotations

from typing import Tuple, Optional
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from grid_diff_tcn.hier.frame_layer.model import HierarchicalGridDiffProbTransformer
from grid_diff_tcn.hier.frame_layer.dinov3_features import DinoV3FeatureExtractor, DINOV3_MODELS
from grid_diff_tcn.masked_v2.masks import CenterMask, MaskedImageModelingLoss
from grid_diff_tcn.masked_v2.decoder import PixelDecoder


class MaskedPixelModel(nn.Module):
    """
    Two-stage model for drilling hole detection with masked image modeling.

    Stage 1: Pre-train encoder + decoder to reconstruct masked pixels
    Stage 2: Fine-tune with classification head

    Architecture:
    - DINOv3 encoder (TRAINABLE during Stage 1 MIM in v2 - key difference from v1)
    - Pixel decoder (for stage 1 reconstruction)
    - Frame encoder + Layer TCN + Classification head (for stage 2)
    """
    
    def __init__(
        self,
        dinov3_model: str = "vit_small",
        dinov3_feat_dim: int = 384,
        dinov3_roi_size: int = 224,
        frame_channels: Tuple[int, ...] = (128, 128),
        layer_tcn_channels: Tuple[int, ...] = (128, 128),
        kernel_size: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_transformer_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        add_kl: bool = True,
        use_multiscale: bool = True,
        freeze_encoder: bool = False,
        mask_ratio: float = 0.75,
        mask_shape: str = "circle",
        decoder_hidden_dim: int = 512,
        stage: int = 2,
        dinov3_chunk_size: int = 4,
        use_cached_features: bool = False,
    ) -> None:
        super().__init__()
        self.stage = int(stage)
        self.dinov3_model = str(dinov3_model)
        self.dinov3_feat_dim = int(dinov3_feat_dim)
        self.dinov3_roi_size = int(dinov3_roi_size)
        self.freeze_encoder = bool(freeze_encoder)
        self.dinov3_chunk_size = int(dinov3_chunk_size)
        self.use_cached_features = bool(use_cached_features)

        # Stage 2 inference with pre-extracted features: skip encoder/decoder to save memory.
        # Stage 1 MIM or Stage 2 with raw images: always build encoder + decoder.
        if self.use_cached_features and self.stage == 2:
            # Feature extraction is done externally; model only needs classifier.
            pass
        else:
            mask_size = int(dinov3_roi_size * (1 - mask_ratio ** 0.5))
            self.mask_size = mask_size
            self.center_mask = CenterMask(
                mask_ratio=float(mask_ratio),
                mask_shape=str(mask_shape),
                image_size=int(dinov3_roi_size),
            )

            self.dinov3_extractor = DinoV3FeatureExtractor(
                model_name=str(dinov3_model),
                pretrained=True,
                pool_strategy="cls",
                image_size=int(dinov3_roi_size),
            )

            if self.freeze_encoder:
                for param in self.dinov3_extractor.parameters():
                    param.requires_grad = False

            self.pixel_decoder = PixelDecoder(
                encoder_dim=int(dinov3_feat_dim),
                hidden_dim=int(decoder_hidden_dim),
                output_channels=3,
                output_size=mask_size,
            )

            self.mim_loss = MaskedImageModelingLoss(loss_type="l1")
        
        self.classifier = HierarchicalGridDiffProbTransformer(
            in_channels_frame=int(dinov3_feat_dim),
            out_channels=2,
            frame_channels=frame_channels,
            layer_tcn_channels=layer_tcn_channels,
            kernel_size=kernel_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_transformer_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            add_kl=add_kl,
            use_multiscale=use_multiscale,
        )
    
    def forward_stage1(
        self,
        images: torch.Tensor,
    ) -> dict:
        """
        Forward pass for stage 1 (pre-training).
        
        Args:
            images: (N, 3, H, W) - ROI images, processed in chunks to save memory
        
        Returns:
            dict with keys:
                - loss: scalar MIM loss
                - pred: (N, 3, H, W) reconstructed pixels
                - target: (N, 3, H, W) original pixels
                - mask: (N, H, W) boolean mask (True = masked region)
        """
        N = images.shape[0]
        H, W = images.shape[2], images.shape[3]
        
        masked_images, mask = self.center_mask(images, return_mask=True)
        
        chunk_size = self.dinov3_chunk_size
        all_features = []
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            chunk = masked_images[start:end]
            feat = self.dinov3_extractor(chunk)
            all_features.append(feat)
        features = torch.cat(all_features, dim=0)
        
        pred = self.pixel_decoder(features)  # (N, 3, mask_size, mask_size)
        
        target = images
        
        if pred.shape[-1] != target.shape[-1] or pred.shape[-2] != target.shape[-2]:
            pred = F.interpolate(
                pred,
                size=(H, W),
                mode="bilinear",
                align_corners=False,
            )
        
        loss = self.mim_loss(pred, target, mask)
        
        return {"loss": loss, "pred": pred, "target": target, "mask": mask}
    
    def forward_stage2(
        self,
        images_or_features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Forward pass for stage 2 (classification).

        Args:
            images_or_features: (B, T, F, 3, H, W) raw images  OR
                                (B, T, F, C) cached DINOv3 features
            frame_mask: (B, T, F) - valid frame mask

        Returns:
            logits: (B, 2, T) - classification logits
        """
        # Detect input format from total dimensionality
        # Raw images:   (B, T, F, 3, H, W) → dim() == 6
        # Cached feats: (B, T, F, feat_dim) → dim() == 4
        ndim = images_or_features.dim()

        if ndim == 6:
            # Raw images: (B, T, F, 3, H, W)
            B, T, F = images_or_features.shape[:3]
            H, W = images_or_features.shape[-2:]
            images_4d = images_or_features.reshape(B * T * F, 3, H, W)
            chunk_size = self.dinov3_chunk_size
            all_features = []
            for start in range(0, images_4d.shape[0], chunk_size):
                end = min(start + chunk_size, images_4d.shape[0])
                feat = self.dinov3_extractor(images_4d[start:end])
                all_features.append(feat)
            features = torch.cat(all_features, dim=0)
            features = features.reshape(B, T, F, -1)
        else:
            # Cached features: (B, T, F, feat_dim)
            features = images_or_features

        logits = self.classifier(features, frame_mask=frame_mask)
        if isinstance(logits, dict):
            logits = logits["logits"]
        return logits

    def _forward_stage2_from_features(
        self,
        features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        Forward pass for stage 2 from pre-extracted features.
        Skips internal feature extraction - assumes features already extracted.

        Args:
            features: (B, T, F, C) - pre-extracted DINOv3 features
            frame_mask: (B, T, F) - valid frame mask

        Returns:
            dict with "logits": (B, 2, T)
        """
        logits = self.classifier(features, frame_mask=frame_mask)
        if isinstance(logits, dict):
            logits = logits["logits"]
        return {"logits": logits}

    def get_features(
        self,
        images_or_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Extract DINOv3 features from images, or return cached features as-is.

        Args:
            images_or_features: (B, T, F, 3, H, W) or (B, T, F, C)

        Returns:
            features: (B, T, F, feat_dim)
        """
        if self.use_cached_features:
            return images_or_features

        B, T, F = images_or_features.shape[:3]
        H, W = images_or_features.shape[-2:]
        images_4d = images_or_features.reshape(B * T * F, 3, H, W)

        chunk_size = self.dinov3_chunk_size
        all_features = []
        for start in range(0, images_4d.shape[0], chunk_size):
            end = min(start + chunk_size, images_4d.shape[0])
            feat = self.dinov3_extractor(images_4d[start:end])
            all_features.append(feat)
        features = torch.cat(all_features, dim=0)

        features = features.reshape(B, T, F, -1)

        return features
    
    def forward(
        self,
        images_or_features: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        return_features: bool = False,
        return_decision_idx: bool = False,
    ) -> dict:
        """
        Unified forward pass.

        Args:
            images_or_features: (B, T, F, 3, H, W) raw images  OR
                                (B, T, F, C) cached DINOv3 features
            frame_mask: (B, T, F)
            return_features: whether to return extracted features
            return_decision_idx: whether to also return learned decision indices (Stage 2 only)

        Returns:
            dict with keys:
              - logits: (B, 2, T) — classification logits
              - decision_idx: (B,) — learned layer index predictions (if return_decision_idx=True)
              - features: (B, T, F, feat_dim) — extracted features (if return_features=True)
        """
        if self.stage == 1:
            H, W = images_or_features.shape[-2:]
            images_flat = images_or_features.reshape(-1, 3, H, W)
            result = self.forward_stage1(images_flat)
            return result
        else:
            ndim = images_or_features.dim()
            if ndim == 6:
                B, T, F = images_or_features.shape[:3]
                H, W = images_or_features.shape[-2:]
                images_4d = images_or_features.reshape(B * T * F, 3, H, W)
                chunk_size = self.dinov3_chunk_size
                all_features = []
                for start in range(0, images_4d.shape[0], chunk_size):
                    end = min(start + chunk_size, images_4d.shape[0])
                    feat = self.dinov3_extractor(images_4d[start:end])
                    all_features.append(feat)
                features = torch.cat(all_features, dim=0).reshape(B, T, F, -1)
            else:
                features = images_or_features

            out_dict: dict = {}

            if return_decision_idx:
                # Classifier forward with decision_idx — returns dict
                classifier_ret = self.classifier(
                    features,
                    frame_mask=frame_mask,
                    return_decision_idx=True,
                )
                out_dict["logits"] = classifier_ret["logits"]
                out_dict["decision_idx"] = classifier_ret.get("decision_idx")
            else:
                result = self._forward_stage2_from_features(features, frame_mask=frame_mask)
                out_dict["logits"] = result["logits"]

            if return_features:
                out_dict["features"] = features
            return out_dict
    
    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features for downstream tasks.
        
        Args:
            images: (B, T, F, 3, H, W)
        
        Returns:
            features: (B, T, F, feat_dim)
        """
        return self.get_features(images)

    def set_encoder_trainable(self, trainable: bool) -> None:
        """
        Enable or disable encoder gradient (for Stage 1 vs Stage 2 switching).
        """
        if hasattr(self, "dinov3_extractor"):
            for param in self.dinov3_extractor.parameters():
                param.requires_grad = trainable

    def freeze_classifier(self) -> None:
        """Freeze classifier parameters (e.g., during Stage 1 encoder+decoder pre-training)."""
        for param in self.classifier.parameters():
            param.requires_grad = False

    def unfreeze_classifier(self) -> None:
        """Unfreeze classifier parameters."""
        for param in self.classifier.parameters():
            param.requires_grad = True


def load_masked_model(
    checkpoint_path: str,
    stage: int = 2,
    unfreeze_encoder: bool = False,
    finetune_classifier: bool = True,
    **kwargs,
) -> MaskedPixelModel:
    """
    Load a pretrained masked model.

    Supports loading Stage 1 checkpoint into Stage 2 model:
    - Stage 1 weights (dinov3_extractor, pixel_decoder) → Stage 2 model
    - If finetune_classifier=True: try to load classifier weights from Stage 1 checkpoint
    - If finetune_classifier=False: classifier is initialized from scratch (default behavior)

    Args:
        checkpoint_path: path to checkpoint
        stage: 1 or 2
        unfreeze_encoder: if True, unfreeze encoder after loading (for Stage 2 fine-tuning)
        finetune_classifier: if True, try to load and finetune Stage 1 classifier weights
        **kwargs: passed to MaskedPixelModel __init__

    Returns:
        model: loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    config = checkpoint.get("config", {})
    config.update(kwargs)
    config["stage"] = int(stage)

    model = MaskedPixelModel(**config)

    sd = checkpoint.get("state_dict", checkpoint)
    pretrained_sd = {k.replace("module.", ""): v for k, v in sd.items()}
    loaded_keys = set()

    if int(stage) == 2:
        encoder_keys = [k for k in pretrained_sd if k.startswith("dinov3_extractor.") or k.startswith("pixel_decoder.")]

        matched = {}
        for k in encoder_keys:
            model_key = k.replace("dinov3_extractor.", "dinov3_extractor.").replace("pixel_decoder.", "pixel_decoder.")
            if model_key in model.state_dict():
                matched[model_key] = pretrained_sd[k]
                loaded_keys.add(k)

        mismatched = model.load_state_dict(matched, strict=False)
        for k in mismatched.missing_keys:
            print(f"  [load] warning: missing key: {k}")
        for k in mismatched.unexpected_keys:
            print(f"  [load] warning: unexpected key: {k}")

        print(f"  [load] Stage1→Stage2: loaded {len(loaded_keys)} encoder/decoder keys")

        # Try to load classifier weights for finetuning
        if finetune_classifier:
            classifier_keys = [k for k in pretrained_sd if k.startswith("classifier.")]
            if classifier_keys:
                classifier_matched = {}
                for k in classifier_keys:
                    model_key = k.replace("classifier.", "classifier.")
                    if model_key in model.state_dict():
                        # Check shape compatibility
                        if model.state_dict()[model_key].shape == pretrained_sd[k].shape:
                            classifier_matched[model_key] = pretrained_sd[k]
                            loaded_keys.add(k)
                        else:
                            print(f"  [load] skip classifier.{k[len('classifier.'):]}: shape mismatch "
                                  f"(checkpoint: {pretrained_sd[k].shape}, model: {model.state_dict()[model_key].shape})")
                
                if classifier_matched:
                    model.load_state_dict(classifier_matched, strict=False)
                    print(f"  [load] Stage1→Stage2: loaded {len(classifier_matched)} classifier keys for finetuning")
                else:
                    print(f"  [load] Stage1→Stage2: no compatible classifier keys found, using initialized weights")

        if unfreeze_encoder and hasattr(model, "dinov3_extractor"):
            for param in model.dinov3_extractor.parameters():
                param.requires_grad = True
            print("  [load] encoder unfrozen for fine-tuning")
    else:
        model.load_state_dict(sd, strict=False)

    return model