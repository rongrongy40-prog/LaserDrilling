# -*- coding: utf-8 -*-
"""
Two-stage training for drilling hole detection (v2 - Standard MAE pre-training).

Stage 1: Standard MAE with FROZEN pretrained encoder
        - Encoder: DINOv3 pretrained backbone (frozen)
        - Decoder: Transformer + pixel head, trains to reconstruct masked patches
        - Benefit: encoder features stay intact; decoder learns domain-adapted representations

Stage 2: Supervised classification fine-tuning (encoder frozen or fine-tuned)
        - Encoder: frozen pretrained features (or fine-tuned)
        - Classifier: Frame TCN + Layer TCN + ProbTransformer head
        - LearnedDecisionHead for direct layer index prediction

This replaces the old CenterMask + CLS-only PixelDecoder approach which had:
  - Decoder input was only a single CLS token (no spatial info)
  - Wrong mask size calculation (30x30 on 224x224 image)
  - MIM objective unrelated to classification task
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
from grid_diff_tcn.masked_v2.mae import StandardMAEPreTrainer


class MaskedPixelModel(nn.Module):
    """
    Two-stage model for drilling hole detection with masked image modeling.

    Stage 1: Standard MAE pre-training with FROZEN pretrained encoder
             (replaces old CenterMask + CLS-only PixelDecoder)
    Stage 2: Fine-tune with classification head

    Architecture:
    - DINOv3 encoder (FROZEN during Stage 1 - uses pretrained features directly)
    - MAE decoder: visible patches + mask tokens → per-patch pixel reconstruction
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
        mae_decoder_dim: int = 256,
        mae_decoder_depth: int = 4,
        mae_decoder_heads: int = 6,
    ) -> None:
        super().__init__()
        self.stage = int(stage)
        self.dinov3_model = str(dinov3_model)
        self.dinov3_feat_dim = int(dinov3_feat_dim)
        self.dinov3_roi_size = int(dinov3_roi_size)
        self.freeze_encoder = bool(freeze_encoder)
        self.dinov3_chunk_size = int(dinov3_chunk_size)
        self.use_cached_features = bool(use_cached_features)
        self.mask_ratio = float(mask_ratio)

        # Stage 2 inference with pre-extracted features: skip encoder/decoder to save memory.
        # Stage 1 MIM or Stage 2 with raw images: always build encoder + MAE decoder.
        if self.use_cached_features and self.stage == 2:
            pass
        else:
            self.dinov3_extractor = DinoV3FeatureExtractor(
                model_name=str(dinov3_model),
                pretrained=True,
                pool_strategy="cls",
                image_size=int(dinov3_roi_size),
            )

            # MAE pre-trainer: encoder + decoder, both optionally trainable
            # Direction B: encoder unfrozen → learns domain-adapted features
            self.mae_pretrainer = StandardMAEPreTrainer(
                encoder=self.dinov3_extractor.backbone,
                encoder_dim=int(dinov3_feat_dim),
                patch_size=16,
                image_size=int(dinov3_roi_size),
                mask_ratio=float(mask_ratio),
                decoder_dim=int(mae_decoder_dim),
                decoder_depth=int(mae_decoder_depth),
                decoder_heads=int(mae_decoder_heads),
                freeze_encoder=self.freeze_encoder,
            )
        
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
        Forward pass for stage 1: Standard MAE pre-training.
        
        Args:
            images: (N, 3, H, W) - ROI images
        
        Returns:
            dict with keys:
                - loss: scalar MAE loss
                - pred: (N, 3, H, W) reconstructed pixels
                - target: (N, 3, H, W) original pixels
                - mask_img: (N, H, W) float mask (1=masked, 0=visible)
        """
        return self.mae_pretrainer(images, return_loss=True)
    
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
        Enable or disable encoder gradient.
        Note: encoder trainability is controlled by freeze_encoder in __init__.
        Direction B: encoder is trainable during Stage 1 for joint fine-tuning.
        """
        if hasattr(self, "dinov3_extractor"):
            for param in self.dinov3_extractor.parameters():
                param.requires_grad = trainable
        if hasattr(self, "mae_pretrainer") and hasattr(self.mae_pretrainer, "encoder"):
            for param in self.mae_pretrainer.encoder.parameters():
                param.requires_grad = trainable

    def set_mae_trainable(self, trainable: bool) -> None:
        """Enable or disable MAE decoder gradient (Stage 1 only)."""
        if hasattr(self, "mae_pretrainer"):
            for param in self.mae_pretrainer.decoder.parameters():
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
    encoder_checkpoint: str | None = None,
    **kwargs,
) -> MaskedPixelModel:
    """
    Load a pretrained masked model.

    Supports loading Stage 1 checkpoint into Stage 2 model:
    - Stage 1 weights (dinov3_extractor.backbone + mae_pretrainer) → Stage 2 model
    - If finetune_classifier=True: try to load classifier weights from Stage 1 checkpoint
    - If finetune_classifier=False: classifier is initialized from scratch

    Also supports loading encoder and classifier from separate checkpoints:
    - encoder_checkpoint: path to Stage 1 checkpoint (provides backbone + mae_pretrainer weights)
    - checkpoint_path: path to Stage 2 checkpoint (provides classifier weights)

    Args:
        checkpoint_path: path to Stage 2 checkpoint (classifier weights)
        stage: 1 or 2
        unfreeze_encoder: if True, unfreeze encoder after loading
        finetune_classifier: if True, try to load classifier weights
        encoder_checkpoint: optional path to Stage 1 checkpoint for encoder weights
        **kwargs: passed to MaskedPixelModel __init__

    Returns:
        model: loaded model
    """
    config = {}
    sd_classifier = {}
    import os
    # Load classifier checkpoint (stage 2)
    if os.path.exists(checkpoint_path):
        ckpt_clf = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        config.update(ckpt_clf.get("config", {}))
        sd_classifier = ckpt_clf.get("state_dict", ckpt_clf)
        sd_classifier = {k.replace("module.", ""): v for k, v in sd_classifier.items()}
    else:
        print(f"  [load] WARNING: classifier checkpoint not found: {checkpoint_path}")

    # Load encoder checkpoint (stage 1) if provided
    sd_encoder = {}
    if encoder_checkpoint and os.path.exists(encoder_checkpoint):
        ckpt_enc = torch.load(encoder_checkpoint, map_location="cpu", weights_only=True)
        # Merge encoder config (but classifier config takes priority)
        enc_config = ckpt_enc.get("config", {})
        for k, v in enc_config.items():
            if k not in config:
                config[k] = v
        sd_encoder = ckpt_enc.get("state_dict", ckpt_enc)
        sd_encoder = {k.replace("module.", ""): v for k, v in sd_encoder.items()}
        print(f"  [load] Loaded encoder from: {encoder_checkpoint}")
    elif checkpoint_path and os.path.exists(checkpoint_path):
        # Fallback: try loading encoder from the same checkpoint
        sd_encoder = sd_classifier

    config.update(kwargs)
    config["stage"] = int(stage)

    model = MaskedPixelModel(**config)

    model_sd = model.state_dict()
    matched = {}

    # ---- Load encoder weights from encoder checkpoint (stage 1) ----
    # 1. MAE decoder weights
    for k in sd_encoder:
        if k.startswith("mae_pretrainer."):
            if k in model_sd and model_sd[k].shape == sd_encoder[k].shape:
                matched[k] = sd_encoder[k]

    # 2. Encoder backbone
    for k in sd_encoder:
        if k.startswith("dinov3_extractor.backbone."):
            if k in model_sd and model_sd[k].shape == sd_encoder[k].shape:
                matched[k] = sd_encoder[k]

    mismatched = model.load_state_dict(matched, strict=False)
    for k in mismatched.missing_keys:
        if not k.startswith("classifier."):
            print(f"  [load] missing key (not in checkpoint): {k}")
    for k in mismatched.unexpected_keys:
        if "classifier" not in k:
            print(f"  [load] unexpected key (not in model): {k}")

    print(f"  [load] Stage1→Stage2: loaded {len(matched)} keys "
          f"(encoder={sum(1 for k in matched if 'backbone' in k)}, "
          f"mae_decoder={sum(1 for k in matched if 'mae_pretrainer.decoder' in k)})")

    # ---- Load classifier weights from classifier checkpoint (stage 2) ----
    if finetune_classifier and sd_classifier:
        classifier_matched = {}
        for k in sd_classifier:
            if k.startswith("classifier."):
                if k in model_sd:
                    if model_sd[k].shape == sd_classifier[k].shape:
                        classifier_matched[k] = sd_classifier[k]
                    else:
                        print(f"  [load] skip classifier.{k[11:]}: shape mismatch "
                              f"({model_sd[k].shape} vs {sd_classifier[k].shape})")

        if classifier_matched:
            model.load_state_dict(classifier_matched, strict=False)
            print(f"  [load] Stage2: loaded {len(classifier_matched)} classifier keys")
        else:
            print(f"  [load] Stage2: no compatible classifier keys, using initialized weights")

    if unfreeze_encoder and hasattr(model, "dinov3_extractor"):
        for param in model.dinov3_extractor.parameters():
            param.requires_grad = True
        print("  [load] encoder unfrozen for fine-tuning")

    return model