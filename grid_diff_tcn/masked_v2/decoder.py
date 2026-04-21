# -*- coding: utf-8 -*-
"""
Pixel decoder for masked image modeling.
Reconstructs the masked region pixels from visible region features.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class PixelDecoder(nn.Module):
    """
    Lightweight decoder that reconstructs masked pixel values.
    
    Architecture:
    - Input: encoder features (CLS token + optional patch features)
    - Output: reconstructed pixels for masked region
    
    Args:
        encoder_dim: dimension of encoder features (e.g., 768 for DINOv3)
        hidden_dim: hidden dimension in decoder MLP
        output_shape: (C, H, W) of output pixel values
    """
    
    def __init__(
        self,
        encoder_dim: int = 768,
        hidden_dim: int = 512,
        output_channels: int = 3,
        output_size: int = 168,
    ) -> None:
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_channels = int(output_channels)
        self.output_size = int(output_size)
        
        output_dim = int(output_channels) * int(output_size) * int(output_size)
        
        self.decoder = nn.Sequential(
            nn.Linear(int(encoder_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(int(hidden_dim), output_dim),
        )
    
    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: (B, encoder_dim) or (B*T*F, encoder_dim)
        
        Returns:
            reconstructed: (B, C, H, W) or (B*T*F, C, H, W)
        """
        return self.decoder(encoder_features).reshape(
            -1, self.output_channels, self.output_size, self.output_size
        )


class PixelDecoderV2(nn.Module):
    """
    Enhanced pixel decoder with transposed convolutions for better spatial reasoning.
    """
    
    def __init__(
        self,
        encoder_dim: int = 768,
        hidden_channels: int = 64,
        output_channels: int = 3,
        output_size: int = 168,
    ) -> None:
        super().__init__()
        self.encoder_dim = int(encoder_dim)
        self.hidden_channels = int(hidden_channels)
        self.output_channels = int(output_channels)
        self.output_size = int(output_size)
        
        self.proj = nn.Linear(int(encoder_dim), int(hidden_channels) * 4 * 4)
        
        self.decode_blocks = nn.Sequential(
            nn.ConvTranspose2d(
                int(hidden_channels),
                int(hidden_channels),
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(int(hidden_channels)),
            nn.GELU(),
            nn.ConvTranspose2d(
                int(hidden_channels),
                int(hidden_channels),
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.BatchNorm2d(int(hidden_channels)),
            nn.GELU(),
            nn.Conv2d(
                int(hidden_channels),
                int(output_channels),
                kernel_size=3,
                stride=1,
                padding=1,
            ),
        )
    
    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: (B, encoder_dim)
        
        Returns:
            reconstructed: (B, C, H, W)
        """
        B = encoder_features.shape[0]
        
        x = self.proj(encoder_features)  # (B, hidden_channels * 4 * 4)
        x = x.reshape(B, self.hidden_channels, 4, 4)  # (B, C, 4, 4)
        
        x = self.decode_blocks(x)  # (B, C, H', W')
        
        if x.shape[-1] != self.output_size or x.shape[-2] != self.output_size:
            x = F.interpolate(
                x,
                size=(self.output_size, self.output_size),
                mode="bilinear",
                align_corners=False,
            )
        
        return x


class SpatialDecoder(nn.Module):
    """
    Decoder that treats the problem as seq2seq: 
    predict each pixel position from encoded features.
    """
    
    def __init__(
        self,
        encoder_dim: int = 768,
        hidden_dim: int = 512,
        num_pixels: int = 28224,  # 168 * 168
        output_channels: int = 3,
    ) -> None:
        super().__init__()
        self.num_pixels = int(num_pixels)
        self.output_channels = int(output_channels)
        
        self.feature_proj = nn.Linear(int(encoder_dim), int(hidden_dim))
        
        self.pos_embedding = nn.Embedding(int(num_pixels), int(hidden_dim))
        
        self.decoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=int(hidden_dim),
                nhead=4,
                dim_feedforward=int(hidden_dim),
                dropout=0.1,
                batch_first=True,
            ),
            num_layers=2,
        )
        
        self.head = nn.Linear(int(hidden_dim), int(output_channels))
    
    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            encoder_features: (B, encoder_dim)
        
        Returns:
            reconstructed: (B, num_pixels, C)
        """
        B = encoder_features.shape[0]
        
        feat = self.feature_proj(encoder_features)  # (B, hidden_dim)
        
        pos_ids = torch.arange(self.num_pixels, device=encoder_features.device)
        pos_emb = self.pos_embedding(pos_ids)  # (num_pixels, hidden_dim)
        
        x = feat.unsqueeze(1) + pos_emb.unsqueeze(0)  # (B, num_pixels, hidden_dim)
        
        x = self.decoder(x)  # (B, num_pixels, hidden_dim)
        
        x = self.head(x)  # (B, num_pixels, C)
        
        size = int(self.num_pixels ** 0.5)
        x = x.reshape(B, size, size, self.output_channels)
        x = x.transpose(1, 2).transpose(2, 3)  # (B, C, H, W)
        
        return x