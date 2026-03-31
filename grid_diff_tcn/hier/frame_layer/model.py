# -*- coding: utf-8 -*-
"""
Two-level temporal model:
1) intra-layer frame modeling (F dimension)
2) inter-layer sequence modeling (T dimension)

Keeps probabilistic transformer blocks from existing codebase.
Input shape:
  x: (B, T, F, C) where C is per-frame feature dim (default 64)
  frame_mask: (B, T, F) bool, True=valid frame
Output:
  logits: (B, 2, T)
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn

from grid_diff_tcn.modules import ProbTransformerEncoderLayer, TCNBlock


class FrameLevelTCNEncoder(nn.Module):
    """
    Encode per-layer frame sequence (F) into a single vector per layer.
    """

    def __init__(
        self,
        in_channels: int = 64,
        frame_channels: Tuple[int, ...] = (64, 64),
        kernel_size: int = 3,
        out_dim: int = 64,
    ) -> None:
        super().__init__()
        blocks = []
        ch_in = int(in_channels)
        for i, ch_out in enumerate(frame_channels):
            blocks.append(TCNBlock(ch_in, int(ch_out), int(kernel_size), dilation=(2 ** i)))
            ch_in = int(ch_out)
        self.blocks = nn.ModuleList(blocks)
        self.proj = nn.Conv1d(ch_in, int(out_dim), kernel_size=1)

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        x: (B, T, F, C)
        frame_mask: (B, T, F) bool
        return: layer embeddings (B, out_dim, T)
        """
        b, t, f, c = x.shape
        y = x.reshape(b * t, f, c).transpose(1, 2)  # (B*T, C, F)
        for blk in self.blocks:
            y = blk(y)
        y = self.proj(y)  # (B*T, D, F)
        y = y.transpose(1, 2)  # (B*T, F, D)

        if frame_mask is None:
            pooled = y.mean(dim=1)  # (B*T, D)
        else:
            m = frame_mask.reshape(b * t, f).to(dtype=y.dtype).unsqueeze(-1)  # (B*T,F,1)
            denom = m.sum(dim=1).clamp(min=1.0)
            pooled = (y * m).sum(dim=1) / denom

        pooled = pooled.reshape(b, t, -1).transpose(1, 2)  # (B, D, T)
        return pooled


class HierarchicalGridDiffProbTransformer(nn.Module):
    """
    Hierarchical frame-layer model with probabilistic transformer at layer level.
    """

    def __init__(
        self,
        in_channels_frame: int = 64,
        out_channels: int = 2,
        frame_channels: Tuple[int, ...] = (64, 64),
        layer_tcn_channels: Tuple[int, ...] = (64, 64),
        kernel_size: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        add_kl: bool = True,
        return_kl: bool = False,
        extra_dim: int = 0,
    ) -> None:
        super().__init__()
        self.return_kl = bool(return_kl)
        self.out_channels = int(out_channels)

        self.frame_encoder = FrameLevelTCNEncoder(
            in_channels=int(in_channels_frame),
            frame_channels=frame_channels,
            kernel_size=kernel_size,
            out_dim=layer_tcn_channels[0] if layer_tcn_channels else d_model,
        )
        self.extra_dim = int(max(0, extra_dim))
        if self.extra_dim > 0:
            self.extra_proj = nn.Sequential(
                nn.Linear(self.extra_dim, int(d_model)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(d_model), int(d_model)),
            )
        else:
            self.extra_proj = None

        # layer-level TCN
        layer_blocks = []
        ch_in = layer_tcn_channels[0] if layer_tcn_channels else d_model
        for i, ch_out in enumerate(layer_tcn_channels):
            layer_blocks.append(TCNBlock(int(ch_in), int(ch_out), int(kernel_size), dilation=(2 ** i)))
            ch_in = int(ch_out)
        self.layer_tcn = nn.ModuleList(layer_blocks)

        self.proj_in = nn.Conv1d(ch_in, int(d_model), kernel_size=1) if ch_in != int(d_model) else nn.Identity()

        tf_layers = []
        for _ in range(int(num_layers)):
            tf_layers.append(
                ProbTransformerEncoderLayer(
                    d_model=int(d_model),
                    nhead=int(nhead),
                    dim_feedforward=int(dim_feedforward),
                    dropout=float(dropout),
                    add_kl=bool(add_kl),
                )
            )
        self.transformer_layers = nn.ModuleList(tf_layers)
        self.head = nn.Conv1d(int(d_model), self.out_channels, kernel_size=1)

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        force_sample_attention: bool = False,
        layer_extra: torch.Tensor | None = None,
    ):
        """
        x: (B,T,F,C)
        frame_mask: (B,T,F)
        """
        z = self.frame_encoder(x, frame_mask=frame_mask)  # (B,D,T)
        if self.extra_proj is not None and layer_extra is not None:
            # layer_extra: (B,T,E) -> (B,D,T)
            e = layer_extra.to(dtype=z.dtype)
            e = self.extra_proj(e).transpose(1, 2)
            z = z + e
        for blk in self.layer_tcn:
            z = blk(z)
        z = self.proj_in(z)  # (B,d_model,T)
        z = z.transpose(1, 2)  # (B,T,D)

        kl_terms = []
        for layer in self.transformer_layers:
            z, extra = layer(z, force_sample_kv=force_sample_attention)
            if isinstance(extra, dict) and "kl_loss" in extra:
                kl_terms.append(extra["kl_loss"])

        z = z.transpose(1, 2)  # (B,D,T)
        logits = self.head(z)  # (B,2,T)
        if self.return_kl and kl_terms:
            return logits, {"kl_loss": torch.stack(kl_terms).mean()}
        return logits

