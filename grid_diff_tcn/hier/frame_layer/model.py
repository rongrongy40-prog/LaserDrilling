# -*- coding: utf-8 -*-
"""
Two-level temporal model:
1) intra-layer frame modeling (F dimension)
2) inter-layer sequence modeling (T dimension)

Features:
- Frame-level: TCN + GRU + Attention Pooling (optional)
- Layer-level: TCN + ProbTransformer
- Multi-scale feature fusion (optional)
"""

from __future__ import annotations

from typing import Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import rnn

from grid_diff_tcn.modules import ProbTransformerEncoderLayer, TCNBlock


class AttentionPooling(nn.Module):
    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.key_proj = nn.Linear(embed_dim, embed_dim)
        self.value_proj = nn.Linear(embed_dim, embed_dim)
        self.scale = embed_dim ** -0.5

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        b, seq_len, d = x.shape
        q = self.query.expand(b, -1, -1)  # (B, 1, D)
        k = self.key_proj(x)  # (B, T, D)
        v = self.value_proj(x)  # (B, T, D)

        scores = torch.matmul(q, k.transpose(1, 2)) * self.scale  # (B, 1, T)
        
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).float()  # (B, 1, T)
            scores = scores.masked_fill(mask_expanded == 0, float('-inf'))
        
        attn_weights = F.softmax(scores, dim=-1)  # (B, 1, T)
        pooled = torch.matmul(attn_weights, v)  # (B, 1, D)
        return pooled.squeeze(1)  # (B, D)


class FrameLevelGRUEncoder(nn.Module):
    """
    GRU-based encoder for per-layer frame sequence.
    Better for short sequences (F < 8).
    """

    def __init__(
        self,
        in_channels: int = 768,   # 768 for DINOv3 ViT-B, 192 for hand-crafted grid
        hidden_dim: int = 128,    # larger to handle higher-dim input
        num_layers: int = 1,
        out_dim: int = 128,       # larger output dim for richer features
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            int(in_channels),
            int(hidden_dim),
            num_layers=int(num_layers),
            batch_first=True,
            bidirectional=False,
        )
        self.proj = nn.Linear(int(hidden_dim), int(out_dim)) if int(hidden_dim) != int(out_dim) else nn.Identity()

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, f, c = x.shape
        y = x.reshape(b * t, f, c)  # (B*T, F, C)
        
        if frame_mask is not None:
            m = frame_mask.reshape(b * t, f)
            lengths = m.sum(dim=1).clamp(min=1).long().cpu()
            if lengths.max() > 0:
                y_packed = rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
                output, hidden = self.gru(y_packed)
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
            else:
                output, hidden = self.gru(y)
        else:
            output, hidden = self.gru(y)
        
        last_output = output[:, -1, :]  # (B*T, hidden_dim)
        pooled = self.proj(last_output)  # (B*T, out_dim)
        pooled = pooled.reshape(b, t, -1).transpose(1, 2)  # (B, out_dim, T)
        return pooled


class FrameLevelTCNWithAttn(nn.Module):
    """
    TCN + Attention Pooling for better feature extraction.
    Addresses short sequence issues and feature extraction.
    """

    def __init__(
        self,
        in_channels: int = 768,           # 768 for DINOv3 ViT-B, 192 for hand-crafted grid
        frame_channels: Tuple[int, ...] = (128, 128),  # wider channels for richer features
        kernel_size: int = 3,
        out_dim: int = 128,               # larger to match richer features
        use_gru: bool = True,
        gru_layers: int = 1,
        use_attention_pool: bool = True,
    ) -> None:
        super().__init__()
        self.use_gru = use_gru
        self.use_attention_pool = use_attention_pool
        self.out_dim = int(out_dim)
        
        blocks = []
        ch_in = int(in_channels)
        for i, ch_out in enumerate(frame_channels):
            blocks.append(TCNBlock(ch_in, int(ch_out), int(kernel_size), dilation=(2 ** i)))
            ch_in = int(ch_out)
        self.tcn_blocks = nn.ModuleList(blocks)
        self.tcn_out_dim = ch_in
        
        if use_gru:
            self.gru = nn.GRU(
                int(ch_in),
                int(out_dim),
                num_layers=int(gru_layers),
                batch_first=True,
                bidirectional=False,
            )
            self.gru_proj = nn.Linear(int(out_dim), int(out_dim)) if int(ch_in) != int(out_dim) else nn.Identity()
        else:
            self.gru = None
            self.gru_proj = nn.Linear(int(ch_in), int(out_dim)) if int(ch_in) != int(out_dim) else nn.Identity()
        
        if use_attention_pool:
            self.attn_pool = AttentionPooling(int(out_dim))
        else:
            self.attn_pool = None

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, f, c = x.shape
        y = x.reshape(b * t, f, c).transpose(1, 2)  # (B*T, C, F)
        
        for blk in self.tcn_blocks:
            y = blk(y)
        
        y = y.transpose(1, 2)  # (B*T, F, D)
        
        if self.use_gru:
            if frame_mask is not None:
                m = frame_mask.reshape(b * t, f)
                lengths = m.sum(dim=1).clamp(min=1).long().cpu()
                y_packed = rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
                output, _ = self.gru(y_packed)
                output, _ = rnn.pad_packed_sequence(output, batch_first=True)
            else:
                output, _ = self.gru(y)
            pooled = output[:, -1, :]  # (B*T, D)
            pooled = self.gru_proj(pooled)
        else:
            if frame_mask is not None:
                m = frame_mask.reshape(b * t, f).unsqueeze(-1).float()
                pooled = (y * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)  # (B*T, D)
            else:
                pooled = y.mean(dim=1)  # (B*T, D)
            if self.tcn_out_dim != self.out_dim:
                pooled = self.gru_proj(pooled.unsqueeze(1)).squeeze(1)
        
        if self.use_attention_pool and frame_mask is not None:
            m = frame_mask.reshape(b * t, f).unsqueeze(-1).float()
            weights = torch.softmax(y.sum(dim=-1).masked_fill(m.squeeze(-1) == 0, float('-inf')), dim=1).unsqueeze(-1)
            weighted = (y * weights * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
        
        pooled = pooled.reshape(b, t, -1).transpose(1, 2)  # (B, D, T)
        return pooled


class MultiScaleFrameEncoder(nn.Module):
    """
    Multi-scale feature extraction from frame sequence.
    Extracts features at different temporal scales and fuses them.
    """

    def __init__(
        self,
        in_channels: int = 768,       # 768 for DINOv3 ViT-B, 192 for hand-crafted grid
        out_channels: int = 128,       # wider to handle richer features
        kernel_size: int = 3,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=1)
        self.conv2 = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=2, dilation=2)
        self.conv3 = nn.Conv1d(int(in_channels), int(out_channels), kernel_size=int(kernel_size), padding=4, dilation=4)
        
        self.bn1 = nn.BatchNorm1d(int(out_channels))
        self.bn2 = nn.BatchNorm1d(int(out_channels))
        self.bn3 = nn.BatchNorm1d(int(out_channels))
        
        self.act = nn.ReLU(inplace=True)
        self.fusion = nn.Linear(int(out_channels) * 3, int(out_channels))
        self.attn_pool = AttentionPooling(int(out_channels))

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, f, c = x.shape
        y = x.reshape(b * t, f, c).transpose(1, 2)  # (B*T, C, F)
        
        y1 = self.act(self.bn1(self.conv1(y)))
        y2 = self.act(self.bn2(self.conv2(y)))
        y3 = self.act(self.bn3(self.conv3(y)))
        
        if frame_mask is not None:
            m = frame_mask.reshape(b * t, f).unsqueeze(1).float()
            y1 = y1 * m
            y2 = y2 * m
            y3 = y3 * m
        
        pooled1 = y1.mean(dim=2)
        pooled2 = y2.mean(dim=2)
        pooled3 = y3.mean(dim=2)
        
        fused = torch.cat([pooled1, pooled2, pooled3], dim=-1)
        fused = self.fusion(fused)
        
        fused = fused.reshape(b, t, -1).transpose(1, 2)
        return fused


class HierarchicalGridDiffProbTransformer(nn.Module):
    """
    Hierarchical frame-layer model with:
    - Frame-level: TCN + GRU + Attention Pooling (optional, default True)
    - Layer-level: TCN + ProbTransformer
    - Multi-scale feature fusion (optional, default True)
    """

    def __init__(
        self,
        in_channels_frame: int = 768,           # 768 for DINOv3 ViT-B CLS token, 192 for hand-crafted grid
        out_channels: int = 2,
        frame_channels: Tuple[int, ...] = (128, 128),   # wider for DINOv3 features
        layer_tcn_channels: Tuple[int, ...] = (128, 128),
        kernel_size: int = 3,
        d_model: int = 128,                              # larger for richer features
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,                     # larger feedforward
        dropout: float = 0.1,
        add_kl: bool = True,
        return_kl: bool = False,
        extra_dim: int = 0,
        use_frame_gru: bool = True,
        use_frame_attn_pool: bool = True,
        frame_gru_layers: int = 1,
        use_multiscale: bool = True,
    ) -> None:
        super().__init__()
        self.return_kl = bool(return_kl)
        self.out_channels = int(out_channels)
        self.use_frame_gru = use_frame_gru
        self.use_frame_attn_pool = use_frame_attn_pool
        self.use_multiscale = use_multiscale

        if use_multiscale:
            self.frame_encoder = MultiScaleFrameEncoder(
                in_channels=int(in_channels_frame),
                out_channels=layer_tcn_channels[0] if layer_tcn_channels else d_model,
                kernel_size=kernel_size,
            )
        else:
            self.frame_encoder = FrameLevelTCNWithAttn(
                in_channels=int(in_channels_frame),
                frame_channels=frame_channels,
                kernel_size=kernel_size,
                out_dim=layer_tcn_channels[0] if layer_tcn_channels else d_model,
                use_gru=use_frame_gru,
                gru_layers=int(frame_gru_layers),
                use_attention_pool=use_frame_attn_pool,
            )

        self.extra_dim = int(max(0, extra_dim))
        if self.extra_dim > 0:
            self.extra_proj = nn.Sequential(
                nn.Linear(self.extra_dim, int(d_model)),
                nn.GELU(),
                nn.Dropout(float(dropout)),
                nn.Linear(int(d_model), int(d_model)),
            )
            self.extra_gate = nn.Sequential(
                nn.Linear(int(d_model), int(d_model)),
                nn.GELU(),
                nn.Linear(int(d_model), 1),
                nn.Sigmoid()
            )
        else:
            self.extra_proj = None
            self.extra_gate = None

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
            e = layer_extra.to(dtype=z.dtype)
            e = self.extra_proj(e).transpose(1, 2)
            if self.extra_gate is not None:
                gate = self.extra_gate(z.transpose(1, 2)).transpose(1, 2)
                z = z + gate * e
            else:
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
