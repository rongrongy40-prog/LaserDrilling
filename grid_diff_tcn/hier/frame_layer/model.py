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


class SoftArgmax1D(nn.Module):
    """
    可微 argmax：将 (B, T) 概率向量映射为 (B,) 期望位置（连续值）。
    公式:  pred = sum_i( softmax(prob / T)[i] * i )
    T 是温度参数，越小分布越尖锐。设为可学习参数。
    """

    def __init__(self, t_max: int = 300, init_temperature: float = 1.0) -> None:
        super().__init__()
        self.t_max = t_max
        # 初始温度 1.0，对应均匀分布；训练中会学到合适的温度
        self.log_temperature = nn.Parameter(torch.tensor(init_temperature).log())

    def forward(self, prob: torch.Tensor) -> torch.Tensor:
        # prob: (B, T)，已经过概率归一化或未归一化
        temperature = self.log_temperature.exp()  # (1,)
        indices = torch.arange(prob.shape[1], device=prob.device, dtype=prob.dtype)
        weights = F.softmax(prob / temperature.clamp(min=0.05), dim=1)  # (B, T)
        pred = (weights * indices.unsqueeze(0)).sum(dim=1)  # (B,)
        return pred


class LearnedDecisionHead(nn.Module):
    """
    训练+推理两用的决策头。
    
    训练时：使用概率期望（可微，可以训练分类器）
    推理时：使用无参的early stop逻辑（找到第一个概率>threshold的层）
    
    优势：
    - 训练时端到端可微
    - 推理时无需调参，固定阈值0.5
    - 符合工业现场的early stop需求
    """

    def __init__(
        self,
        d_model: int = 128,
        use_attention: bool = True,
        t_max: int = 300,
        inference_threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.t_max = t_max
        self.inference_threshold = inference_threshold

    def forward(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        自动选择训练/推理模式：
        - 训练时(self.training=True): 用概率期望（可微）
        - 推理时(self.training=False): 用first-above-threshold（无参early stop）
        
        z: (B, T, d_model) — transformer 输出
        logits: (B, 2, T) — 原始分类 logits  
        frame_mask: (B, T, F) — 帧有效掩码

        Returns:
            pred_idx: (B,) — 预测的 0-based 层索引
        """
        if self.training:
            # 训练模式：概率期望（可微）
            return self.forward_training(z, logits, frame_mask)
        else:
            # 推理模式：first-above-threshold（无参early stop）
            return self._find_first_above_threshold_batch(logits, frame_mask)

    def _find_first_above_threshold_batch(
        self,
        logits: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        """
        批量推理模式：从前往后扫，找到第一个概率>threshold的层
        
        这是一个无参操作，固定阈值0.5。
        符合工业现场的early stop需求。
        """
        b, _, t = logits.shape
        
        raw_prob = F.softmax(logits, dim=1)[:, 1]  # (B, T)
        
        if frame_mask is not None:
            mask_2d = frame_mask.any(dim=2)  # (B, T)
        else:
            mask_2d = torch.ones(b, t, dtype=torch.bool, device=logits.device)

        indices = torch.arange(t, device=logits.device).unsqueeze(0)  # (1, T)
        
        pred_idx = torch.zeros(b, device=logits.device)
        
        for bi in range(b):
            first_idx = -1
            for ti in range(t):
                if mask_2d[bi, ti] and raw_prob[bi, ti] > threshold:
                    first_idx = ti
                    break
            if first_idx >= 0:
                pred_idx[bi] = float(first_idx)
            else:
                # Fallback: 概率期望
                valid_mask = mask_2d[bi]
                valid_prob = raw_prob[bi][valid_mask]
                if valid_prob.numel() > 0:
                    valid_indices = indices[0][valid_mask]
                    valid_prob_norm = valid_prob / valid_prob.sum().clamp(min=1e-8)
                    pred_idx[bi] = (valid_prob_norm * valid_indices).sum()
                else:
                    pred_idx[bi] = 0.0
        
        return pred_idx

    def forward_training(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        训练时使用的版本：用概率期望（可微）
        
        这样分类器可以学到更好的概率分布。
        """
        b, t, _ = z.shape
        
        raw_prob = F.softmax(logits, dim=1)[:, 1]  # (B, T)
        
        if frame_mask is not None:
            mask_2d = frame_mask.any(dim=2)
            raw_prob = raw_prob.masked_fill(~mask_2d, 0.0)
        else:
            mask_2d = torch.ones(b, t, dtype=torch.bool, device=z.device)

        # 概率期望
        indices = torch.arange(t, device=logits.device, dtype=logits.dtype)
        prob_sum = raw_prob.sum(dim=1, keepdim=True).clamp(min=1e-8)
        prob_norm = raw_prob / prob_sum
        pred_idx = (prob_norm * indices.unsqueeze(0)).sum(dim=1)
        
        return pred_idx


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
    - Streaming support: can run in batch mode or step-by-step mode
    """

    def __init__(
        self,
        in_channels_frame: int = 768,
        out_channels: int = 2,
        frame_channels: Tuple[int, ...] = (128, 128),
        layer_tcn_channels: Tuple[int, ...] = (128, 128),
        kernel_size: int = 3,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        add_kl: bool = True,
        return_kl: bool = False,
        extra_dim: int = 0,
        use_frame_gru: bool = True,
        use_frame_attn_pool: bool = True,
        frame_gru_layers: int = 1,
        use_multiscale: bool = True,
        # Streaming decision parameters
        decision_threshold: float = 0.5,
        decision_wait: int = 3,
    ) -> None:
        super().__init__()
        self.return_kl = bool(return_kl)
        self.out_channels = int(out_channels)
        self.use_frame_gru = use_frame_gru
        self.use_frame_attn_pool = use_frame_attn_pool
        self.use_multiscale = use_multiscale
        self.decision_threshold = decision_threshold
        self.decision_wait = decision_wait

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

        # Learned decision head: can work in batch or streaming mode
        self.decision_head = LearnedDecisionHead(d_model=int(d_model), use_attention=True)

    def forward(
        self,
        x: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
        force_sample_attention: bool = False,
        layer_extra: torch.Tensor | None = None,
        return_decision_idx: bool = False,
    ):
        """
        x: (B,T,F,C)
        frame_mask: (B,T,F)
        return_decision_idx: if True, also compute and return learned decision indices
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

        z_for_head = z.transpose(1, 2)  # (B,D,T)
        logits = self.head(z_for_head)  # (B,2,T)

        ret = {"logits": logits}
        if self.return_kl and kl_terms:
            ret["kl_loss"] = torch.stack(kl_terms).mean()
        if return_decision_idx:
            ret["decision_idx"] = self.decision_head(z, logits, frame_mask)  # (B,)
        return ret

    # ------------------------------------------------------------------
    # Streaming inference support
    # ------------------------------------------------------------------

    def reset_hidden(self) -> None:
        """
        Reset all streaming state. Call this before starting a new well.
        """
        self._z_seq: list[torch.Tensor] = []
        self._logits_seq: list[torch.Tensor] = []
        self._kv_caches: list[list[torch.Tensor] | None] = [
            None for _ in self.transformer_layers
        ]
        self._step_count = 0

    def forward_step(
        self,
        x_step: torch.Tensor,
        frame_mask_step: torch.Tensor | None = None,
    ) -> dict:
        """
        Streaming forward: process one layer's frame features at a time.

        Call reset_hidden() once before the first step.

        Args:
            x_step: (B, 1, F, C) — one layer's frame features
            frame_mask_step: (B, 1, F) — mask for this step's frames

        Returns:
            dict with keys:
              - logits_step: (B, 2, 1) — logits for this step only
              - logits_full: (B, 2, t) — all logits accumulated so far
              - prob_step:  (B, 1) — penetration prob for this step
              - prob_full:  (B, t) — penetration probs for all steps
              - decision_idx: (B,) — learned layer index (full sequence)
        """
        b = x_step.shape[0]

        if not hasattr(self, "_z_seq"):
            self.reset_hidden()

        z = self.frame_encoder(x_step, frame_mask=frame_mask_step)
        z = z.squeeze(2) if z.shape[2] == 1 else z  # (B, D) -> squeeze T=1

        for li, blk in enumerate(self.layer_tcn):
            z = blk(z)

        z = self.proj_in(z)  # (B, d_model)
        z_step = z.unsqueeze(1)  # (B, 1, d_model)

        if self._step_count == 0:
            z_acc = z_step
        else:
            z_acc = torch.cat([self._z_seq[-1], z_step], dim=1)  # (B, t+1, D)

        for li, layer in enumerate(self.transformer_layers):
            z_acc, _ = layer(
                z_acc,
                use_cache=False,
                layer_idx=li,
            )

        logits_step = self.head(z_acc[:, -1:, :].transpose(1, 2))  # (B, 2, 1)
        logits_full = torch.cat(
            [torch.zeros(b, 2, self._step_count, device=z_acc.device, dtype=z_acc.dtype)
             if self._step_count > 0 else torch.empty(b, 2, 0, device=z_acc.device, dtype=z_acc.dtype),
             logits_step], dim=2
        )

        prob_step = F.softmax(logits_step, dim=1)[:, 1]  # (B, 1)
        prob_full = F.softmax(logits_full, dim=1)[:, 1]  # (B, t)

        self._z_seq.append(z_acc)
        self._logits_seq.append(logits_step)
        self._step_count += 1

        decision_idx = self.decision_head(z_acc, logits_full, frame_mask=None)

        return {
            "logits_step": logits_step,
            "logits_full": logits_full,
            "prob_step": prob_step,
            "prob_full": prob_full,
            "decision_idx": decision_idx,
        }
