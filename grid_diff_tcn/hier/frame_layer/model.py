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


class TemporalDecisionHead(nn.Module):
    """
    训练/推理一致性决策头。

    每个时间步 t 用全部历史均值 + 当前层特征 + lookahead 特征，
    预测 t 是否为首次穿透层（t >= pen_t → 1，t < pen_t → 0）。

    - 训练：BCE loss 在 causal 累积标签上
    - 推理：找第一个 prob > threshold 的位置
    - 两者的 forward 逻辑完全一致，不存在训练/推理 gap

    Args:
        d_model: transformer 输出维度
        lookback: 历史累积层数（设大值，使用全部历史；当前实现为 full lookback）
        lookahead: 向前看几层（默认 2）
        threshold: 推理阈值
    """

    def __init__(
        self,
        d_model: int = 128,
        lookback: int = 999,   # 设大值表示全部历史
        lookahead: int = 2,
        threshold: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.lookback = lookback
        self.lookahead = lookahead
        self.threshold = threshold

        # ctx = [prob_ctx(4) + history_mean(d) + current(d) + future(d)] = 4 + 3d
        self.proj = nn.Sequential(
            nn.Linear(d_model * 3 + 4, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
        frame_mask: torch.Tensor | None = None,
    ) -> dict:
        """
        z: (B, T, d_model) — transformer 输出
        logits: (B, 2, T) — 原始二分类 logits
        frame_mask: (B, T, F) — 帧有效掩码

        Returns:
            dict with keys:
              - decision_logits: (B, T) — 决策头对每个时间步的 logits
              - decision_probs: (B, T) — 决策头对每个时间步的概率 (0..1)
              - pred_idx: (B,) — 推理决策：第一个 prob > threshold 的位置
        """
        decision_logits, decision_probs = self._compute_decision_logits(z, logits)
        pred_idx = self._find_first_above_threshold(decision_probs, frame_mask)
        return {
            "decision_logits": decision_logits,
            "decision_probs": decision_probs,
            "pred_idx": pred_idx,
        }

    def _compute_decision_logits(
        self,
        z: torch.Tensor,
        logits: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        每个时间步 t：
          - 基础概率 prob_base = softmax(logits)[:, 1]
          - 累积历史均值 prob_cummean = mean(prob_base[:, 0..t])   ← 因果累积
          - 当前层 + lookahead 层概率
          - Transformer 特征 (history_mean + current + future)
        用 2 层 MLP 融合上述全部信息，输出每个 t 是否为穿透层。

        训练标签（causal cumulative）：
          causal_label[b, t] = 1 if t >= pen_t[b] else 0
        推理：找第一个 prob > threshold。

        Returns (decision_logits, decision_probs), both (B, T).
        """
        b, t, d = z.shape
        device = z.device

        # ---- 1. 原始分类概率（原始模型已学好的穿透信号） ----
        prob_base = torch.softmax(logits, dim=1)[:, 1]  # (B, T)

        # ---- 2. 概率的历史累积均值（causal cumsum） ----
        # cummean[b, t] = mean(prob_base[b, 0..t])，shape (B, T)
        prob_cumsum = prob_base.cumsum(dim=1)
        count = torch.arange(1, t + 1, device=device, dtype=z.dtype)
        prob_cummean = prob_cumsum / count.unsqueeze(0)  # (B, T)

        # ---- 3. 当前层 + lookahead 层概率 ----
        # prob_current = prob_base  (B, T)
        prob_future = F.pad(prob_base[:, 2:], (0, 2), mode="replicate")  # (B, T)

        # ---- 4. Transformer 特征的因果聚合 ----
        # prefix_sum[b, i] = sum_{j=0..i} z[b, j]
        prefix_sum = z.cumsum(dim=1)
        cnt = torch.arange(1, t + 1, device=device, dtype=z.dtype).unsqueeze(0).unsqueeze(-1)
        history_mean = prefix_sum / cnt.clamp(min=1)   # (B, T, d)
        current = z                                    # (B, T, d)
        future = F.pad(z[:, 2:, :], (0, 0, 0, 2), mode="replicate")  # (B, T, d)

        # ---- 5. 全部拼在一起：概率标量 + Transformer 特征 ----
        # prob: 4 个标量 (cummean, current, lookahead) → expand 成 (B, T, 4)
        prob_ctx = torch.stack([
            prob_cummean,                    # 因果累积均值
            prob_base,                        # 当前层概率
            prob_future,                      # lookahead 层概率
            prob_cummean * prob_base,         # 交互项
        ], dim=-1)                             # (B, T, 4)

        # concat: (B, T, 4 + d + d + d) = (B, T, 4 + 3d)
        ctx = torch.cat([
            prob_ctx,      # (B, T, 4)
            history_mean,  # (B, T, d)
            current,       # (B, T, d)
            future,        # (B, T, d)
        ], dim=-1)      # (B, T, 4 + 3d)

        raw = self.proj(ctx).squeeze(-1)  # (B, T)

        decision_probs = torch.sigmoid(raw)  # (B, T)

        return raw, decision_probs

    def _find_first_above_threshold(
        self,
        probs: torch.Tensor,
        frame_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """
        推理决策：找第一个 prob > threshold 的位置（batch 并行）。
        训练时不用这个，由 BCE loss 覆盖。

        Returns: pred_idx (B,)
        """
        b, t = probs.shape
        device = probs.device

        if frame_mask is not None:
            mask_2d = frame_mask.any(dim=2)  # (B, T)
        else:
            mask_2d = torch.ones(b, t, dtype=torch.bool, device=device)

        # 扩展一列用于 fallback（全 False 时 argmax 返回 0）
        probs_ext = F.pad(probs, (0, 1), value=0.0)     # (B, T+1)
        mask_ext = F.pad(mask_2d, (0, 1), value=False)  # (B, T+1)

        above = (probs_ext > self.threshold) & mask_ext  # (B, T+1)
        first_above = above.long().argmax(dim=1)          # (B,)
        fallback = ~above.any(dim=1)
        first_above = torch.where(fallback, torch.full_like(first_above, t), first_above)
        return first_above.float()  # (B,)


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

    # ------------------------------------------------------------------
    # Streaming support: persist GRU hidden state across steps
    # ------------------------------------------------------------------

    def reset_hidden(self) -> None:
        """Reset streaming state. Call before starting a new well."""
        if self.use_gru:
            self._gru_h = None  # (num_layers, B, D)

    def forward_step(
        self,
        x_step: torch.Tensor,
        frame_mask_step: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Streaming forward: process one layer's frame batch at a time.

        Persists GRU hidden state across steps. Call reset_hidden() once
        before the first step of each new well.

        Args:
            x_step: (B, 1, F, C) — one layer's frame features
            frame_mask_step: (B, 1, F) — mask for this step's frames

        Returns:
            pooled: (B, D, 1) — frame-level features for this layer
        """
        b, t, f, c = x_step.shape  # t==1 for step mode
        y = x_step.reshape(b * t, f, c).transpose(1, 2)  # (B, C, F)

        for blk in self.tcn_blocks:
            y = blk(y)

        y = y.transpose(1, 2)  # (B, F, D)

        if self.use_gru:
            if frame_mask_step is not None:
                m = frame_mask_step.reshape(b * t, f)
                lengths = m.sum(dim=1).clamp(min=1).long().cpu()
                y_packed = rnn.pack_padded_sequence(y, lengths, batch_first=True, enforce_sorted=False)
                _, h = self.gru(y_packed, self._gru_h)
                self._gru_h = h.detach() if self._gru_h is not None else h
            else:
                _, h = self.gru(y, self._gru_h)
                self._gru_h = h.detach() if self._gru_h is not None else h
            pooled = h[-1]  # (B, D)
            pooled = self.gru_proj(pooled)
        else:
            if frame_mask_step is not None:
                m = frame_mask_step.reshape(b * t, f).unsqueeze(-1).float()
                pooled = (y * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                pooled = y.mean(dim=1)
            if self.tcn_out_dim != self.out_dim:
                pooled = self.gru_proj(pooled.unsqueeze(1)).squeeze(1)

        pooled = pooled.reshape(b, 1, -1)  # (B, 1, D)
        pooled = pooled.transpose(1, 2)     # (B, D, 1)
        return pooled


class MultiScaleFrameEncoder(nn.Module):
    """
    Multi-scale feature extraction from frame sequence.
    Extracts features at different temporal scales and fuses them.
    Supports lookahead of multiple future layers for forward-looking decision making.
    """

    def __init__(
        self,
        in_channels: int = 768,       # 768 for DINOv3 ViT-B, 192 for hand-crafted grid
        out_channels: int = 128,       # wider to handle richer features
        kernel_size: int = 3,
        use_lookahead: bool = True,
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

        self.use_lookahead = use_lookahead
        self.lookahead_depth = 2  # number of future layers to look ahead
        if use_lookahead:
            self.lookahead_proj = nn.Conv1d(int(out_channels) * (1 + self.lookahead_depth), int(out_channels), kernel_size=1)

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
        if self.use_lookahead and self.lookahead_depth > 0:
            expected_in = fused.shape[1] * (1 + self.lookahead_depth)
            if self.lookahead_proj.in_channels != expected_in:
                self.lookahead_proj = nn.Conv1d(expected_in, fused.shape[1], kernel_size=1).to(fused.device)
            # Pad sequence dim (last dim) so that shift=k can safely read padded[:,:,k:k+t]
            padded = F.pad(fused, (0, self.lookahead_depth), mode="constant", value=0.0)  # (B, D, T+depth)
            shifted_list = [fused]
            for shift in range(1, self.lookahead_depth + 1):
                sliced = padded[:, :, shift:shift + t]  # (B, D, T)
                shifted_list.append(sliced)
            fused = torch.cat(shifted_list, dim=1)       # (B, (1+depth)*D, T)
            fused = self.lookahead_proj(fused)            # (B, D, T)
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
                use_lookahead=True,
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

        # Temporal decision head: per-step binary classification with context
        self.decision_head = TemporalDecisionHead(d_model=int(d_model), lookback=1, lookahead=2)

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
            dec = self.decision_head(z, logits, frame_mask)
            ret["decision_idx"] = dec["pred_idx"]           # (B,) — for compat
            ret["decision_probs"] = dec["decision_probs"]    # (B, T) — for BCE loss
        return ret

    # ------------------------------------------------------------------
    # Streaming inference support
    # ------------------------------------------------------------------

    def reset_hidden(self) -> None:
        """
        Reset all streaming state. Call this before starting a new well.
        Resets: frame encoder state, TCN/GRU hidden, transformer KV cache.
        """
        # Frame-level streaming state (GRU hidden)
        if hasattr(self.frame_encoder, "reset_hidden"):
            self.frame_encoder.reset_hidden()
        self._z_seq: list[torch.Tensor] = []
        self._logits_seq: list[torch.Tensor] = []
        self._kv_caches: list[list[torch.Tensor] | None] = [
            None for _ in self.transformer_layers
        ]
        self._step_count = 0
        self._lock_layers = 0

    def forward_step(
        self,
        x_step: torch.Tensor,
        frame_mask_step: torch.Tensor | None = None,
        lock_layers: int = 0,
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
        self._lock_layers = lock_layers

        if not hasattr(self, "_z_seq"):
            self.reset_hidden()
            self._lock_layers = lock_layers

        # Frame-level feature for this layer — use streaming method if available,
        # otherwise fall back to regular forward (safe for statelss encoders)
        if hasattr(self.frame_encoder, "forward_step"):
            frame_feat = self.frame_encoder.forward_step(x_step, frame_mask=frame_mask_step)
        else:
            frame_feat = self.frame_encoder(x_step, frame_mask=frame_mask_step)
        # frame_feat: (B, D, 1) per layer

        # TCN blocks: (B, D, 1) → (B, d_model, 1)
        for blk in self.layer_tcn:
            frame_feat = blk(frame_feat)

        # proj_in: (B, d_model, 1)
        frame_feat = self.proj_in(frame_feat)  # (B, d_model, 1)
        z_step = frame_feat.squeeze(2).unsqueeze(1)  # (B, 1, d_model)

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

        # ---- lock_layers: zero-out class=1 probability before softmax ----
        # Mirrors infer_simple.py line 222-223: logits[:, 1, :lock_layers] = -inf
        if self._lock_layers > 0:
            logits_full[:, 1, :self._lock_layers] = float("-inf")

        prob_step = F.softmax(logits_step, dim=1)[:, 1]  # (B, 1)
        prob_full = F.softmax(logits_full, dim=1)[:, 1]  # (B, t)

        self._z_seq.append(z_acc)
        self._logits_seq.append(logits_step)
        self._step_count += 1

        # decision_head._compute_decision_logits needs T >= 3 (prob_base[:, 2:] and prob_future padding)
        prob_full_T = logits_full.shape[2]
        if prob_full_T < 3:
            decision_idx = torch.zeros(b, device=z_acc.device, dtype=torch.long)
        else:
            decision_idx = self.decision_head(z_acc, logits_full, frame_mask=None)["pred_idx"]

        return {
            "logits_step": logits_step,
            "logits_full": logits_full,
            "prob_step": prob_step,
            "prob_full": prob_full,
            "decision_idx": decision_idx,
        }
