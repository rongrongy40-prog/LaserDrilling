#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TCN + Transformer 组合模型，以及带概率 Key/Value 的自注意力模块。

设计目标：
  - 输入保持与 GridDiffTCN 一致：(B, 64, T)，输出 (B, 2, T)。
  - 在若干 TCNBlock 之后接入 Transformer 编码器层，以更好地建模长程依赖。
  - 自注意力层中对 K/V 使用高斯分布参数化，并通过重参数化采样：
        K = mu_K + sigma_K * eps_K
        V = mu_V + sigma_V * eps_V
    从而在多次前向中体现概率注意力行为，可用于不确定性估计。

注意：
  - 为保持兼容性，forward 默认返回 logits；若需要 KL 正则，可选择返回 (logits, extra_dict)。
"""

from __future__ import annotations

from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# 与 train.py 的用法保持一致：脚本从 grid_diff_tcn 目录直接运行时，用同目录导入
from tcn_model import TCNBlock


class ProbabilisticSelfAttention(nn.Module):
    """
    概率自注意力（单头 or 多头），接口近似 nn.MultiheadAttention 的自注意力场景。

    - 对输入序列 x (B, T, D)，首先映射得到 query / key / value 的均值与 log 方差：
          mu_q, logvar_q, mu_k, logvar_k, mu_v, logvar_v
    - 对 K/V 使用重参数化：
          K = mu_k + sigma_k * eps_k
          V = mu_v + sigma_v * eps_v
      其中 sigma = exp(0.5 * logvar)，eps ~ N(0, I)。
    - 再使用标准缩放点积注意力：
          softmax(Q K^T / sqrt(d_k)) V
    - 可选 KL 正则：约束 (mu_v, sigma_v) 接近 N(0, I)，在外部以小权重加入损失。
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        add_kl: bool = True,
    ) -> None:
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.add_kl = add_kl

        # 对输入做一次线性映射，得到 q/k/v 对应的 mu 与 logvar
        proj_dim = embed_dim * 6  # q_mu, q_logvar, k_mu, k_logvar, v_mu, v_logvar
        self.qkv_proj = nn.Linear(embed_dim, proj_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, D) -> (B, num_heads, T, head_dim)
        B, T, D = x.shape
        x = x.view(B, T, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    @staticmethod
    def _kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        与 N(0, I) 的 KL 散度（逐元素），最后在 batch+time+dim 上求平均或和。
        KL = -0.5 * (1 + logvar - mu^2 - exp(logvar))
        """
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
        force_sample_kv: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        x: (B, T, D)
        返回:
          - out: (B, T, D)
          - extra: dict, 含 kl_loss 等辅助信息
        """
        B, T, D = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 6D)
        q_mu, q_logvar, k_mu, k_logvar, v_mu, v_logvar = torch.chunk(qkv, 6, dim=-1)

        # 当前实现中只对 K/V 做随机采样，Q 使用其均值即可。
        q = q_mu
        if self.training or force_sample_kv:
            # 重参数化采样 K/V（eval + force_sample_kv 时用于 MC 式不确定性）
            eps_k = torch.randn_like(k_mu)
            eps_v = torch.randn_like(v_mu)
            k = k_mu + torch.exp(0.5 * k_logvar) * eps_k
            v = v_mu + torch.exp(0.5 * v_logvar) * eps_v
        else:
            k = k_mu
            v = v_mu

        # 形状整理到多头
        qh = self._reshape_heads(q)  # (B, H, T, Dh)
        kh = self._reshape_heads(k)
        vh = self._reshape_heads(v)

        # 注意力权重
        attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # (B, H, T, T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 注意力输出
        out = torch.matmul(attn_weights, vh)  # (B, H, T, Dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, D)
        out = self.out_proj(out)

        extra: Dict[str, Any] = {}
        if self.add_kl:
            # 针对 V 的分布计算 KL 正则，按 batch/time 取平均
            kl = self._kl_standard_normal(v_mu, v_logvar)
            extra["kl_loss"] = kl.mean()
        if need_weights:
            extra["attn_weights"] = attn_weights  # (B, H, T, T)
        return out, extra


class ProbTransformerEncoderLayer(nn.Module):
    """
    基于概率自注意力的 Transformer Encoder Layer。
    结构：Norm -> ProbAttn -> Dropout -> Residual -> Norm -> FFN -> Dropout -> Residual。
    """

    def __init__(
        self,
        d_model: int,
        nhead: int = 4,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        add_kl: bool = True,
    ) -> None:
        super().__init__()
        self.self_attn = ProbabilisticSelfAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            add_kl=add_kl,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, force_sample_kv: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        src: (B, T, D)
        返回:
          - out: (B, T, D)
          - extra: dict, 累积 kl_loss 等
        """
        # Self Attention + 残差
        src2, extra_attn = self.self_attn(self.norm1(src), need_weights=False, force_sample_kv=force_sample_kv)
        src = src + self.dropout1(src2)

        # FFN + 残差
        ff = self.linear2(self.dropout2(self.activation(self.linear1(self.norm2(src)))))
        src = src + ff

        return src, extra_attn


class GridDiffTCNWithTransformer(nn.Module):
    """
    在若干 TCNBlock 之后接入概率 Transformer 编码器的模型。

    - 输入: (B, 64, T)
    - 输出: (B, 2, T)
    - 兼容原有 GridDiffTCN 接口，便于替换使用。
    """

    def __init__(
        self,
        in_channels: int = 64,
        out_channels: int = 2,
        tcn_channels: Tuple[int, ...] = (64, 64),
        kernel_size: int = 3,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        add_kl: bool = True,
        return_kl: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.return_kl = return_kl

        # TCN 前端
        tcn_layers = []
        ch_in = in_channels
        for i, ch_out in enumerate(tcn_channels):
            dilation = 2 ** i
            tcn_layers.append(TCNBlock(ch_in, ch_out, kernel_size, dilation))
            ch_in = ch_out
        self.tcn_layers = nn.ModuleList(tcn_layers)

        # Transformer 编码器堆叠
        self.d_model = d_model
        if ch_in != d_model:
            self.proj_in = nn.Conv1d(ch_in, d_model, kernel_size=1)
        else:
            self.proj_in = nn.Identity()

        layers = []
        for _ in range(num_layers):
            layers.append(
                ProbTransformerEncoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dim_feedforward=dim_feedforward,
                    dropout=dropout,
                    add_kl=add_kl,
                )
            )
        self.transformer_layers = nn.ModuleList(layers)

        # 输出头：回到通道维度后接 Conv1d 映射到 2 类
        self.head = nn.Conv1d(d_model, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, force_sample_attention: bool = False):
        """
        x: (B, 64, T)
        force_sample_attention: True 时在 eval 下仍对 K/V 采样，供多次前向估计不确定性。
        返回:
          - logits: (B, 2, T)
          - （可选）extra: {"kl_loss": tensor}
        """
        # TCN 前端
        for block in self.tcn_layers:
            x = block(x)  # (B, C_mid, T)

        # 通道维 -> d_model，并转为 (B, T, D)
        x = self.proj_in(x)  # (B, d_model, T)
        x = x.transpose(1, 2)  # (B, T, D)

        kl_terms = []
        for layer in self.transformer_layers:
            x, extra = layer(x, force_sample_kv=force_sample_attention)
            if "kl_loss" in extra:
                kl_terms.append(extra["kl_loss"])

        x = x.transpose(1, 2)  # (B, D, T)
        logits = self.head(x)  # (B, 2, T)

        if self.return_kl and kl_terms:
            kl_loss = torch.stack(kl_terms).mean()
            return logits, {"kl_loss": kl_loss}
        return logits


def build_tcn_or_transformer(
    use_transformer: bool,
    **kwargs: Any,
) -> nn.Module:
    """
    工厂函数：根据 use_transformer 标志构建原始 GridDiffTCN 或带 Transformer 的版本。
    仅在 train / inference 脚本内部使用，避免到处写分支。
    """
    from tcn_model import GridDiffTCN

    if not use_transformer:
        return GridDiffTCN(
            in_channels=kwargs.get("in_channels", 64),
            out_channels=kwargs.get("out_channels", 2),
            num_channels=kwargs.get("tcn_channels", (64, 64, 64, 64)),
            kernel_size=kwargs.get("kernel_size", 3),
        )
    return GridDiffTCNWithTransformer(
        in_channels=kwargs.get("in_channels", 64),
        out_channels=kwargs.get("out_channels", 2),
        tcn_channels=kwargs.get("tcn_channels", (64, 64)),
        kernel_size=kwargs.get("kernel_size", 3),
        d_model=kwargs.get("d_model", 64),
        nhead=kwargs.get("nhead", 4),
        num_layers=kwargs.get("num_layers", 2),
        dim_feedforward=kwargs.get("dim_feedforward", 256),
        dropout=kwargs.get("dropout", 0.1),
        add_kl=kwargs.get("add_kl", True),
        return_kl=kwargs.get("return_kl", False),
    )

