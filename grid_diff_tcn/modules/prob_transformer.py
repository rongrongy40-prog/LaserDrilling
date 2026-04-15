from __future__ import annotations

from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProbabilisticSelfAttention(nn.Module):
    """
    概率自注意力：对 K/V 使用高斯分布参数化并重参数化采样。
    输入/输出：x shape (B, T, D)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.0,
        add_kl: bool = True,
    ) -> None:
        super().__init__()
        assert int(embed_dim) % int(num_heads) == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = self.head_dim**-0.5
        self.add_kl = bool(add_kl)

        proj_dim = self.embed_dim * 6  # q_mu, q_logvar, k_mu, k_logvar, v_mu, v_logvar
        self.qkv_proj = nn.Linear(self.embed_dim, proj_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.dropout = nn.Dropout(float(dropout))

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        x = x.view(b, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)  # (B,H,T,Dh)

    @staticmethod
    def _kl_standard_normal(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())

    def forward(
        self,
        x: torch.Tensor,
        need_weights: bool = False,
        force_sample_kv: bool = False,
    ) -> Tuple[torch.Tensor, Dict[str, Any]]:
        b, t, d = x.shape
        qkv = self.qkv_proj(x)  # (B,T,6D)
        q_mu, q_logvar, k_mu, k_logvar, v_mu, v_logvar = torch.chunk(qkv, 6, dim=-1)

        q = q_mu
        if self.training or bool(force_sample_kv):
            eps_k = torch.randn_like(k_mu)
            eps_v = torch.randn_like(v_mu)
            k = k_mu + torch.exp(0.5 * k_logvar) * eps_k
            v = v_mu + torch.exp(0.5 * v_logvar) * eps_v
        else:
            k = k_mu
            v = v_mu

        qh = self._reshape_heads(q)
        kh = self._reshape_heads(k)
        vh = self._reshape_heads(v)

        attn_scores = torch.matmul(qh, kh.transpose(-2, -1)) * self.scale  # (B,H,T,T)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, vh)  # (B,H,T,Dh)
        out = out.permute(0, 2, 1, 3).contiguous().view(b, t, d)
        out = self.out_proj(out)

        extra: Dict[str, Any] = {}
        if self.add_kl:
            kl_k = self._kl_standard_normal(k_mu, k_logvar).mean()
            kl_v = self._kl_standard_normal(v_mu, v_logvar).mean()
            extra["kl_loss"] = 0.5 * (kl_k + kl_v)
        if need_weights:
            extra["attn_weights"] = attn_weights
        return out, extra


class ProbTransformerEncoderLayer(nn.Module):
    """
    基于概率自注意力的 Transformer Encoder Layer。
    输入/输出：src shape (B, T, D)
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
            embed_dim=int(d_model),
            num_heads=int(nhead),
            dropout=float(dropout),
            add_kl=bool(add_kl),
        )
        self.linear1 = nn.Linear(int(d_model), int(dim_feedforward))
        self.linear2 = nn.Linear(int(dim_feedforward), int(d_model))
        self.norm1 = nn.LayerNorm(int(d_model))
        self.norm2 = nn.LayerNorm(int(d_model))
        self.dropout1 = nn.Dropout(float(dropout))
        self.dropout2 = nn.Dropout(float(dropout))
        self.activation = nn.GELU()

    def forward(self, src: torch.Tensor, force_sample_kv: bool = False) -> Tuple[torch.Tensor, Dict[str, Any]]:
        src2, extra_attn = self.self_attn(self.norm1(src), need_weights=False, force_sample_kv=force_sample_kv)
        src = src + self.dropout1(src2)
        ff = self.linear2(self.dropout2(self.activation(self.linear1(self.norm2(src)))))
        src = src + ff
        return src, extra_attn

