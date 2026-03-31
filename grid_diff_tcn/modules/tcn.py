from __future__ import annotations

import torch
import torch.nn as nn


class TCNBlock(nn.Module):
    """
    单层 TCN 块：因果膨胀卷积 + BatchNorm1d + ReLU + 残差连接。
    输入/输出：x shape (B, C, T)
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int) -> None:
        super().__init__()
        self.pad_left = (int(kernel_size) - 1) * int(dilation)
        self.conv = nn.Conv1d(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            padding=int(self.pad_left),
            dilation=int(dilation),
        )
        self.bn = nn.BatchNorm1d(int(out_channels))
        self.act = nn.ReLU(inplace=True)
        self.residual = nn.Conv1d(int(in_channels), int(out_channels), 1) if int(in_channels) != int(out_channels) else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.pad_left > 0:
            out = out[:, :, self.pad_left :]
        out = self.bn(out)
        out = self.act(out)
        res = self.residual(x)
        return out + res

