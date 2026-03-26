# -*- coding: utf-8 -*-
"""
模块二：轻量级因果时序卷积网络 (Model 模块)。

功能：定义 GridDiffTCN 模型，对单孔 [Seq_Len, 64] 网格差分序列做逐层二分类（未穿透/穿透）。
      采用因果膨胀卷积，保证时刻 t 仅依赖 t 及之前，无未来泄漏。
依赖：torch
输入形状: (Batch, 64, Seq_Len)，即 (B, in_channels, T)。
输出形状: (Batch, 2, Seq_Len)，每时间步 2 维 logits（未做 Softmax，供 CrossEntropyLoss 使用）。
用法：from tcn_model import GridDiffTCN
      model = GridDiffTCN(in_channels=64, out_channels=2, num_channels=(64,64,64,64), kernel_size=3)
"""

import torch
import torch.nn as nn
import math


class CausalConv1d(nn.Module):
    """
    因果一维卷积：保证输出 t 时刻只依赖 t 及之前时刻，不泄漏未来信息。
    通过 left padding (kernel_size - 1) 实现。
    """

    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.padding = (kernel_size - 1)
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding)

    def forward(self, x):
        # x: (B, C_in, T)
        out = self.conv(x)
        # 去掉右侧 padding 对应的输出，保持因果性
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class TCNBlock(nn.Module):
    """
    单层 TCN 块：因果膨胀卷积 + BatchNorm1d + ReLU + 残差连接。
    若 in_ch != out_ch，用 1x1 卷积对齐通道再相加。
    """

    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        # 因果膨胀卷积：左侧 padding，输出后裁掉右侧多出的部分
        self.pad_left = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.pad_left,
            dilation=dilation
        )
        self.bn = nn.BatchNorm1d(out_channels)
        self.act = nn.ReLU(inplace=True)

        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, 1)
        else:
            self.residual = nn.Identity()

    def forward(self, x):
        # x: (B, C_in, T) -> 因果：只保留与输入长度相同的尾部
        out = self.conv(x)
        if self.pad_left > 0:
            out = out[:, :, self.pad_left:]
        out = self.bn(out)
        out = self.act(out)
        res = self.residual(x)
        return out + res


class GridDiffTCN(nn.Module):
    """
    基于 8x8 网格差分输入的 64 维时序卷积网络。

    结构概览：
      - 输入: (B, 64, T)
      - 多层 CausalConv1d + BN + ReLU + Residual，膨胀因子递增
      - 最后一层映射到 2 通道（未穿透/穿透）logits
      - 不在网络内做 Softmax，由损失函数或推理时再算概率
    """

    def __init__(
        self,
        in_channels=64,
        out_channels=2,
        num_channels=(64, 64, 64, 64),
        kernel_size=3,
    ):
        """
        in_channels: 输入通道数，对应 8x8 网格的 64 维
        out_channels: 输出类别数，2（未穿透=0，穿透=1）
        num_channels: 每层隐藏通道数，如 (64,64,64,64) 表示 4 层
        kernel_size: 每层卷积核大小
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_levels = len(num_channels)

        layers = []
        channel_in = in_channels
        for i, channel_out in enumerate(num_channels):
            dilation = 2 ** i  # 1, 2, 4, 8 ...
            layers.append(
                TCNBlock(channel_in, channel_out, kernel_size, dilation)
            )
            channel_in = channel_out
        self.tcn_layers = nn.ModuleList(layers)

        # 最后一层：将最后一层 TCN 输出映射到 2 类 logits
        self.head = nn.Conv1d(channel_in, out_channels, 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B, 64, T)
        返回: (B, 2, T)，即每个时间步的 2 维 logits
        """
        for block in self.tcn_layers:
            x = block(x)
        logits = self.head(x)
        return logits


def get_logits_per_layer(model, x):
    """
    前向得到每层（时间步）的 logits。
    x: (B, 64, T) -> logits: (B, 2, T)
    若需要每层概率，在外部对 logits 做 softmax 取第 1 维（穿透类）。
    """
    return model(x)
