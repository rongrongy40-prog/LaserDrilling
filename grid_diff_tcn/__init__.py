# -*- coding: utf-8 -*-
"""
grid_diff_tcn 包入口。

功能：提供 Grid-Diff 1D-TCN 穿透检测的核心类，供本目录下 train / inference 等脚本引用。
主要导出：GridDiffDrillingDataset（单孔/整序列数据加载与 8×8 网格差分特征）、
         collate_fn（batch 拼接）、GridDiffTCN（因果 TCN 模型）。
"""

from .dataset import GridDiffDrillingDataset, collate_fn
from .tcn_model import GridDiffTCN

__all__ = ["GridDiffDrillingDataset", "collate_fn", "GridDiffTCN"]
