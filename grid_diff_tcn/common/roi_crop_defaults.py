# -*- coding: utf-8 -*-
"""
与 hier/visualize_roi.py 的裁剪参数保持一致，供 precompute/train/infer/dataset 共用。
"""

from __future__ import annotations

# load_image_as_float(..., target_size=(W, H))
DEFAULT_TARGET_WH: tuple[int, int] = (128, 128)
DEFAULT_ROI_SIZE: int = 96
DEFAULT_MAX_FRAMES_PER_LAYER: int = 8
DEFAULT_FINAL_ROI_SCALE: float = 0.85
DEFAULT_CC_MIN_AREA: int = 12
DEFAULT_CC_EXPAND_RATIO: float = 0.2
DEFAULT_MIN_LASER_PIXELS: int = 0
DEFAULT_MIN_LASER_AREA_RATIO: float = 0.0
DEFAULT_ROI_BRIGHT_MIN_RATIO: float = 0.0
DEFAULT_ROI_GRAY_P95_MIN: float = 0.0
DEFAULT_USE_COLOR_CC_V2_GEOMETRY: bool = True
# 与 visualize_roi：--roi_window_side 默认 32；<=0 表示关闭（随检测框大小变化）
DEFAULT_ROI_WINDOW_SIDE: int = 32


def norm_roi_window_side(v: int | None) -> int | None:
    """
    None → 使用 DEFAULT_ROI_WINDOW_SIDE；
    <=0 → None（不强制正方形窗口）；
    >0 → 固定正方形边长。
    """
    if v is None:
        v = DEFAULT_ROI_WINDOW_SIDE
    iv = int(v)
    return None if iv <= 0 else iv
