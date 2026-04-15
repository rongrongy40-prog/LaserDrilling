"""
仅保留新版 `hier` 需要的“决策函数”入口。

旧版窗口模型（GridDiffTCN / two-tower / transformer_tcn 等）推理脚本已移除，以便彻底删除 `legacy/`。
如需决策函数，请直接从 `grid_diff_tcn.hier.decision` 导入。
"""

from grid_diff_tcn.common.decision import (  # noqa: F401
    SAFETY_LOCK_LAYERS,
    apply_safety_lock,
    s3wd_decide,
    topkmedian_decide,
    topkmedian_with_uncertainty_gate,
)

__all__ = [
    "SAFETY_LOCK_LAYERS",
    "apply_safety_lock",
    "s3wd_decide",
    "topkmedian_decide",
    "topkmedian_with_uncertainty_gate",
]
