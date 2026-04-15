from __future__ import annotations

import numpy as np
import torch

# 前 SAFETY_LOCK_LAYERS 层强制认为未穿透（物理上不可能已打穿）
SAFETY_LOCK_LAYERS = 30

# S3WD 阈值
PROB_ACCEPT = 0.9
PROB_REJECT = 0.75
WAIT_CONSECUTIVE = 3


def apply_safety_lock(probs, lock_layers: int = SAFETY_LOCK_LAYERS):
    """将前 lock_layers 层的穿透概率强制置 0。"""
    if torch.is_tensor(probs):
        out = probs.clone()
        n = min(int(lock_layers), int(out.numel()))
        if n > 0:
            out[:n] = 0.0
        return out
    out = np.asarray(probs, dtype=np.float64).copy().ravel()
    n = min(int(lock_layers), int(out.size))
    if n > 0:
        out[:n] = 0.0
    return out


def s3wd_decide(
    probs,
    accept_thresh: float = PROB_ACCEPT,
    reject_thresh: float = PROB_REJECT,
    wait_consecutive: int = WAIT_CONSECUTIVE,
):
    """
    序贯三支决策 (S3WD)：
    - P_t >= accept_thresh：立即 Accept
    - reject_thresh < P_t < accept_thresh：进入 Wait，连续 wait_consecutive 次则 Accept
    - P_t <= reject_thresh：Reject，清零 Wait 计数
    """
    probs = np.asarray(probs).ravel()
    wait_count = 0
    for t, p in enumerate(probs.tolist()):
        p = float(p)
        if p >= float(accept_thresh):
            return True, int(t)
        if p > float(reject_thresh):
            wait_count += 1
            if wait_count >= int(wait_consecutive):
                return True, int(t - int(wait_consecutive) + 1)
        else:
            wait_count = 0
    return False, None


def topkmedian_decide(probs, k: int = 9, min_thresh: float = 0.4):
    """TopKMedian 决策：top-k 概率中位数过阈值则判穿透，层号取 top-k 索引中位数。"""
    probs = np.asarray(probs).ravel()
    t = int(probs.size)
    if t == 0:
        return False, None
    kk = int(max(1, min(int(k), t)))
    topk_idx = np.argsort(-probs)[:kk]
    topk_vals = probs[topk_idx]
    med_val = float(np.median(topk_vals))
    if med_val < float(min_thresh):
        return False, None
    topk_idx_sorted = np.sort(topk_idx)
    pred_idx = int(topk_idx_sorted[len(topk_idx_sorted) // 2])
    return True, pred_idx


def _uncertainty_topk_median(mean_probs: np.ndarray, var_probs: np.ndarray, k: int) -> float:
    mean_probs = np.asarray(mean_probs).ravel()
    var_probs = np.asarray(var_probs).ravel()
    t = int(mean_probs.size)
    if t == 0 or int(var_probs.size) != t:
        return 0.0
    kk = int(max(1, min(int(k), t)))
    topk_idx = np.argsort(-mean_probs)[:kk]
    return float(np.median(var_probs[topk_idx]))


def topkmedian_with_uncertainty_gate(
    mean_probs,
    var_probs,
    k: int = 9,
    min_thresh: float = 0.4,
    unc_var_median_thresh: float = 0.05,
):
    """
    先对 mean_probs 做 TopKMedian；若判穿透，再看 top-k 位置方差中位数：
    - med(var[top-k]) > unc_var_median_thresh 则否决穿透。
    """
    mean_probs = np.asarray(mean_probs).ravel()
    var_probs = np.asarray(var_probs).ravel() if var_probs is not None else None
    is_pen, pred_idx = topkmedian_decide(mean_probs, k=k, min_thresh=min_thresh)
    meta = {"unc_gated": False, "unc_topk_median": None}
    if (not is_pen) or var_probs is None or int(var_probs.size) != int(mean_probs.size):
        return is_pen, pred_idx, meta
    med_var = _uncertainty_topk_median(mean_probs, var_probs, k)
    meta["unc_topk_median"] = med_var
    if med_var > float(unc_var_median_thresh):
        meta["unc_gated"] = True
        return False, None, meta
    return is_pen, pred_idx, meta


__all__ = [
    "SAFETY_LOCK_LAYERS",
    "apply_safety_lock",
    "s3wd_decide",
    "topkmedian_decide",
    "topkmedian_with_uncertainty_gate",
]

