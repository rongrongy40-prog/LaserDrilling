# -*- coding: utf-8 -*-
"""
Hierarchical training script (frame-level + layer-level).

Features:
- Uses hier/frame_layer dataset and model.
- Keeps probabilistic transformer in layer-level modeling.
- Validation metrics aligned with current pipeline:
  <=3 / <=5 / >10 layer error on penetrated holes.
"""

import os
import json
import math
from glob import glob
import random
import argparse
from typing import Dict, Any, Optional, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRID_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

from grid_diff_tcn.common.roi_crop_defaults import DEFAULT_ROI_WINDOW_SIDE
from grid_diff_tcn.hier.frame_layer import (
    HierarchicalFrameLayerDataset,
    collate_hierarchical_batch,
    HierarchicalGridDiffProbTransformer,
    DinoV3FeatureExtractor,
    HierarchicalDinoV3Dataset,
    DINOV3_MODELS,
    DINOV3_DEFAULT_MODEL,
    DINOV3_FEAT_DIMS,
)
from grid_diff_tcn.common.decision import (
    apply_safety_lock,
    s3wd_decide,
    topkmedian_decide,
    topkmedian_with_uncertainty_gate,
)


def focal_cross_entropy(logits_bt, seq_y_bt, mask_bt, gamma=2.0, alpha_pos=0.75):
    seq_y_bt = seq_y_bt.clamp(0, 1)
    valid_logits = logits_bt[mask_bt]
    valid_y = seq_y_bt[mask_bt]
    probs = F.softmax(valid_logits, dim=1)
    pt = probs.gather(1, valid_y.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    alpha_t = torch.where(
        valid_y == 1,
        torch.tensor(alpha_pos, device=logits_bt.device, dtype=logits_bt.dtype),
        torch.tensor(1.0 - alpha_pos, device=logits_bt.device, dtype=logits_bt.dtype),
    )
    focal_w = (1 - pt).pow(gamma)
    step_loss = focal_w * (-pt.log()) * alpha_t
    full_loss = torch.zeros(mask_bt.size(0), device=logits_bt.device, dtype=step_loss.dtype)
    full_loss[mask_bt] = step_loss
    return full_loss


def aligned_losses(logits: torch.Tensor, seq_y: torch.Tensor):
    """
    logits: (B,2,T), seq_y: (B,T)
    """
    b, _, t = logits.shape
    has_pen = (seq_y.sum(dim=1) > 0)
    if not has_pen.any():
        return None, None

    ar = torch.arange(t, device=logits.device, dtype=logits.dtype)
    prob_t = F.softmax(logits[:, 1, :], dim=1)
    true_layer = seq_y.argmax(dim=1).float()

    pred_soft = (prob_t * ar.unsqueeze(0)).sum(dim=1)
    dist = (pred_soft - true_layer).abs()
    loss_loc5 = (F.relu(dist - 5.0) ** 2)[has_pen].mean()

    true_idx = seq_y.argmax(dim=1)
    idx = torch.arange(t, device=logits.device).unsqueeze(0).expand(b, -1)
    band_mask = (idx - true_idx.unsqueeze(1)).abs() <= 5
    p_within5 = (prob_t * band_mask.to(prob_t.dtype)).sum(dim=1).clamp(min=1e-8)
    loss_within5 = (-torch.log(p_within5[has_pen])).mean()
    return loss_loc5, loss_within5


def window_ce_weights(seq_y: torch.Tensor, window_radius: int = 5, in_window_weight: float = 2.0) -> torch.Tensor:
    """
    给穿透孔的序列位置加权：真值层±window_radius 的 timestep 权重更高，促进 <=5。
    返回: (B,T) float
    """
    b, t = seq_y.shape
    w = torch.ones((b, t), device=seq_y.device, dtype=torch.float32)
    has_pen = (seq_y.sum(dim=1) > 0)
    if not has_pen.any():
        return w
    true_idx = seq_y.argmax(dim=1)  # (B,)
    idx = torch.arange(t, device=seq_y.device).unsqueeze(0).expand(b, -1)
    band = (idx - true_idx.unsqueeze(1)).abs() <= int(window_radius)
    w = torch.where(has_pen.unsqueeze(1) & band, torch.full_like(w, float(in_window_weight)), w)
    return w


def compute_hole_metrics(records: List[Dict[str, Any]]) -> Dict[str, float]:
    n_pen = 0
    n3 = n5 = n10 = 0
    for r in records:
        if int(r.get("true_label", 0)) != 1:
            continue
        n_pen += 1
        true_idx = r.get("true_penetration_index")
        pred_idx = r.get("pred_penetration_index")
        if true_idx is None or pred_idx is None:
            n10 += 1
            continue
        e = abs(int(pred_idx) - int(true_idx))
        if e <= 3:
            n3 += 1
        if e <= 5:
            n5 += 1
        if e > 10:
            n10 += 1
    return {
        "n_penetrated": n_pen,
        "pct_within_3": (n3 / n_pen * 100.0) if n_pen else 0.0,
        "pct_within_5": (n5 / n_pen * 100.0) if n_pen else 0.0,
        "pct_over_10": (n10 / n_pen * 100.0) if n_pen else 0.0,
    }


def grid_search_s3wd(
    probs_all: List[np.ndarray],
    layer_lists: List[List[int]],
    labels: List[int],
    pen_layers: List[int],
    lock_layers: int,
    accept_candidates: List[float],
    reject_candidates: List[float],
    wait_candidates: List[int],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    网格搜索 S3WD 阈值参数。

    返回: {
        "best": {accept, reject, wait, metrics},
        "grid": [{{accept, reject, wait, metrics}, ...}],
    }
    """
    best_score = -1.0
    best_params = {}
    best_metrics = {}
    grid_results = []

    total = len(accept_candidates) * len(reject_candidates) * len(wait_candidates)
    print(f"[grid_search_s3wd] 开始网格搜索，共 {total} 种组合 ...")

    for acc in accept_candidates:
        for rej in reject_candidates:
            if rej >= acc:
                continue
            for wait in wait_candidates:
                recs = []
                for p, layer_list, label, pen_layer in zip(
                    probs_all, layer_lists, labels, pen_layers
                ):
                    p_safe = apply_safety_lock(p, lock_layers=lock_layers)
                    pred_pen, pred_idx = s3wd_decide(
                        p_safe,
                        accept_thresh=float(acc),
                        reject_thresh=float(rej),
                        wait_consecutive=int(wait),
                    )
                    true_label = int(label)
                    true_layer = int(pen_layer)
                    true_idx = layer_list.index(true_layer) if (true_label == 1 and true_layer in layer_list) else None
                    recs.append(
                        {
                            "true_label": true_label,
                            "true_penetration_index": true_idx,
                            "pred_penetration_index": pred_idx if pred_pen else None,
                        }
                    )
                met = compute_hole_metrics(recs)
                score = met["pct_within_5"]
                entry = {
                    "accept": float(acc),
                    "reject": float(rej),
                    "wait": int(wait),
                    "metrics": met,
                }
                grid_results.append(entry)
                if score > best_score:
                    best_score = score
                    best_params = {"accept": float(acc), "reject": float(rej), "wait": int(wait)}
                    best_metrics = met

    grid_results.sort(key=lambda x: x["metrics"]["pct_within_5"], reverse=True)

    if verbose:
        print(f"[grid_search_s3wd] 最优: accept={best_params['accept']}, reject={best_params['reject']}, "
              f"wait={best_params['wait']}, <=5={best_score:.1f}% "
              f"(<=3:{best_metrics['pct_within_3']:.1f}%, >10:{best_metrics['pct_over_10']:.1f}%)")
        print("[grid_search_s3wd] Top-5 组合:")
        for r in grid_results[:5]:
            m = r["metrics"]
            print(f"  accept={r['accept']:.2f} reject={r['reject']:.2f} wait={r['wait']}  "
                  f"<=3:{m['pct_within_3']:.1f}% <=5:{m['pct_within_5']:.1f}% >10:{m['pct_over_10']:.1f}%")

    return {"best": {"params": best_params, "metrics": best_metrics}, "grid": grid_results}


def run_validation(
    model,
    loader,
    device,
    lock_layers: int,
    decision_method: str,
    topk_k: int = 9,
    topk_min_thresh: float = 0.3,
    s3wd_accept_thresh: float = 0.9,
    s3wd_reject_thresh: float = 0.75,
    s3wd_wait_consecutive: int = 3,
    unc_samples: int = 1,
    use_uncertainty_gate: bool = False,
    unc_var_median_thresh: float = 0.05,
    use_amp: bool = False,
) -> tuple[Dict[str, float], Dict[str, Any]]:
    """
    返回 (metrics, raw_data)，其中 raw_data 包含:
      probs_all, layer_lists, labels, pen_layers
    用于后续网格搜索。
    """
    model.eval()
    recs = []
    probs_all: List[np.ndarray] = []
    layer_lists: List[List[int]] = []
    labels_all: List[int] = []
    pen_layers_all: List[int] = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Val(hole)", leave=False):
            x = batch["frame_data"].to(device, non_blocking=True)
            fm = batch["frame_mask"].to(device, non_blocking=True)
            lx = batch.get("layer_extra")
            lx = lx.to(device, non_blocking=True) if torch.is_tensor(lx) else None
            layer_mask = batch["layer_mask"].to(device, non_blocking=True)

            if use_amp and device.type == "cuda":
                with torch.cuda.amp.autocast():
                    if int(unc_samples) > 1:
                        runs = []
                        for _ in range(int(unc_samples)):
                            out = model(x, frame_mask=fm, force_sample_attention=True, layer_extra=lx)
                            logits = out[0] if isinstance(out, tuple) else out
                            runs.append(F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy())
                        runs = np.stack(runs, axis=0)
                        probs = runs.mean(axis=0)
                    else:
                        out = model(x, frame_mask=fm, layer_extra=lx)
                        logits = out[0] if isinstance(out, tuple) else out
                        probs = F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy()
            else:
                if int(unc_samples) > 1:
                    runs = []
                    for _ in range(int(unc_samples)):
                        out = model(x, frame_mask=fm, force_sample_attention=True, layer_extra=lx)
                        logits = out[0] if isinstance(out, tuple) else out
                        runs.append(F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy())
                    runs = np.stack(runs, axis=0)
                    probs = runs.mean(axis=0)
                else:
                    out = model(x, frame_mask=fm, layer_extra=lx)
                    logits = out[0] if isinstance(out, tuple) else out
                    probs = F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy()

            batch_labels = [int(v.item()) for v in batch["label"]]
            batch_pen_layers = [int(v.item()) for v in batch["penetration_layer"]]
            batch_layer_lists = batch["layer_list"]

            for i in range(probs.shape[0]):
                valid_t = int(layer_mask[i].sum().item())
                p_raw = np.asarray(probs[i, :valid_t], dtype=np.float64)
                p_safe = apply_safety_lock(p_raw, lock_layers=lock_layers)

                # 收集原始数据（用于网格搜索）
                probs_all.append(p_safe)
                layer_lists.append(batch_layer_lists[i])
                labels_all.append(batch_labels[i])
                pen_layers_all.append(batch_pen_layers[i])

                # 当前阈值下的决策（用于训练阶段选模型）
                if decision_method == "s3wd":
                    pred_pen, pred_idx = s3wd_decide(
                        p_safe,
                        accept_thresh=float(s3wd_accept_thresh),
                        reject_thresh=float(s3wd_reject_thresh),
                        wait_consecutive=int(s3wd_wait_consecutive),
                    )
                else:
                    if use_uncertainty_gate:
                        # 方差门控暂不支持网格搜索，直接跳过
                        pred_pen, pred_idx = topkmedian_decide(p_safe, k=topk_k, min_thresh=topk_min_thresh)
                    else:
                        pred_pen, pred_idx = topkmedian_decide(p_safe, k=topk_k, min_thresh=topk_min_thresh)

                true_label = batch_labels[i]
                true_layer = batch_pen_layers[i]
                true_idx = batch_layer_lists[i].index(true_layer) if (true_label == 1 and true_layer in batch_layer_lists[i]) else None
                recs.append(
                    {
                        "true_label": true_label,
                        "true_penetration_index": true_idx,
                        "pred_penetration_index": pred_idx if pred_pen else None,
                    }
                )

    metrics = compute_hole_metrics(recs)
    raw_data = {
        "probs_all": probs_all,
        "layer_lists": layer_lists,
        "labels": labels_all,
        "pen_layers": pen_layers_all,
    }
    return metrics, raw_data


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    use_focal=True,
    focal_gamma=2.0,
    focal_alpha=0.75,
    weight_pos=10.0,
    loc5_weight=0.3,
    within5_weight=0.7,
    window_radius: int = 5,
    in_window_weight: float = 2.0,
    kl_weight=0.0,
    use_amp=False,
    scaler=None,
):
    model.train()
    total_loss = 0.0
    total_steps = 0
    for batch in tqdm(loader, desc="Train", leave=False):
        x = batch["frame_data"].to(device, non_blocking=True)  # (B,T,F,C)
        fm = batch["frame_mask"].to(device, non_blocking=True)
        lx = batch.get("layer_extra")
        lx = lx.to(device, non_blocking=True) if torch.is_tensor(lx) else None
        seq_y = batch["seq_label"].to(device, non_blocking=True)  # (B,T)
        layer_mask = batch["layer_mask"].to(device, non_blocking=True)  # (B,T)
        if not layer_mask.any() or not fm.any():
            continue
        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda" and scaler is not None:
            with torch.cuda.amp.autocast():
                out = model(x, frame_mask=fm, layer_extra=lx)
                if isinstance(out, tuple):
                    logits, extra = out
                else:
                    logits, extra = out, {}
                logits_bt = logits.permute(0, 2, 1).reshape(-1, 2)
                y_bt = seq_y.reshape(-1)
                m_bt = layer_mask.reshape(-1)
                if use_focal:
                    step_loss = focal_cross_entropy(logits_bt, y_bt, m_bt, gamma=focal_gamma, alpha_pos=focal_alpha)
                else:
                    w = torch.where(
                        y_bt == 1,
                        torch.tensor(weight_pos, device=device, dtype=logits.dtype),
                        torch.tensor(1.0, device=device, dtype=logits.dtype),
                    )
                    step_loss = F.cross_entropy(logits_bt, y_bt, reduction="none") * w
                # timestep weights (emphasize +/- window around true penetration for penetrated holes)
                w_bt = window_ce_weights(seq_y, window_radius=window_radius, in_window_weight=in_window_weight).to(device)
                w_bt = w_bt.reshape(-1).to(dtype=step_loss.dtype)
                base = (step_loss * w_bt)[m_bt].mean()
                loss = base
                l_loc5, l_w5 = aligned_losses(logits, seq_y)
                if l_loc5 is not None and loc5_weight > 0:
                    loss = loss + float(loc5_weight) * l_loc5
                if l_w5 is not None and within5_weight > 0:
                    loss = loss + float(within5_weight) * l_w5
                if kl_weight > 0.0 and isinstance(extra, dict) and "kl_loss" in extra:
                    loss = loss + float(kl_weight) * extra["kl_loss"]
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(x, frame_mask=fm, layer_extra=lx)
            if isinstance(out, tuple):
                logits, extra = out
            else:
                logits, extra = out, {}
            logits_bt = logits.permute(0, 2, 1).reshape(-1, 2)
            y_bt = seq_y.reshape(-1)
            m_bt = layer_mask.reshape(-1)
            if use_focal:
                step_loss = focal_cross_entropy(logits_bt, y_bt, m_bt, gamma=focal_gamma, alpha_pos=focal_alpha)
            else:
                w = torch.where(
                    y_bt == 1,
                    torch.tensor(weight_pos, device=device, dtype=logits.dtype),
                    torch.tensor(1.0, device=device, dtype=logits.dtype),
                )
                step_loss = F.cross_entropy(logits_bt, y_bt, reduction="none") * w
            w_bt = window_ce_weights(seq_y, window_radius=window_radius, in_window_weight=in_window_weight).to(device)
            w_bt = w_bt.reshape(-1).to(dtype=step_loss.dtype)
            base = (step_loss * w_bt)[m_bt].mean()
            loss = base
            l_loc5, l_w5 = aligned_losses(logits, seq_y)
            if l_loc5 is not None and loc5_weight > 0:
                loss = loss + float(loc5_weight) * l_loc5
            if l_w5 is not None and within5_weight > 0:
                loss = loss + float(within5_weight) * l_w5
            if kl_weight > 0.0 and isinstance(extra, dict) and "kl_loss" in extra:
                loss = loss + float(kl_weight) * extra["kl_loss"]
            loss.backward()
            optimizer.step()

        total_loss += float(loss.item())
        total_steps += 1
    return total_loss / max(1, total_steps)


def main():
    ap = argparse.ArgumentParser(description="Train hierarchical frame-layer model")
    ap.add_argument("--samples_info", type=str, default=None)
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--save", type=str, default=os.path.join(_GRID_ROOT, "grid_diff_tcn_hierarchical.pt"))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--val_ratio", type=float, default=0.2)
    ap.add_argument("--val_seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--roi_size", type=int, default=96)
    ap.add_argument("--max_frames_per_layer", type=int, default=8)
    ap.add_argument("--max_layers", type=int, default=None)
    ap.add_argument("--penetration_radius", type=int, default=2)
    ap.add_argument("--precomputed_dir", type=str, default=None, help="hierarchical precomputed .pt directory")

    ap.add_argument("--cc_min_area", type=int, default=12)
    ap.add_argument("--cc_expand_ratio", type=float, default=0.2)
    ap.add_argument("--final_roi_scale", type=float, default=0.85)
    ap.add_argument("--exclude_json", type=str, default=os.path.join(_REPO_ROOT, "data_drilling", "no_laser_change_equalbox_full_mad00005_center_and_below.json"))
    ap.add_argument("--min_laser_pixels", type=int, default=0, help="全图 HSV 亮区像素下限；0=关闭")
    ap.add_argument("--min_laser_area_ratio", type=float, default=0.0, help="全图亮区面积比下限；0=关闭")
    ap.add_argument(
        "--roi_window_side",
        type=int,
        default=DEFAULT_ROI_WINDOW_SIDE,
        help="与 visualize_roi/precompute 一致：固定正方形边长；0=关闭",
    )
    ap.add_argument("--roi_bright_min_ratio", type=float, default=0.0, help="letterbox 后 ROI 亮区占比下限；0=关闭")
    ap.add_argument("--roi_gray_p95_min", type=float, default=0.0, help="letterbox 后灰度 p95 下限；0=关闭")
    ap.add_argument(
        "--legacy_color_cc_geometry",
        action="store_true",
        help="使用旧 shrink(min边) 几何",
    )
    ap.add_argument("--frame_channels", type=str, default="192,192")
    ap.add_argument("--layer_tcn_channels", type=str, default="64,64")
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_transformer_layers", type=int, default=2)
    ap.add_argument("--dim_feedforward", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--kl_weight", type=float, default=0.0)
    ap.add_argument("--extra_dim", type=int, default=0, help="在线 layer_extra 特征维度；0=禁用融合")
    ap.add_argument("--use_frame_gru", action="store_true", default=True, help="帧级使用GRU编码")
    ap.add_argument("--no_frame_gru", action="store_true", help="禁用帧级GRU")
    ap.add_argument("--use_frame_attn_pool", action="store_true", default=True, help="帧级使用Attention Pooling")
    ap.add_argument("--no_frame_attn_pool", action="store_true", help="禁用帧级Attention Pooling")
    ap.add_argument("--frame_gru_layers", type=int, default=1, help="帧级GRU层数")
    ap.add_argument("--use_multiscale", action="store_true", default=True, help="使用多尺度帧特征融合")
    ap.add_argument("--no_multiscale", action="store_true", help="禁用多尺度帧特征")

    # DINOv3 feature extraction
    ap.add_argument(
        "--use_dinov3",
        action="store_true",
        default=False,
        help="使用 DINOv3 ViT 特征替代手工 8x8 网格特征",
    )
    ap.add_argument(
        "--dinov3_model",
        type=str,
        default=DINOV3_DEFAULT_MODEL,
        choices=list(DINOV3_MODELS.keys()),
        help="DINOv3 模型规模",
    )
    ap.add_argument(
        "--dinov3_feat_dim",
        type=int,
        default=None,
        help="DINOv3 特征维度（默认从模型自动推断）",
    )
    ap.add_argument(
        "--dinov3_roi_size",
        type=int,
        default=224,
        help="DINOv3 ROI 尺寸（必须能被 16 整除，默认 224）",
    )
    ap.add_argument(
        "--dinov3_batch_size",
        type=int,
        default=8,
        help="DINOv3 在线特征提取时的批大小（仅在 --use_dinov3 且无 --precomputed_dir 时生效）",
    )

    ap.add_argument("--no_focal", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--weight_pos", type=float, default=10.0)
    ap.add_argument("--loc5_weight", type=float, default=0.3)
    ap.add_argument("--within5_weight", type=float, default=0.7)
    ap.add_argument("--window_radius", type=int, default=5, help="穿透孔：真值层±R 范围内 timestep 加权半径")
    ap.add_argument("--in_window_weight", type=float, default=2.0, help="穿透孔窗口内 timestep CE 权重倍数")
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--use_grayscale", action="store_true", default=False, help="使用灰度图进行训练（默认False，即使用彩色图）")
    ap.add_argument("--patience", type=int, default=10, help="早停耐心值")
    ap.add_argument("--lr_scheduler", action="store_true", default=False, help="启用学习率调度器（ReduceLROnPlateau）")
    ap.add_argument("--lr_patience", type=int, default=5, help="学习率调度器耐心值")
    ap.add_argument("--lr_factor", type=float, default=0.5, help="学习率衰减因子")
    # Deep positional encoding
    ap.add_argument("--use_deep_pos", action="store_true", default=False, help="使用深度累积位置编码")

    ap.add_argument("--lock_layers", type=int, default=30)
    ap.add_argument("--val_decision", type=str, default="s3wd", choices=["s3wd", "topkmedian"], help="验证时决策方法")
    ap.add_argument("--val_k", type=int, default=9)
    ap.add_argument("--val_min_thresh", type=float, default=0.3)
    ap.add_argument("--val_s3wd_accept", type=float, default=0.9, help="S3WD 接受阈值")
    ap.add_argument("--val_s3wd_reject", type=float, default=0.75, help="S3WD 拒绝阈值")
    ap.add_argument("--val_s3wd_wait", type=int, default=3, help="S3WD 连续等待次数")
    ap.add_argument("--val_unc_samples", type=int, default=1, help="验证时 MC 采样次数；>1 启用方差估计")
    ap.add_argument("--val_unc_gate", action="store_true", help="验证时启用方差门控（仅 topkmedian 模式）")
    ap.add_argument("--val_unc_var_median_thresh", type=float, default=0.05, help="方差门控阈值（top-k 方差中位数）")

    # 网格搜索 S3WD 阈值
    ap.add_argument("--gs_s3wd", action="store_true", default=True, help="启用 S3WD 网格搜索（仅 val_decision=s3wd 时生效）")
    ap.add_argument(
        "--gs_accept_range", type=float, nargs="+", default=[0.7, 0.75, 0.8, 0.85, 0.9, 0.95],
        help="S3WD accept 阈值候选列表，如: 0.7 0.75 0.8 0.85 0.9 0.95",
    )
    ap.add_argument(
        "--gs_reject_range", type=float, nargs="+", default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75],
        help="S3WD reject 阈值候选列表（必须 < accept），如: 0.5 0.55 0.6 0.65 0.7 0.75",
    )
    ap.add_argument(
        "--gs_wait_range", type=int, nargs="+", default=[1, 2, 3, 4, 5],
        help="S3WD wait_consecutive 候选列表，如: 1 2 3 4 5",
    )
    ap.add_argument("--gs_save_grid", action="store_true", default=True, help="将网格搜索完整结果保存为 JSON 文件")

    args = ap.parse_args()

    if args.samples_info is None:
        args.samples_info = os.path.join(_REPO_ROOT, "data_drilling", "samples_info_train.json")
    args.samples_info = os.path.normpath(args.samples_info.replace("\\", os.sep))
    if not os.path.isfile(args.samples_info):
        raise FileNotFoundError(f"samples_info not found: {args.samples_info}")

    device = torch.device(args.device)
    use_amp = device.type == "cuda" and (not args.no_amp)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp) if device.type == "cuda" else None

    frame_channels = tuple(int(x) for x in str(args.frame_channels).split(",") if str(x).strip())
    layer_tcn_channels = tuple(int(x) for x in str(args.layer_tcn_channels).split(",") if str(x).strip())

    # DINOv3 feature dimension
    if args.use_dinov3:
        feat_dim = args.dinov3_feat_dim or DINOV3_FEAT_DIMS[args.dinov3_model]
        roi_size = args.dinov3_roi_size
        dinov3_img_size = args.dinov3_roi_size  # must be divisible by 16
        dinov3_roi_target = args.dinov3_roi_size
    else:
        feat_dim = 192  # hand-crafted 8x8 grid features
        roi_size = args.roi_size
        dinov3_img_size = None
        dinov3_roi_target = None

    # If using DINOv3 without precomputed features, initialize the extractor
    dinov3_extractor = None
    if args.use_dinov3:
        if args.precomputed_dir:
            print(f"[DINOv3] Using precomputed features from: {args.precomputed_dir}")
            print(f"[DINOv3] model={args.dinov3_model}, feat_dim={feat_dim}, roi_size={roi_size}")
        else:
            print(f"[DINOv3] Online feature extraction: model={args.dinov3_model}, feat_dim={feat_dim}")
            print(f"[DINOv3] roi_size={roi_size}, batch_size={args.dinov3_batch_size}")
            dinov3_extractor = DinoV3FeatureExtractor(
                model_name=args.dinov3_model,
                pretrained=True,
                pool_strategy="cls",
                image_size=dinov3_img_size,
                device=args.device,
            )
            dinov3_extractor = dinov3_extractor.to(device)
            dinov3_extractor.eval()

    if args.use_dinov3:
        ds = HierarchicalDinoV3Dataset(
            samples_info_path=args.samples_info,
            dinov3_extractor=dinov3_extractor,
            dinov3_feat_dim=feat_dim,
            roi_size=roi_size,
            target_size=(int(args.img_size), int(args.img_size)),
            max_layers=args.max_layers,
            max_frames_per_layer=int(args.max_frames_per_layer),
            penetration_radius=int(args.penetration_radius),
            exclude_json=args.exclude_json,
            final_roi_scale=float(args.final_roi_scale),
            cc_min_area=int(args.cc_min_area),
            cc_expand_ratio=float(args.cc_expand_ratio),
            min_laser_pixels=int(args.min_laser_pixels),
            min_laser_area_ratio=float(args.min_laser_area_ratio),
            roi_window_side=int(args.roi_window_side),
            roi_bright_min_ratio=float(args.roi_bright_min_ratio),
            roi_gray_p95_min=float(args.roi_gray_p95_min),
            use_color_cc_v2_geometry=(not bool(args.legacy_color_cc_geometry)),
            precomputed_dir=(os.path.normpath(args.precomputed_dir) if args.precomputed_dir else None),
            use_grayscale=bool(args.use_grayscale),
            _dinov3_target_size=dinov3_roi_target or 224,
        )
    else:
        ds = HierarchicalFrameLayerDataset(
            samples_info_path=args.samples_info,
            base_dir=args.base_dir,
            target_size=(int(args.img_size), int(args.img_size)),
            roi_size=int(args.roi_size),
            grid=(8, 8),
            pool_stats=("mean", "std", "max"),
            max_layers=args.max_layers,
            max_frames_per_layer=int(args.max_frames_per_layer),
            penetration_radius=int(args.penetration_radius),
            exclude_json=args.exclude_json,
            final_roi_scale=float(args.final_roi_scale),
            cc_min_area=int(args.cc_min_area),
            cc_expand_ratio=float(args.cc_expand_ratio),
            min_laser_pixels=int(args.min_laser_pixels),
            min_laser_area_ratio=float(args.min_laser_area_ratio),
            roi_window_side=int(args.roi_window_side),
            roi_bright_min_ratio=float(args.roi_bright_min_ratio),
            roi_gray_p95_min=float(args.roi_gray_p95_min),
            use_color_cc_v2_geometry=(not bool(args.legacy_color_cc_geometry)),
            precomputed_dir=(os.path.normpath(args.precomputed_dir) if args.precomputed_dir else None),
            use_grayscale=bool(args.use_grayscale),
        )
    n = len(ds)
    if n == 0:
        raise RuntimeError("empty dataset")

    feat_type = f"DINOv3 {args.dinov3_model}" if args.use_dinov3 else "grid 8x8"
    pc = ds.precomputed_dir
    if not pc:
        print(
            f"[hierarchical] 未设置 --precomputed_dir，训练将在线裁 ROI + {feat_type} 特征（慢）。"
            " 可先运行 hier/dinov3_precompute.py 或 hier/precompute.py 再传入同一目录。"
        )
    else:
        n_pt = len(glob(os.path.join(pc, "*.pt")))
        print(
            f"[hierarchical] precomputed_dir={pc}，目录内 .pt 数量={n_pt}，数据集样本数={n}"
        )
        if n_pt == 0:
            print(
                "[hierarchical] 警告：预计算目录下没有 .pt，将全部退回在线计算。"
            )
        elif n_pt < n:
            print(
                f"[hierarchical] 警告：.pt 少于样本数（缺 {n - n_pt}），"
                "缺缓存的样本将退回在线计算。若预计算启用了「无 ROI 跳过」，"
                "未写入的孔见预计算目录下 precompute_skipped_no_roi.txt。"
            )

    all_idx = list(range(n))
    rng = random.Random(int(args.val_seed))
    rng.shuffle(all_idx)
    nv = int(max(1, round(n * float(args.val_ratio)))) if 0.0 < float(args.val_ratio) < 1.0 else 0
    val_idx = set(all_idx[:nv])
    tr_idx = [i for i in all_idx if i not in val_idx]
    va_idx = [i for i in all_idx if i in val_idx]
    train_ds = Subset(ds, tr_idx)
    val_ds = Subset(ds, va_idx) if va_idx else None

    train_loader = DataLoader(
        train_ds,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=int(args.num_workers),
        collate_fn=collate_hierarchical_batch,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(args.batch_size),
            shuffle=False,
            num_workers=max(0, int(args.num_workers) // 2),
            collate_fn=collate_hierarchical_batch,
            pin_memory=(device.type == "cuda"),
            persistent_workers=(int(args.num_workers) > 1),
            prefetch_factor=2 if int(args.num_workers) > 1 else None,
        )

    model = HierarchicalGridDiffProbTransformer(
        in_channels_frame=feat_dim,   # 768 for DINOv3 ViT-B, 192 for hand-crafted grid
        out_channels=2,
        frame_channels=frame_channels if frame_channels else (128, 128),
        layer_tcn_channels=layer_tcn_channels if layer_tcn_channels else (128, 128),
        kernel_size=int(args.kernel_size),
        d_model=int(args.d_model),
        nhead=int(args.num_heads),
        num_layers=int(args.num_transformer_layers),
        dim_feedforward=int(args.dim_feedforward),
        dropout=float(args.dropout),
        add_kl=True,
        return_kl=(float(args.kl_weight) > 0.0),
        extra_dim=int(args.extra_dim),
        use_frame_gru=(not args.no_frame_gru) if args.use_frame_gru else False,
        use_frame_attn_pool=(not args.no_frame_attn_pool) if args.use_frame_attn_pool else False,
        frame_gru_layers=int(args.frame_gru_layers),
        use_multiscale=(not args.no_multiscale) if args.use_multiscale else False,

    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))
    
    scheduler = None
    if args.lr_scheduler and val_loader is not None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=float(args.lr_factor), patience=int(args.lr_patience)
        )
    
    best_pct5 = -1.0
    best_s3wd_params: Optional[Dict[str, Any]] = None
    patience_counter = 0
    os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
    for ep in range(int(args.epochs)):
        loss = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_focal=(not args.no_focal),
            focal_gamma=float(args.focal_gamma),
            focal_alpha=float(args.focal_alpha),
            weight_pos=float(args.weight_pos),
            loc5_weight=float(args.loc5_weight),
            within5_weight=float(args.within5_weight),
            window_radius=int(args.window_radius),
            in_window_weight=float(args.in_window_weight),
            kl_weight=float(args.kl_weight),
            use_amp=use_amp,
            scaler=scaler,
        )
        line = f"Epoch {ep+1}/{args.epochs} TrainLoss={loss:.4f}"
        if val_loader is not None:
            met, raw_data = run_validation(
                model,
                val_loader,
                device=device,
                lock_layers=int(args.lock_layers),
                decision_method=str(args.val_decision),
                topk_k=int(args.val_k),
                topk_min_thresh=float(args.val_min_thresh),
                s3wd_accept_thresh=float(args.val_s3wd_accept),
                s3wd_reject_thresh=float(args.val_s3wd_reject),
                s3wd_wait_consecutive=int(args.val_s3wd_wait),
                unc_samples=int(args.val_unc_samples),
                use_uncertainty_gate=bool(args.val_unc_gate),
                unc_var_median_thresh=float(args.val_unc_var_median_thresh),
                use_amp=use_amp,
            )

            # 网格搜索 S3WD 阈值（仅 s3wd 模式生效）
            if str(args.val_decision) == "s3wd" and args.gs_s3wd:
                gs_result = grid_search_s3wd(
                    probs_all=raw_data["probs_all"],
                    layer_lists=raw_data["layer_lists"],
                    labels=raw_data["labels"],
                    pen_layers=raw_data["pen_layers"],
                    lock_layers=int(args.lock_layers),
                    accept_candidates=args.gs_accept_range,
                    reject_candidates=args.gs_reject_range,
                    wait_candidates=args.gs_wait_range,
                )
                # 记录最优参数到 checkpoint
                best_s3wd_params = gs_result["best"]["params"]
                met = gs_result["best"]["metrics"]
                line += (
                    f" | GS3WD accept={best_s3wd_params['accept']:.2f}"
                    f" reject={best_s3wd_params['reject']:.2f}"
                    f" wait={best_s3wd_params['wait']}"
                    f" <=3:{met['pct_within_3']:.1f}% <=5:{met['pct_within_5']:.1f}% >10:{met['pct_over_10']:.1f}%"
                )
                if args.gs_save_grid and gs_result.get("grid"):
                    import json as _json
                    grid_path = os.path.splitext(args.save)[0] + "_s3wd_grid.json"
                    with open(grid_path, "w", encoding="utf-8") as _f:
                        _json.dump(gs_result, _f, ensure_ascii=False, indent=2)
                    print(f"[grid_search] 网格搜索结果已保存: {grid_path}")

            line += (
                f" | ValHole n={met['n_penetrated']} <=3:{met['pct_within_3']:.1f}%"
                f" <=5:{met['pct_within_5']:.1f}% >10:{met['pct_over_10']:.1f}%"
            )
            if met["pct_within_5"] > best_pct5:
                best_pct5 = met["pct_within_5"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "config": {
                            "in_channels_frame": feat_dim,
                            "frame_channels": frame_channels,
                            "layer_tcn_channels": layer_tcn_channels,
                            "kernel_size": int(args.kernel_size),
                            "d_model": int(args.d_model),
                            "num_heads": int(args.num_heads),
                            "num_transformer_layers": int(args.num_transformer_layers),
                            "dinov3_model": args.dinov3_model if args.use_dinov3 else None,
                            "dinov3_feat_dim": feat_dim if args.use_dinov3 else None,
                            "s3wd_best_params": best_s3wd_params if args.gs_s3wd and str(args.val_decision) == "s3wd" else None,
                        },
                    },
                    args.save,
                )
                line += "  [saved_best]"
                patience_counter = 0
            else:
                patience_counter += 1
                if scheduler is not None:
                    scheduler.step(met["pct_within_5"])
            print(line)
            
            if patience_counter >= int(args.patience):
                print(f"Early stopping at epoch {ep+1}")
                break

    if best_pct5 < 0:
        torch.save({"model": model.state_dict()}, args.save)
        print(f"saved final model: {args.save}")
    else:
        print(f"best <=5: {best_pct5:.1f}% | model: {args.save}")


if __name__ == "__main__":
    main()

