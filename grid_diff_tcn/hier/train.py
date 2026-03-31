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

from grid_diff_tcn.hier.frame_layer import (
    HierarchicalFrameLayerDataset,
    collate_hierarchical_batch,
    HierarchicalGridDiffProbTransformer,
)
from grid_diff_tcn.hier.decision import (
    apply_safety_lock,
    topkmedian_decide,
    topkmedian_with_uncertainty_gate,
)


def focal_cross_entropy(logits_bt, seq_y_bt, gamma=2.0, alpha_pos=0.75):
    probs = F.softmax(logits_bt, dim=1)
    pt = probs.gather(1, seq_y_bt.unsqueeze(1)).squeeze(1).clamp(min=1e-8)
    alpha_t = torch.where(
        seq_y_bt == 1,
        torch.tensor(alpha_pos, device=logits_bt.device, dtype=logits_bt.dtype),
        torch.tensor(1.0 - alpha_pos, device=logits_bt.device, dtype=logits_bt.dtype),
    )
    focal_w = (1 - pt).pow(gamma)
    return focal_w * (-pt.log()) * alpha_t


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


def run_validation(
    model,
    loader,
    device,
    lock_layers: int,
    topk_k: int,
    topk_min_thresh: float,
    unc_samples: int = 1,
    use_uncertainty_gate: bool = False,
    unc_var_median_thresh: float = 0.05,
):
    model.eval()
    recs = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Val(hole)", leave=False):
            x = batch["frame_data"].to(device, non_blocking=True)  # (B,T,F,C)
            fm = batch["frame_mask"].to(device, non_blocking=True)
            lx = batch.get("layer_extra")
            lx = lx.to(device, non_blocking=True) if torch.is_tensor(lx) else None
            layer_mask = batch["layer_mask"].to(device, non_blocking=True)
            if int(unc_samples) > 1:
                runs = []
                for _ in range(int(unc_samples)):
                    out = model(x, frame_mask=fm, force_sample_attention=True, layer_extra=lx)
                    logits = out[0] if isinstance(out, tuple) else out
                    runs.append(F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy())  # (B,T)
                runs = np.stack(runs, axis=0)  # (S,B,T)
                probs_mean = runs.mean(axis=0)
                probs_var = runs.var(axis=0)
                probs = probs_mean
            else:
                out = model(x, frame_mask=fm, layer_extra=lx)
                logits = out[0] if isinstance(out, tuple) else out
                probs = F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy()  # (B,T)
                probs_var = None

            for i in range(probs.shape[0]):
                valid_t = int(layer_mask[i].sum().item())
                p = np.asarray(probs[i, :valid_t], dtype=np.float64)
                p = apply_safety_lock(p, lock_layers=lock_layers)
                if use_uncertainty_gate and probs_var is not None:
                    pv = np.asarray(probs_var[i, :valid_t], dtype=np.float64)
                    pred_pen, pred_idx, _ = topkmedian_with_uncertainty_gate(
                        mean_probs=p,
                        var_probs=pv,
                        k=topk_k,
                        min_thresh=topk_min_thresh,
                        unc_var_median_thresh=float(unc_var_median_thresh),
                    )
                else:
                    pred_pen, pred_idx = topkmedian_decide(p, k=topk_k, min_thresh=topk_min_thresh)
                true_label = int(batch["label"][i].item())
                true_layer = int(batch["penetration_layer"][i].item())
                layer_list = batch["layer_list"][i]
                true_idx = layer_list.index(true_layer) if (true_label == 1 and true_layer in layer_list) else None
                recs.append(
                    {
                        "true_label": true_label,
                        "true_penetration_index": true_idx,
                        "pred_penetration_index": pred_idx if pred_pen else None,
                    }
                )
    return compute_hole_metrics(recs)


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
                    step_loss = focal_cross_entropy(logits_bt, y_bt, gamma=focal_gamma, alpha_pos=focal_alpha)
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
                step_loss = focal_cross_entropy(logits_bt, y_bt, gamma=focal_gamma, alpha_pos=focal_alpha)
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
    ap.add_argument("--roi_window_side", type=int, default=None, help="color_cc 中心固定正方形裁切边长")
    ap.add_argument("--roi_bright_min_ratio", type=float, default=0.0, help="letterbox 后 ROI 亮区占比下限；0=关闭")
    ap.add_argument("--roi_gray_p95_min", type=float, default=0.0, help="letterbox 后灰度 p95 下限；0=关闭")
    ap.add_argument(
        "--legacy_color_cc_geometry",
        action="store_true",
        help="使用旧 shrink(min边) 几何",
    )
    ap.add_argument("--use_hole_anchor_box", action="store_true", help="每孔前 N 张定锚框，全孔复用")
    ap.add_argument("--hole_anchor_num_images", type=int, default=10)

    ap.add_argument("--frame_channels", type=str, default="64,64")
    ap.add_argument("--layer_tcn_channels", type=str, default="64,64")
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=64)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_transformer_layers", type=int, default=2)
    ap.add_argument("--dim_feedforward", type=int, default=256)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--kl_weight", type=float, default=0.0)
    ap.add_argument("--extra_dim", type=int, default=41, help="在线 layer_extra 特征维度；0=禁用融合")

    ap.add_argument("--no_focal", action="store_true")
    ap.add_argument("--focal_gamma", type=float, default=2.0)
    ap.add_argument("--focal_alpha", type=float, default=0.75)
    ap.add_argument("--weight_pos", type=float, default=10.0)
    ap.add_argument("--loc5_weight", type=float, default=0.3)
    ap.add_argument("--within5_weight", type=float, default=0.7)
    ap.add_argument("--window_radius", type=int, default=5, help="穿透孔：真值层±R 范围内 timestep 加权半径")
    ap.add_argument("--in_window_weight", type=float, default=2.0, help="穿透孔窗口内 timestep CE 权重倍数")
    ap.add_argument("--no_amp", action="store_true")

    ap.add_argument("--lock_layers", type=int, default=30)
    ap.add_argument("--val_k", type=int, default=9)
    ap.add_argument("--val_min_thresh", type=float, default=0.3)
    ap.add_argument("--val_unc_samples", type=int, default=1, help="验证时 MC 采样次数；>1 启用方差估计")
    ap.add_argument("--val_unc_gate", action="store_true", help="验证时启用方差门控（TopKMedian + uncertainty gate）")
    ap.add_argument("--val_unc_var_median_thresh", type=float, default=0.05, help="方差门控阈值（top-k 方差中位数）")
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

    ds = HierarchicalFrameLayerDataset(
        samples_info_path=args.samples_info,
        base_dir=args.base_dir,
        target_size=(int(args.img_size), int(args.img_size)),
        roi_size=int(args.roi_size),
        grid=(8, 8),
        max_layers=args.max_layers,
        max_frames_per_layer=int(args.max_frames_per_layer),
        penetration_radius=int(args.penetration_radius),
        exclude_json=args.exclude_json,
        final_roi_scale=float(args.final_roi_scale),
        cc_min_area=int(args.cc_min_area),
        cc_expand_ratio=float(args.cc_expand_ratio),
        min_laser_pixels=int(args.min_laser_pixels),
        min_laser_area_ratio=float(args.min_laser_area_ratio),
        roi_window_side=(int(args.roi_window_side) if args.roi_window_side is not None else None),
        roi_bright_min_ratio=float(args.roi_bright_min_ratio),
        roi_gray_p95_min=float(args.roi_gray_p95_min),
        use_color_cc_v2_geometry=(not bool(args.legacy_color_cc_geometry)),
        use_hole_anchor_box=bool(args.use_hole_anchor_box),
        hole_anchor_num_images=int(args.hole_anchor_num_images),
        precomputed_dir=(os.path.normpath(args.precomputed_dir) if args.precomputed_dir else None),
    )
    n = len(ds)
    if n == 0:
        raise RuntimeError("empty dataset")

    pc = ds.precomputed_dir
    if not pc:
        print(
            "[hierarchical] 未设置 --precomputed_dir，训练将在线裁 ROI + grid 特征（慢）。"
            " 可先运行 hier/precompute.py 再传入同一目录。"
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
                "未生成缓存的样本仍会在线计算，建议补跑预计算脚本。"
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
        in_channels_frame=64,
        out_channels=2,
        frame_channels=frame_channels if frame_channels else (64, 64),
        layer_tcn_channels=layer_tcn_channels if layer_tcn_channels else (64, 64),
        kernel_size=int(args.kernel_size),
        d_model=int(args.d_model),
        nhead=int(args.num_heads),
        num_layers=int(args.num_transformer_layers),
        dim_feedforward=int(args.dim_feedforward),
        dropout=float(args.dropout),
        add_kl=True,
        return_kl=(float(args.kl_weight) > 0.0),
        extra_dim=int(args.extra_dim),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args.lr))

    best_pct5 = -1.0
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
            met = run_validation(
                model,
                val_loader,
                device=device,
                lock_layers=int(args.lock_layers),
                topk_k=int(args.val_k),
                topk_min_thresh=float(args.val_min_thresh),
                unc_samples=int(args.val_unc_samples),
                use_uncertainty_gate=bool(args.val_unc_gate),
                unc_var_median_thresh=float(args.val_unc_var_median_thresh),
            )
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
                            "in_channels_frame": 64,
                            "frame_channels": frame_channels,
                            "layer_tcn_channels": layer_tcn_channels,
                            "kernel_size": int(args.kernel_size),
                            "d_model": int(args.d_model),
                            "num_heads": int(args.num_heads),
                            "num_transformer_layers": int(args.num_transformer_layers),
                        },
                    },
                    args.save,
                )
                line += "  [saved_best]"
        print(line)

    if best_pct5 < 0:
        torch.save({"model": model.state_dict()}, args.save)
        print(f"saved final model: {args.save}")
    else:
        print(f"best <=5: {best_pct5:.1f}% | model: {args.save}")


if __name__ == "__main__":
    main()

