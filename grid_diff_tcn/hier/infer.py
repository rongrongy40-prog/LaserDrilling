# -*- coding: utf-8 -*-
"""
Hierarchical model inference script.

Loads HierarchicalGridDiffProbTransformer and evaluates hole-level penetration
prediction with the same metric style as existing inference.py.
"""

import os
import json
import argparse
from glob import glob
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRID_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

from grid_diff_tcn.common.roi_crop_defaults import DEFAULT_ROI_WINDOW_SIDE
from grid_diff_tcn.hier.frame_layer import (  # noqa: E402
    DinoV3FeatureExtractor,
    HierarchicalDinoV3Dataset,
    HierarchicalFrameLayerDataset,
    collate_hierarchical_batch,
    HierarchicalGridDiffProbTransformer,
    DINOV3_FEAT_DIMS,
)
from grid_diff_tcn.common.decision import (  # noqa: E402
    apply_safety_lock,
    s3wd_decide,
    topkmedian_decide,
    topkmedian_with_uncertainty_gate,
)


def _to_jsonable(x: Any):
    if x is None:
        return None
    if hasattr(x, "tolist"):
        return x.tolist()
    if hasattr(x, "__array__"):
        return np.asarray(x).tolist()
    return x


def _metrics_from_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    n_pen = 0
    n3 = n5 = n10 = 0
    for r in results:
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
        "n_penetrated": int(n_pen),
        "pct_within_3": (float(n3) / n_pen * 100.0) if n_pen else 0.0,
        "pct_within_5": (float(n5) / n_pen * 100.0) if n_pen else 0.0,
        "pct_over_10": (float(n10) / n_pen * 100.0) if n_pen else 0.0,
    }


def _build_model(
    ckpt: Dict[str, Any],
    args: argparse.Namespace,
    device: torch.device,
) -> HierarchicalGridDiffProbTransformer:
    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    frame_channels = tuple(int(x) for x in str(args.frame_channels).split(",") if str(x).strip())
    layer_tcn_channels = tuple(int(x) for x in str(args.layer_tcn_channels).split(",") if str(x).strip())

    model = HierarchicalGridDiffProbTransformer(
        in_channels_frame=int(cfg.get("in_channels_frame", 192)),
        out_channels=2,
        frame_channels=tuple(cfg.get("frame_channels", frame_channels if frame_channels else (64, 64))),
        layer_tcn_channels=tuple(cfg.get("layer_tcn_channels", layer_tcn_channels if layer_tcn_channels else (64, 64))),
        kernel_size=int(cfg.get("kernel_size", args.kernel_size)),
        d_model=int(cfg.get("d_model", args.d_model)),
        nhead=int(cfg.get("num_heads", args.num_heads)),
        num_layers=int(cfg.get("num_transformer_layers", args.num_transformer_layers)),
        dim_feedforward=int(getattr(args, "dim_feedforward", 256)),
        dropout=float(args.dropout),
        add_kl=True,
        return_kl=False,
        extra_dim=int(getattr(args, "extra_dim", 0)),
        use_frame_gru=bool(getattr(args, "use_frame_gru", True)),
        use_frame_attn_pool=bool(getattr(args, "use_frame_attn_pool", True)),
        frame_gru_layers=int(getattr(args, "frame_gru_layers", 1)),
        use_multiscale=bool(getattr(args, "use_multiscale", True)),
    ).to(device)

    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[_build_model] missing keys (will use defaults): {missing}")
    if unexpected:
        print(f"[_build_model] unexpected keys (ignored): {unexpected}")
    model.eval()
    return model


def _decide(
    probs: np.ndarray,
    method: str,
    topk_k: int,
    topk_min_thresh: float,
    s3wd_accept_thresh: float = 0.9,
    s3wd_reject_thresh: float = 0.75,
    s3wd_wait_consecutive: int = 3,
) -> Tuple[bool, Optional[int]]:
    if method == "s3wd":
        return s3wd_decide(
            probs,
            accept_thresh=float(s3wd_accept_thresh),
            reject_thresh=float(s3wd_reject_thresh),
            wait_consecutive=int(s3wd_wait_consecutive),
        )
    return topkmedian_decide(probs, k=topk_k, min_thresh=topk_min_thresh)


def evaluate_dataset(
    model: HierarchicalGridDiffProbTransformer,
    loader: DataLoader,
    device: torch.device,
    lock_layers: int,
    decision_method: str,
    topk_k: int,
    topk_min_thresh: float,
    s3wd_accept_thresh: float = 0.9,
    s3wd_reject_thresh: float = 0.75,
    s3wd_wait_consecutive: int = 3,
    unc_samples: int = 0,
    use_uncertainty_gate: bool = False,
    unc_var_median_thresh: float = 0.05,
) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="推理", unit="孔"):
            x = batch["frame_data"].to(device, non_blocking=True)  # (B,T,F,C)
            fm = batch["frame_mask"].to(device, non_blocking=True)
            lx = batch.get("layer_extra")
            lx = lx.to(device, non_blocking=True) if torch.is_tensor(lx) else None
            lm = batch["layer_mask"].to(device, non_blocking=True)
            bsz = x.size(0)

            if int(unc_samples) > 1:
                prob_runs = []
                for _ in range(int(unc_samples)):
                    out = model(x, frame_mask=fm, force_sample_attention=True, layer_extra=lx)
                    logits = out[0] if isinstance(out, tuple) else out
                    prob_runs.append(F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy())  # (B,T)
                pr = np.stack(prob_runs, axis=0)  # (S,B,T)
                probs_mean_bt = pr.mean(axis=0)
                probs_var_bt = pr.var(axis=0)
                probs_bt = probs_mean_bt
            else:
                out = model(x, frame_mask=fm, force_sample_attention=False, layer_extra=lx)
                logits = out[0] if isinstance(out, tuple) else out
                probs_bt = F.softmax(logits, dim=1)[:, 1, :].detach().cpu().numpy()
                probs_mean_bt = None
                probs_var_bt = None

            for i in range(bsz):
                valid_t = int(lm[i].sum().item())
                layer_list = batch["layer_list"][i]
                p = np.asarray(probs_bt[i, :valid_t], dtype=np.float64)
                p = apply_safety_lock(p, lock_layers=lock_layers)
                if decision_method == "topkmedian" and use_uncertainty_gate and probs_var_bt is not None:
                    pv = np.asarray(probs_var_bt[i, :valid_t], dtype=np.float64)
                    pred_pen, pred_idx, _ = topkmedian_with_uncertainty_gate(
                        mean_probs=p,
                        var_probs=pv,
                        k=topk_k,
                        min_thresh=topk_min_thresh,
                        unc_var_median_thresh=float(unc_var_median_thresh),
                    )
                else:
                    pred_pen, pred_idx = _decide(
                        p,
                        method=decision_method,
                        topk_k=topk_k,
                        topk_min_thresh=topk_min_thresh,
                        s3wd_accept_thresh=float(s3wd_accept_thresh),
                        s3wd_reject_thresh=float(s3wd_reject_thresh),
                        s3wd_wait_consecutive=int(s3wd_wait_consecutive),
                    )

                true_label = int(batch["label"][i].item())
                true_layer = int(batch["penetration_layer"][i].item())
                true_idx = layer_list.index(true_layer) if (true_label == 1 and true_layer in layer_list) else None
                pred_layer = layer_list[pred_idx] if (pred_pen and pred_idx is not None and 0 <= pred_idx < len(layer_list)) else None

                pm = None if probs_mean_bt is None else apply_safety_lock(np.asarray(probs_mean_bt[i, :valid_t], dtype=np.float64), lock_layers=lock_layers)
                pv = None if probs_var_bt is None else np.asarray(probs_var_bt[i, :valid_t], dtype=np.float64)

                results.append(
                    {
                        "sample_path": batch["sample_path"][i],
                        "true_label": true_label,
                        "true_penetration_layer": true_layer,
                        "true_penetration_index": true_idx,
                        "pred_penetrated": bool(pred_pen),
                        "pred_penetration_layer": pred_layer,
                        "pred_penetration_index": pred_idx if pred_pen else None,
                        "probs": _to_jsonable(p),
                        "probs_mean": _to_jsonable(pm),
                        "probs_var": _to_jsonable(pv),
                    }
                )
    return results


def main():
    ap = argparse.ArgumentParser(description="Hierarchical model inference")
    ap.add_argument("--samples_info", type=str, default=None)
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--ckpt", type=str, required=True, help="hierarchical model checkpoint path")
    ap.add_argument("--output", type=str, default=os.path.join(_GRID_ROOT, "inference_results_hierarchical.json"))
    ap.add_argument("--max_samples", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=4)
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
    ap.add_argument("--min_laser_pixels", type=int, default=0)
    ap.add_argument("--min_laser_area_ratio", type=float, default=0.0)
    ap.add_argument(
        "--roi_window_side",
        type=int,
        default=DEFAULT_ROI_WINDOW_SIDE,
        help="与 visualize_roi/precompute 一致；0=关闭",
    )
    ap.add_argument("--roi_bright_min_ratio", type=float, default=0.0)
    ap.add_argument("--roi_gray_p95_min", type=float, default=0.0)
    ap.add_argument("--legacy_color_cc_geometry", action="store_true")
    ap.add_argument("--use_grayscale", action="store_true", default=False, help="使用灰度图进行推理（默认False，即使用彩色图）")
    ap.add_argument("--frame_channels", type=str, default="128,128")
    ap.add_argument("--layer_tcn_channels", type=str, default="128,128")
    ap.add_argument("--kernel_size", type=int, default=3)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_transformer_layers", type=int, default=2)
    ap.add_argument("--dim_feedforward", type=int, default=512)
    ap.add_argument("--dropout", type=float, default=0.1)
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
        help="使用 DINOv3 特征代替手工 8x8 网格特征",
    )
    ap.add_argument(
        "--dinov3_model",
        type=str,
        default="vit_base",
        choices=list(DINOV3_FEAT_DIMS.keys()),
        help="DINOv3 模型名称",
    )
    ap.add_argument(
        "--dinov3_feat_dim",
        type=int,
        default=None,
        help="DINOv3 特征维度（自动从模型名推导，可手动覆盖）",
    )
    ap.add_argument(
        "--dinov3_roi_size",
        type=int,
        default=224,
        help="DINOv3 ROI 裁剪目标尺寸（必须是 16 的倍数）",
    )
    ap.add_argument("--dinov3_batch_size", type=int, default=32, help="DINOv3 在线特征提取批大小")

    ap.add_argument("--decision_method", type=str, default="s3wd", choices=["topkmedian", "s3wd"])
    ap.add_argument("--lock_layers", type=int, default=30)
    ap.add_argument("--k", type=int, default=9, help="topkmedian k")
    ap.add_argument("--min_thresh", type=float, default=0.3, help="topkmedian min threshold")
    ap.add_argument("--s3wd_accept", type=float, default=0.9, help="S3WD accept threshold")
    ap.add_argument("--s3wd_reject", type=float, default=0.75, help="S3WD reject threshold")
    ap.add_argument("--s3wd_wait", type=int, default=3, help="S3WD consecutive wait count")
    ap.add_argument("--unc_samples", type=int, default=0, help=">1 enables stochastic attention sampling")
    ap.add_argument("--unc_gate", action="store_true", help="TopKMedian 下启用方差门控")
    ap.add_argument("--unc_var_median_thresh", type=float, default=0.05, help="TopKMedian 方差门控阈值")
    args = ap.parse_args()

    if args.samples_info is None:
        args.samples_info = os.path.join(_REPO_ROOT, "data_drilling", "samples_info_test.json")
    args.samples_info = os.path.normpath(args.samples_info.replace("\\", os.sep))
    if not os.path.isfile(args.samples_info):
        raise FileNotFoundError(f"samples_info not found: {args.samples_info}")
    if not os.path.isfile(args.ckpt):
        raise FileNotFoundError(f"ckpt not found: {args.ckpt}")

    device = torch.device(args.device)

    # DINOv3 feature dimension
    feat_dim = 192
    if getattr(args, "use_dinov3", False):
        feat_dim = args.dinov3_feat_dim or DINOV3_FEAT_DIMS[args.dinov3_model]
        if args.precomputed_dir:
            print(f"[DINOv3] Using precomputed features from: {args.precomputed_dir}")
            print(f"[DINOv3] model={args.dinov3_model}, feat_dim={feat_dim}")
        else:
            print(f"[DINOv3] Online feature extraction: model={args.dinov3_model}, feat_dim={feat_dim}")

    ckpt = torch.load(args.ckpt, map_location=device)
    model = _build_model(ckpt, args=args, device=device)

    if getattr(args, "use_dinov3", False):
        dinov3_extractor = None
        if not args.precomputed_dir:
            dinov3_extractor = DinoV3FeatureExtractor(
                model_name=args.dinov3_model,
                pretrained=True,
                pool_strategy="cls",
                image_size=args.dinov3_roi_size,
                device=args.device,
            )
            dinov3_extractor = dinov3_extractor.to(device)
            dinov3_extractor.eval()
            print(f"[DINOv3] Online extractor ready. roi_size={args.dinov3_roi_size}, batch_size={args.dinov3_batch_size}")

        ds = HierarchicalDinoV3Dataset(
            samples_info_path=args.samples_info,
            dinov3_extractor=dinov3_extractor,
            dinov3_feat_dim=feat_dim,
            roi_size=args.dinov3_roi_size,
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
            _dinov3_target_size=args.dinov3_roi_size,
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
    if len(ds) == 0:
        raise RuntimeError("empty dataset")
    if args.max_samples is not None:
        n = max(0, min(int(args.max_samples), len(ds)))
        ds.samples = ds.samples[:n]
        ds._precomputed_name_for_idx = ds._precomputed_name_for_idx[:n]

    pc = ds.precomputed_dir
    if not pc:
        print(
            "[hier_infer] 未设置 --precomputed_dir，推理将在线裁 ROI + 提特征（通常较慢）。"
        )
    else:
        n_all = len(ds)
        n_pt = len(glob(os.path.join(pc, "*.pt")))
        hit = 0
        for i in range(n_all):
            p1 = os.path.join(pc, ds._precomputed_name_for_idx[i])
            p2 = os.path.join(pc, f"{i}.pt")
            if os.path.isfile(p1) or os.path.isfile(p2):
                hit += 1
        print(
            f"[hier_infer] precomputed_dir={pc} | .pt={n_pt} | 样本数={n_all} | 缓存命中={hit}/{n_all}"
        )
        if hit < n_all:
            print(
                f"[hier_infer] 警告：有 {n_all - hit} 个样本缺缓存，会退回在线处理，速度明显变慢。"
            )

    loader = DataLoader(
        ds,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=int(args.num_workers),
        collate_fn=collate_hierarchical_batch,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(int(args.num_workers) > 0),
        prefetch_factor=2 if int(args.num_workers) > 0 else None,
    )

    results = evaluate_dataset(
        model=model,
        loader=loader,
        device=device,
        lock_layers=int(args.lock_layers),
        decision_method=str(args.decision_method),
        topk_k=int(args.k),
        topk_min_thresh=float(args.min_thresh),
        s3wd_accept_thresh=float(args.s3wd_accept),
        s3wd_reject_thresh=float(args.s3wd_reject),
        s3wd_wait_consecutive=int(args.s3wd_wait),
        unc_samples=int(args.unc_samples),
        use_uncertainty_gate=bool(args.unc_gate),
        unc_var_median_thresh=float(args.unc_var_median_thresh),
    )
    metrics = _metrics_from_results(results)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    payload = {
        "decision_method": str(args.decision_method),
        "decision_params": {
            "lock_layers": int(args.lock_layers),
            "k": int(args.k),
            "min_thresh": float(args.min_thresh),
            "s3wd_accept": float(args.s3wd_accept),
            "s3wd_reject": float(args.s3wd_reject),
            "s3wd_wait": int(args.s3wd_wait),
            "unc_samples": int(args.unc_samples),
            "unc_gate": bool(args.unc_gate),
            "unc_var_median_thresh": float(args.unc_var_median_thresh),
        },
        "metrics": metrics,
        "results": results,
    }
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Done. n_pen={metrics['n_penetrated']} "
        f"<=3:{metrics['pct_within_3']:.1f}% "
        f"<=5:{metrics['pct_within_5']:.1f}% "
        f">10:{metrics['pct_over_10']:.1f}%"
    )
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

