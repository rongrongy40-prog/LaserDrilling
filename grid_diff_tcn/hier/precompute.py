# -*- coding: utf-8 -*-
"""
Precompute hierarchical per-hole features for hier/train.py.

Each sample saved as .pt contains:
  - frame_data: (T,F,192)  8*8*3 (mean+std+max) grid features
  - frame_mask: (T,F)
  - seq_label:  (T,)
  - label, penetration_layer, layer_list, sample_path
"""

from __future__ import annotations

import os
import argparse
import multiprocessing as mp
import torch
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grid_diff_tcn.common.roi_crop_defaults import (
    DEFAULT_CC_EXPAND_RATIO,
    DEFAULT_CC_MIN_AREA,
    DEFAULT_FINAL_ROI_SCALE,
    DEFAULT_MIN_LASER_AREA_RATIO,
    DEFAULT_MIN_LASER_PIXELS,
    DEFAULT_ROI_BRIGHT_MIN_RATIO,
    DEFAULT_ROI_GRAY_P95_MIN,
    DEFAULT_ROI_SIZE,
    DEFAULT_ROI_WINDOW_SIDE,
    DEFAULT_TARGET_WH,
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRID_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

# Worker globals（通过 initializer 每个子进程各一份）
_worker_ds = None


def _init_worker(ds_kwargs: dict) -> None:
    global _worker_ds
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        import cv2

        cv2.setNumThreads(0)
    except ImportError:
        pass

    from grid_diff_tcn.hier.frame_layer import HierarchicalFrameLayerDataset

    _worker_ds = HierarchicalFrameLayerDataset(**ds_kwargs)


def _compute_one(args: tuple) -> tuple[str, str]:
    i, out_dir = args
    global _worker_ds
    assert _worker_ds is not None
    sample_path = str(_worker_ds.samples[i].get("sample_path", ""))
    name = _worker_ds._precomputed_name_for_idx[i]
    out_path = os.path.join(out_dir, name)
    if os.path.isfile(out_path):
        return ("skip_exists", sample_path)
    s = _worker_ds[i]
    if not bool(s["frame_mask"].any()):
        return ("skip_no_roi", sample_path)
    raw = {
        "frame_data": s["frame_data"],
        "frame_mask": s["frame_mask"],
        "seq_label": s["seq_label"],
        "label": int(s["label"]),
        "penetration_layer": int(s["penetration_layer"]),
        "layer_list": [int(x) for x in s["layer_list"]],
        "sample_path": s["sample_path"],
    }
    torch.save(raw, out_path)
    return ("ok", sample_path)


def _build_ds_kwargs(args: argparse.Namespace) -> dict:
    excl = args.exclude_json
    if excl:
        excl = os.path.abspath(os.path.normpath(excl.replace("\\", os.sep)))
    return {
        "samples_info_path": args.samples_info,
        "base_dir": args.base_dir,
        "target_size": (int(args.img_size), int(args.img_size)),
        "roi_size": int(args.roi_size),
        "grid": (8, 8),
        "pool_stats": ("mean", "std", "max"),
        "max_layers": args.max_layers,
        "max_frames_per_layer": int(args.max_frames_per_layer),
        "penetration_radius": int(args.penetration_radius),
        "exclude_json": excl,
        "final_roi_scale": float(args.final_roi_scale),
        "cc_min_area": int(args.cc_min_area),
        "cc_expand_ratio": float(args.cc_expand_ratio),
        "min_laser_pixels": int(args.min_laser_pixels),
        "min_laser_area_ratio": float(args.min_laser_area_ratio),
        "roi_window_side": int(args.roi_window_side),
        "roi_bright_min_ratio": float(args.roi_bright_min_ratio),
        "roi_gray_p95_min": float(args.roi_gray_p95_min),
        "use_color_cc_v2_geometry": (not bool(args.legacy_color_cc_geometry)),
        "precomputed_dir": None,
        "use_grayscale": bool(args.use_grayscale),
    }


def main():
    ap = argparse.ArgumentParser(description="Precompute hierarchical frame-layer features")
    ap.add_argument("--samples_info", type=str, default=None)
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="默认 grid_diff_tcn/cache_hierarchical_features")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="0=单进程顺序；>0 时用多进程池（每个 worker 内各建一份 Dataset）",
    )

    ap.add_argument("--img_size", type=int, default=DEFAULT_TARGET_WH[0])
    ap.add_argument("--roi_size", type=int, default=DEFAULT_ROI_SIZE)
    ap.add_argument("--max_frames_per_layer", type=int, default=8)
    ap.add_argument("--max_layers", type=int, default=None)
    ap.add_argument("--penetration_radius", type=int, default=2)

    ap.add_argument("--cc_min_area", type=int, default=DEFAULT_CC_MIN_AREA)
    ap.add_argument("--cc_expand_ratio", type=float, default=DEFAULT_CC_EXPAND_RATIO)
    ap.add_argument("--final_roi_scale", type=float, default=DEFAULT_FINAL_ROI_SCALE)
    ap.add_argument("--min_laser_pixels", type=int, default=DEFAULT_MIN_LASER_PIXELS, help="全图 HSV 亮区像素下限；0=关闭")
    ap.add_argument("--min_laser_area_ratio", type=float, default=DEFAULT_MIN_LASER_AREA_RATIO, help="全图亮区占 H*W 比例下限；0=关闭")
    ap.add_argument(
        "--roi_window_side",
        type=int,
        default=DEFAULT_ROI_WINDOW_SIDE,
        help="与 visualize_roi 一致：固定正方形边长；0=关闭（随检测框）",
    )
    ap.add_argument("--roi_bright_min_ratio", type=float, default=DEFAULT_ROI_BRIGHT_MIN_RATIO, help="letterbox 后 ROI 内亮区占比下限；0=关闭")
    ap.add_argument("--roi_gray_p95_min", type=float, default=DEFAULT_ROI_GRAY_P95_MIN, help="letterbox 后灰度 95%% 分位下限 [0,1]；0=关闭")
    ap.add_argument(
        "--legacy_color_cc_geometry",
        action="store_true",
        help="使用旧版 shrink(min边) 几何，而非 v2(max边 union)",
    )
    ap.add_argument(
        "--use_grayscale",
        action="store_true",
        default=False,
        help="使用灰度图预计算（默认False，即使用彩色图）",
    )
    ap.add_argument(
        "--exclude_json",
        type=str,
        default=os.path.join(_REPO_ROOT, "data_drilling", "no_laser_change_equalbox_full_mad00005_center_and_below.json"),
    )
    args = ap.parse_args()

    if args.samples_info is None:
        args.samples_info = os.path.join(_REPO_ROOT, "data_drilling", "samples_info_train.json")
    args.samples_info = os.path.normpath(args.samples_info.replace("\\", os.sep))
    if args.out_dir is None:
        args.out_dir = os.path.join(_GRID_ROOT, "cache_hierarchical_features")
    if not os.path.isfile(args.samples_info):
        raise FileNotFoundError(f"samples_info not found: {args.samples_info}")

    ds_kwargs = _build_ds_kwargs(args)
    from grid_diff_tcn.hier.frame_layer import HierarchicalFrameLayerDataset

    ds = HierarchicalFrameLayerDataset(**ds_kwargs)

    os.makedirs(args.out_dir, exist_ok=True)
    out_abs = os.path.abspath(args.out_dir)
    n = len(ds)
    nw = max(0, int(args.num_workers))
    print(f"precompute hierarchical: n={n}, out={out_abs}, num_workers={nw}")

    indices = list(range(n))
    skip_no_roi: list[str] = []
    ok = skip_exists = 0
    if nw <= 1:
        for i in tqdm(indices, desc="precompute_hier"):
            name = ds._precomputed_name_for_idx[i]
            out_path = os.path.join(args.out_dir, name)
            sp = str(ds.samples[i].get("sample_path", ""))
            if os.path.isfile(out_path):
                skip_exists += 1
                continue
            s = ds[i]
            if not bool(s["frame_mask"].any()):
                skip_no_roi.append(sp)
                continue
            raw = {
                "frame_data": s["frame_data"],
                "frame_mask": s["frame_mask"],
                "seq_label": s["seq_label"],
                "label": int(s["label"]),
                "penetration_layer": int(s["penetration_layer"]),
                "layer_list": [int(x) for x in s["layer_list"]],
                "sample_path": s["sample_path"],
            }
            torch.save(raw, out_path)
            ok += 1
    else:
        tasks = [(i, args.out_dir) for i in indices]
        chunksize = max(1, min(8, n // (nw * 8) or 1))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nw, initializer=_init_worker, initargs=(ds_kwargs,)) as pool:
            for tag, sp in tqdm(
                pool.imap_unordered(_compute_one, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="precompute_hier",
            ):
                if tag == "ok":
                    ok += 1
                elif tag == "skip_exists":
                    skip_exists += 1
                elif tag == "skip_no_roi":
                    skip_no_roi.append(sp)

    if skip_no_roi:
        skip_path = os.path.join(out_abs, "precompute_skipped_no_roi.txt")
        with open(skip_path, "w", encoding="utf-8") as f:
            f.write("\n".join(skip_no_roi) + "\n")
        print(f"wrote skip list ({len(skip_no_roi)}): {skip_path}")
    print(f"done: saved={ok}, skip_existing={skip_exists}, skip_no_roi={len(skip_no_roi)}")


if __name__ == "__main__":
    main()
