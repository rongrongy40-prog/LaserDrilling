# -*- coding: utf-8 -*-
"""
Precompute hierarchical per-hole features for hier/train.py.

Each sample saved as .pt contains:
  - frame_data: (T,F,64)
  - frame_mask: (T,F)
  - seq_label:  (T,)
  - label, penetration_layer, layer_list, sample_path
"""

import os
import argparse
import multiprocessing as mp
import torch

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


def _compute_one(args: tuple) -> str:
    i, out_dir, by_name = args
    global _worker_ds
    assert _worker_ds is not None
    name = _worker_ds._precomputed_name_for_idx[i] if by_name else f"{i}.pt"
    out_path = os.path.join(out_dir, name)
    if os.path.isfile(out_path):
        return "skip"
    s = _worker_ds[i]
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
    return "ok"


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
        "max_layers": args.max_layers,
        "max_frames_per_layer": int(args.max_frames_per_layer),
        "penetration_radius": int(args.penetration_radius),
        "exclude_json": excl,
        "final_roi_scale": float(args.final_roi_scale),
        "cc_min_area": int(args.cc_min_area),
        "cc_expand_ratio": float(args.cc_expand_ratio),
        "min_laser_pixels": int(args.min_laser_pixels),
        "min_laser_area_ratio": float(args.min_laser_area_ratio),
        "roi_window_side": (int(args.roi_window_side) if args.roi_window_side is not None else None),
        "roi_bright_min_ratio": float(args.roi_bright_min_ratio),
        "roi_gray_p95_min": float(args.roi_gray_p95_min),
        "use_color_cc_v2_geometry": (not bool(args.legacy_color_cc_geometry)),
        "use_hole_anchor_box": bool(getattr(args, "use_hole_anchor_box", False)),
        "hole_anchor_num_images": int(getattr(args, "hole_anchor_num_images", 10)),
    }


def main():
    ap = argparse.ArgumentParser(description="Precompute hierarchical frame-layer features")
    ap.add_argument("--samples_info", type=str, default=None)
    ap.add_argument("--base_dir", type=str, default=None)
    ap.add_argument("--out_dir", type=str, default=None, help="默认 grid_diff_tcn/cache_hierarchical_features")
    ap.add_argument("--by_name", action="store_true", help="save as unique sample basename .pt")
    ap.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="0=单进程顺序；>0 时用多进程池（每个 worker 内各建一份 Dataset）",
    )

    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--roi_size", type=int, default=96)
    ap.add_argument("--max_frames_per_layer", type=int, default=8)
    ap.add_argument("--max_layers", type=int, default=None)
    ap.add_argument("--penetration_radius", type=int, default=2)

    ap.add_argument("--cc_min_area", type=int, default=12)
    ap.add_argument("--cc_expand_ratio", type=float, default=0.2)
    ap.add_argument("--final_roi_scale", type=float, default=0.85)
    ap.add_argument("--min_laser_pixels", type=int, default=0, help="全图 HSV 亮区像素下限；0=关闭")
    ap.add_argument("--min_laser_area_ratio", type=float, default=0.0, help="全图亮区占 H*W 比例下限；0=关闭")
    ap.add_argument("--roi_window_side", type=int, default=None, help="以 color_cc 中心裁固定正方形边长；不设则沿用矩形框")
    ap.add_argument("--roi_bright_min_ratio", type=float, default=0.0, help="letterbox 后 ROI 内亮区占比下限；0=关闭")
    ap.add_argument("--roi_gray_p95_min", type=float, default=0.0, help="letterbox 后灰度 95%% 分位下限 [0,1]；0=关闭")
    ap.add_argument(
        "--legacy_color_cc_geometry",
        action="store_true",
        help="使用旧版 shrink(min边) 几何，而非 v2(max边 union)",
    )
    ap.add_argument("--use_hole_anchor_box", action="store_true", help="每孔前 N 张定锚框，全孔复用")
    ap.add_argument("--hole_anchor_num_images", type=int, default=10)
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
    if nw <= 1:
        for i in tqdm(indices, desc="precompute_hier"):
            name = ds._precomputed_name_for_idx[i] if args.by_name else f"{i}.pt"
            out_path = os.path.join(args.out_dir, name)
            if os.path.isfile(out_path):
                continue
            s = ds[i]
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
    else:
        tasks = [(i, args.out_dir, bool(args.by_name)) for i in indices]
        chunksize = max(1, min(8, n // (nw * 8) or 1))
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=nw, initializer=_init_worker, initargs=(ds_kwargs,)) as pool:
            for _ in tqdm(
                pool.imap_unordered(_compute_one, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="precompute_hier",
            ):
                pass

    print("done")


if __name__ == "__main__":
    main()
