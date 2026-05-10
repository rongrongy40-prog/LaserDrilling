# -*- coding: utf-8 -*-
"""
Precompute DINOv3 features for hier/train.py with --use_dinov3.

Each sample saved as .pt contains:
  - frame_data: (T, F, dinov3_feat_dim)  DINOv3 CLS token features (768 for ViT-B)
  - frame_mask: (T, F)
  - seq_label:  (T,)
  - label, penetration_layer, layer_list, sample_path

Usage:
  # With GPU (fast, recommended):
  python -m grid_diff_tcn.hier.dinov3_precompute \
    --samples_info data_drilling/samples_info_train.json \
    --out_dir grid_diff_tcn/cache_dinov3_features_vitb \
    --dinov3_model vit_base \
    --dinov3_feat_dim 768 \
    --roi_size 224 \
    --num_workers 8 \
    --device cuda

  # With CPU (slow, for machines without GPU):
  python -m grid_diff_tcn.hier.dinov3_precompute \
    --samples_info data_drilling/samples_info_train.json \
    --out_dir grid_diff_tcn/cache_dinov3_features_vitb \
    --dinov3_model vit_small \
    --dinov3_feat_dim 384 \
    --roi_size 224 \
    --num_workers 4 \
    --device cpu
"""

from __future__ import annotations

import os
import argparse
import multiprocessing as mp
import torch
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_GRID_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, ".."))
_REPO_ROOT = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", ".."))
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
    DEFAULT_ROI_WINDOW_SIDE,
    DEFAULT_TARGET_WH,
    DEFAULT_ROI_SIZE,
)
from grid_diff_tcn.hier.frame_layer import DINOV3_MODELS, DINOV3_FEAT_DIMS, DINOV3_DEFAULT_MODEL

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# --- Multiprocessing worker state ---
_worker_ds = None


def _init_worker(
    ds_kwargs: dict,
    dinov3_model_name: str,
    dinov3_device: str,
    dinov3_feat_dim: int,
) -> None:
    """
    Initialize one DINOv3 extractor + HierarchicalDinoV3Dataset per worker process.

    We re-load the pretrained DINOv3 from disk in each subprocess to avoid
    sharing large model tensors across process boundaries.
    """
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

    from grid_diff_tcn.hier.frame_layer import DinoV3FeatureExtractor, HierarchicalDinoV3Dataset

    extractor = DinoV3FeatureExtractor(
        model_name=dinov3_model_name,
        pretrained=True,
        pool_strategy="cls",
        image_size=224,
        device=dinov3_device,
    )
    extractor.eval()
    # Ensure params are on CPU for multiprocessing safety
    extractor = extractor.to(torch.device("cpu"))
    extractor._device = torch.device("cpu")

    # Inject extractor into dataset kwargs
    ds_kwargs = dict(ds_kwargs)
    ds_kwargs["dinov3_extractor"] = extractor
    ds_kwargs["dinov3_feat_dim"] = dinov3_feat_dim

    _worker_ds = HierarchicalDinoV3Dataset(**ds_kwargs)


def _compute_one(args: tuple) -> tuple[str, str]:
    i, out_dir = args
    global _worker_ds
    if _worker_ds is None:
        raise RuntimeError("Worker not initialized. Call _init_worker first.")
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
        "target_size": (int(args.img_size), int(args.img_size)),
        "roi_size": int(args.roi_size),
        "max_layers": args.max_layers,
        "max_frames_per_layer": int(args.max_frames_per_layer),
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
        "_dinov3_target_size": int(args.roi_size),
    }


def main():
    ap = argparse.ArgumentParser(
        description="Precompute DINOv3 features for hierarchical training"
    )
    ap.add_argument("--samples_info", type=str, default=None)
    ap.add_argument(
        "--out_dir", type=str, default=None,
        help="default: grid_diff_tcn/cache_dinov3_features_<model>"
    )
    ap.add_argument(
        "--num_workers", type=int, default=4,
        help="0=single process; >0 uses multiprocessing pool"
    )

    # DINOv3 options
    ap.add_argument(
        "--dinov3_model", type=str, default=DINOV3_DEFAULT_MODEL,
        choices=list(DINOV3_MODELS.keys()),
        help="DINOv3 model size"
    )
    ap.add_argument(
        "--dinov3_feat_dim", type=int, default=None,
        help="DINOv3 feature dimension (auto-inferred from model if not set)"
    )
    ap.add_argument(
        "--device", type=str, default="cuda",
        choices=["cuda", "cpu"],
        help="device for DINOv3 feature extraction in workers"
    )
    ap.add_argument(
        "--roi_size", type=int, default=224,
        help="ROI crop size fed to DINOv3 (must be divisible by 16)"
    )

    # ROI / image options (same as precompute.py)
    ap.add_argument("--img_size", type=int, default=DEFAULT_TARGET_WH[0])
    ap.add_argument("--max_frames_per_layer", type=int, default=8)
    ap.add_argument("--max_layers", type=int, default=None)
    ap.add_argument("--cc_min_area", type=int, default=DEFAULT_CC_MIN_AREA)
    ap.add_argument("--cc_expand_ratio", type=float, default=DEFAULT_CC_EXPAND_RATIO)
    ap.add_argument("--final_roi_scale", type=float, default=DEFAULT_FINAL_ROI_SCALE)
    ap.add_argument("--min_laser_pixels", type=int, default=DEFAULT_MIN_LASER_PIXELS)
    ap.add_argument("--min_laser_area_ratio", type=float, default=DEFAULT_MIN_LASER_AREA_RATIO)
    ap.add_argument("--roi_window_side", type=int, default=DEFAULT_ROI_WINDOW_SIDE)
    ap.add_argument("--roi_bright_min_ratio", type=float, default=DEFAULT_ROI_BRIGHT_MIN_RATIO)
    ap.add_argument("--roi_gray_p95_min", type=float, default=DEFAULT_ROI_GRAY_P95_MIN)
    ap.add_argument("--legacy_color_cc_geometry", action="store_true")
    ap.add_argument("--use_grayscale", action="store_true", default=False)
    ap.add_argument(
        "--exclude_json", type=str,
        default=os.path.join(_REPO_ROOT, "data_drilling",
                             "no_laser_change_equalbox_full_mad00005_center_and_below.json"),
    )
    args = ap.parse_args()

    if args.samples_info is None:
        args.samples_info = os.path.join(_REPO_ROOT, "data_drilling", "samples_info_train.json")
    args.samples_info = os.path.normpath(args.samples_info.replace("\\", os.sep))
    if not os.path.isfile(args.samples_info):
        raise FileNotFoundError(f"samples_info not found: {args.samples_info}")

    if args.roi_size % 16 != 0:
        raise ValueError(f"--roi_size must be divisible by 16, got {args.roi_size}")

    feat_dim = args.dinov3_feat_dim or DINOV3_FEAT_DIMS[args.dinov3_model]
    if args.out_dir is None:
        args.out_dir = os.path.join(_GRID_ROOT, f"cache_dinov3_features_{args.dinov3_model}")

    os.makedirs(args.out_dir, exist_ok=True)
    out_abs = os.path.abspath(args.out_dir)

    ds_kwargs = _build_ds_kwargs(args)

    print(
        f"[dinov3_precompute] model={args.dinov3_model}, feat_dim={feat_dim}, "
        f"roi_size={args.roi_size}, device={args.device}, "
        f"out={out_abs}, num_workers={args.num_workers}"
    )

    nw = max(0, int(args.num_workers))

    # Quick dry-run: load one sample on main process to verify DINOv3 works
    print("[dinov3_precompute] Testing DINOv3 feature extraction on sample 0...")
    from grid_diff_tcn.hier.frame_layer import DinoV3FeatureExtractor, HierarchicalDinoV3Dataset
    test_extractor = DinoV3FeatureExtractor(
        model_name=args.dinov3_model,
        pretrained=True,
        pool_strategy="cls",
        image_size=224,
        device=args.device,
    )
    test_ds = HierarchicalDinoV3Dataset(
        dinov3_extractor=test_extractor,
        dinov3_feat_dim=feat_dim,
        samples_info_path=args.samples_info,
        roi_size=int(args.roi_size),
        target_size=(int(args.img_size), int(args.img_size)),
        max_layers=args.max_layers,
        max_frames_per_layer=int(args.max_frames_per_layer),
        exclude_json=ds_kwargs.get("exclude_json"),
        final_roi_scale=float(args.final_roi_scale),
        cc_min_area=int(args.cc_min_area),
        cc_expand_ratio=float(args.cc_expand_ratio),
        min_laser_pixels=int(args.min_laser_pixels),
        min_laser_area_ratio=float(args.min_laser_area_ratio),
        roi_window_side=int(args.roi_window_side),
        roi_bright_min_ratio=float(args.roi_bright_min_ratio),
        roi_gray_p95_min=float(args.roi_gray_p95_min),
        use_color_cc_v2_geometry=(not bool(args.legacy_color_cc_geometry)),
        precomputed_dir=None,
        use_grayscale=bool(args.use_grayscale),
        _dinov3_target_size=int(args.roi_size),
    )
    _ = test_ds[0]  # may return zero mask if no valid ROI, that's ok
    print(f"[dinov3_precompute] Dry-run OK. Feature dim confirmed: {feat_dim}")
    n_total = len(test_ds)
    del test_extractor, test_ds
    import gc
    gc.collect()

    indices = list(range(n_total))
    skip_no_roi: list[str] = []
    ok = skip_exists = 0

    if nw <= 1:
        # Single-process mode: reuse main-process extractor
        from grid_diff_tcn.hier.frame_layer import DinoV3FeatureExtractor, HierarchicalDinoV3Dataset
        main_extractor = DinoV3FeatureExtractor(
            model_name=args.dinov3_model,
            pretrained=True,
            pool_strategy="cls",
            image_size=224,
            device=args.device,
        )
        main_extractor.eval()
        main_extractor = main_extractor.to(torch.device("cpu"))
        main_extractor._device = torch.device("cpu")
        ds_kwargs["dinov3_extractor"] = main_extractor
        ds_kwargs["dinov3_feat_dim"] = feat_dim
        ds = HierarchicalDinoV3Dataset(**ds_kwargs)

        for i in tqdm(indices, desc="precompute_dinov3"):
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
        # Multiprocess mode: each worker loads its own DINOv3
        tasks = [(i, args.out_dir) for i in indices]
        chunksize = max(1, min(8, len(indices) // (nw * 8) or 1))
        ctx = mp.get_context("spawn")
        with ctx.Pool(
            processes=nw,
            initializer=_init_worker,
            initargs=(ds_kwargs, args.dinov3_model, args.device, feat_dim),
        ) as pool:
            for tag, sp in tqdm(
                pool.imap_unordered(_compute_one, tasks, chunksize=chunksize),
                total=len(tasks),
                desc="precompute_dinov3",
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
    print(
        f"done: saved={ok}, skip_existing={skip_exists}, "
        f"skip_no_roi={len(skip_no_roi)}, feat_dim={feat_dim}"
    )


if __name__ == "__main__":
    main()
