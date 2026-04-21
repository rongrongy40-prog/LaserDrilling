# -*- coding: utf-8 -*-
"""
预裁剪脚本：离线提取所有 ROI 并存为 .pt 文件。
每个孔的裁剪结果存为单个 .pt，内含：
    - frames: (T, F, 3, H, W) float32 tensor，0-1 归一化
    - mask:    (T, F) bool tensor，表示哪些位置有效
    - layers:  (T,) list[int]，每层的实际编号

使用方式：
    python pre_crop.py --samples_info data_drilling/samples_info_train_split.json \
                       --cache_dir data_drilling/roi_cache \
                       --roi_size 224 --max_frames 8 --max_workers 8

cache_dir 结构：
    cache_dir/
    ├── 10-1_2024_10_01_01_00_01_443.pt   # 一孔一文件
    ├── 10-2_2024_10_01_02_00_01_443.pt
    └── ...

生成完后删除 .pt 文件对应原图已删除的孔，重新运行即可。
"""

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from glob import glob
from multiprocessing import Pool, cpu_count
from pathlib import Path
from typing import Any

import numpy as np
import torch

# 确保能 import grid_diff_tcn
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from grid_diff_tcn.common.image_ops import (
    load_image_as_float,
    to_grayscale,
    _color_cc_resolve_box,
    _crop,
    _resize_rgb_letterbox,
)
from grid_diff_tcn.common.roi_crop_defaults import norm_roi_window_side

# ---------------------- 裁剪逻辑（模块级，供多进程调用） ----------------------


def _crop_one_image(
    img_path: str,
    roi_size: int,
    final_roi_scale: float,
    cc_min_area: int,
    cc_expand_ratio: float,
    min_laser_pixels: int,
    min_laser_area_ratio: float,
    roi_window_side: int,
    use_color_cc_v2_geometry: bool,
) -> np.ndarray | None:
    """裁剪单张图，返回 float32 [0,1] (H,W,3) 或 None。"""
    img = load_image_as_float(img_path, (roi_size, roi_size))
    if img is None:
        return None
    gray = to_grayscale(img)
    box = _color_cc_resolve_box(
        rgb01=img,
        gray=gray,
        final_roi_scale=final_roi_scale,
        cc_min_area=cc_min_area,
        cc_expand_ratio=cc_expand_ratio,
        use_color_cc_v2_geometry=use_color_cc_v2_geometry,
        min_laser_pixels=min_laser_pixels,
        min_laser_area_ratio=min_laser_area_ratio,
        roi_window_side=roi_window_side,
    )
    if box is None:
        return None
    roi = _crop(img, box)
    if roi.size == 0:
        return None
    roi = _resize_rgb_letterbox(roi, (roi_size, roi_size), pad_value=0.0)
    return roi.astype(np.float32)


def _process_one_sample(args: tuple) -> dict:
    """
    处理单个孔：收集所有帧的裁剪结果，存为 .pt。
    返回 {"sample_path": ..., "status": "ok"|"skipped"|"error", "detail": ...}
    """
    (sample_path, cache_path, roi_size, max_frames_per_layer,
     max_layers, final_roi_scale, cc_min_area, cc_expand_ratio,
     min_laser_pixels, min_laser_area_ratio, roi_window_side,
     use_color_cc_v2_geometry) = args

    try:
        # 1. 收集该孔所有图片，按层分组
        by_layer: dict[int, list] = defaultdict(list)
        for p in glob(os.path.join(sample_path, "*.jpg")):
            fn = os.path.basename(p)
            # 解析文件名：frame_layer 格式
            parts = fn.replace(".jpg", "").split("_")
            if len(parts) < 2:
                continue
            try:
                frame = int(parts[0])
                layer = int(parts[1])
            except ValueError:
                continue
            by_layer[layer].append((frame, p))

        layer_list = sorted(by_layer.keys())
        if max_layers and len(layer_list) > max_layers:
            layer_list = layer_list[:max_layers]

        if not layer_list:
            return {"sample_path": sample_path, "status": "skipped",
                    "detail": "no layers"}

        T = len(layer_list)
        F = max_frames_per_layer

        # 2. 为每层采样帧并裁剪
        data = np.zeros((T, F, 3, roi_size, roi_size), dtype=np.float32)
        mask = np.zeros((T, F), dtype=bool)

        for ti, ly in enumerate(layer_list):
            items = sorted(by_layer[ly], key=lambda x: x[0])
            if len(items) <= F:
                picks = [p for _, p in items]
            else:
                idx = np.linspace(0, len(items) - 1, F, dtype=int)
                picks = [items[i][1] for i in idx]

            for fi, p in enumerate(picks[:F]):
                roi = _crop_one_image(
                    p, roi_size, final_roi_scale, cc_min_area, cc_expand_ratio,
                    min_laser_pixels, min_laser_area_ratio, roi_window_side,
                    use_color_cc_v2_geometry,
                )
                if roi is None:
                    continue
                # roi is (H, W, 3) float32 [0,1], convert to (3, H, W)
                data[ti, fi] = roi.transpose(2, 0, 1)
                mask[ti, fi] = True

        # 3. 如果所有帧都裁剪失败，跳过
        if not mask.any():
            return {"sample_path": sample_path, "status": "skipped",
                    "detail": "all crops failed"}

        # 4. 写 .pt
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        torch.save({
            "frames": torch.from_numpy(data),
            "mask": torch.from_numpy(mask),
            "layers": layer_list,
            "sample_path": sample_path,
        }, tmp_path)
        os.replace(tmp_path, cache_path)

        n_valid = int(mask.sum())
        return {"sample_path": sample_path, "status": "ok",
                "detail": f"T={T} F={F} valid={n_valid}"}

    except Exception as e:
        return {"sample_path": sample_path, "status": "error",
                "detail": str(e)}


# ---------------------- 主函数 ----------------------

def main():
    parser = argparse.ArgumentParser(description="预裁剪 ROI 图片到 .pt 缓存")
    parser.add_argument("--samples_info", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="缓存输出目录，一孔一 .pt 文件")
    parser.add_argument("--roi_size", type=int, default=224)
    parser.add_argument("--max_frames", type=int, default=8,
                        help="每层最多保留帧数")
    parser.add_argument("--max_layers", type=int, default=None,
                        help="每孔最多层数（None=不限）")
    parser.add_argument("--final_roi_scale", type=float, default=0.85)
    parser.add_argument("--cc_min_area", type=int, default=12)
    parser.add_argument("--cc_expand_ratio", type=float, default=0.2)
    parser.add_argument("--min_laser_pixels", type=int, default=0)
    parser.add_argument("--min_laser_area_ratio", type=float, default=0.0)
    parser.add_argument("--roi_window_side", type=int, default=None)
    parser.add_argument("--use_color_cc_v2_geometry", type=lambda x: x.lower() == "true",
                        default=True)
    parser.add_argument("--max_workers", type=int, default=None,
                        help="并行进程数，默认为 CPU 核数")
    parser.add_argument("--overwrite", action="store_true",
                        help="重新生成已存在的 .pt 文件")
    parser.add_argument("--dry_run", action="store_true",
                        help="只扫描，不写文件")
    args = parser.parse_args()

    max_workers = args.max_workers or max(1, cpu_count() - 1)
    roi_window_side = norm_roi_window_side(args.roi_window_side)

    # 读取样本列表
    with open(args.samples_info) as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("Categories", [])
    samples = raw if isinstance(raw, list) else []

    # 建立 sample_path -> cache_path 映射
    tasks = []
    for s in samples:
        sample_path = s.get("sample_path", "")
        if not sample_path:
            continue
        # cache 文件名：用原始路径的哈希或 basename
        # 保证不同样本集写到同一 cache 时不冲突
        rel = os.path.relpath(sample_path, os.getcwd())
        safe = rel.replace(os.sep, "_").replace("/", "_")
        cache_path = os.path.join(args.cache_dir, f"{safe}.pt")

        if os.path.exists(cache_path) and not args.overwrite:
            continue  # 已存在，跳过

        tasks.append((
            sample_path, cache_path,
            args.roi_size, args.max_frames, args.max_layers,
            args.final_roi_scale, args.cc_min_area, args.cc_expand_ratio,
            args.min_laser_pixels, args.min_laser_area_ratio,
            roi_window_side, args.use_color_cc_v2_geometry,
        ))

    print(f"[pre_crop] 样本数: {len(samples)}, 待处理: {len(tasks)}, "
          f"缓存目录: {args.cache_dir}, 并行进程: {max_workers}")
    print(f"[pre_crop] roi_size={args.roi_size} max_frames={args.max_frames} "
          f"max_layers={args.max_layers}")

    if args.dry_run:
        print("[pre_crop] dry_run 模式，只打印，不写文件")
        for t in tasks[:10]:
            print(f"  -> {t[0]}")
        print(f"  ... ({len(tasks)} tasks total)")
        return

    if not tasks:
        print("[pre_crop] 所有样本已缓存完成，无需处理")
        return

    t0 = time.time()
    ok = skipped = error = 0

    with Pool(max_workers) as pool:
        from tqdm import tqdm
        for res in tqdm(pool.imap_unordered(_process_one_sample, tasks,
                                              chunksize=max(1, len(tasks) // max_workers // 4)),
                        total=len(tasks), desc="预裁剪 ROI"):
            if res["status"] == "ok":
                ok += 1
            elif res["status"] == "skipped":
                skipped += 1
            else:
                error += 1
                print(f"\n  ERROR {res['sample_path']}: {res['detail']}")

    elapsed = time.time() - t0
    total = ok + skipped + error
    print(f"\n[pre_crop] 完成！({total} 样本, {elapsed:.0f}s, "
          f"ok={ok} skip={skipped} err={error})")


if __name__ == "__main__":
    main()
