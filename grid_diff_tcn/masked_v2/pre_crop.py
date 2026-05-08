# -*- coding: utf-8 -*-
"""
预裁剪脚本：离线提取所有 ROI 并存为 .pt 文件。
每个孔的裁剪结果存为单个 .pt，内含：
    - frames: (T, F, 3, H, W) float32 tensor，0-1 归一化
    - mask:    (T, F) bool tensor，表示哪些位置有效
    - layers:  (T,) list[int]，每层的实际编号

使用方式（固定 box 策略）：
    python -m grid_diff_tcn.masked_v2.pre_crop \
        --samples_info data_drilling/samples_info_train_split.json \
        --cache_dir data_drilling/roi_cache \
        --skip_layers 30 \
        --num_anchor_frames 20 \
        --roi_size 64 \
        --max_frames 8 --max_workers 8

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
    _resize_rgb_letterbox,
)

# ---------------------- 裁剪逻辑（模块级，供多进程调用） ----------------------


def _clamp(val, lo, hi):
    return max(lo, min(hi, int(val)))


def _gather_anchor_boxes(
    sample_path: str,
    layer_start: int,
    num_anchor_frames: int,
    cc_min_area: int,
    cc_expand_ratio: float,
    final_roi_scale: float,
):
    """
    策略：
      1. 从 layer_start 层起按 frame 顺序扫描，检测 box
      2. 过滤异常大的框（> 中位数 * 2.5）
      3. 累计够 num_anchor_frames 个后计算并集；不够则用现有全部
    返回 (fixed_box | None, list_of_valid_boxes, h_img, w_img)
    """
    all_files = glob(os.path.join(sample_path, "*.jpg"))

    # 收集 layer >= layer_start 层，按 (layer, frame) 排序
    rows: list[tuple[int, int, str]] = []
    for p in all_files:
        fn = os.path.basename(p)
        parts = fn.replace(".jpg", "").split("_")
        if len(parts) < 2:
            continue
        try:
            frame = int(parts[-2])
            layer = int(parts[-1])
        except ValueError:
            continue
        if layer >= layer_start:
            rows.append((layer, frame, p))

    rows.sort()  # 按 (layer, frame) 排序

    valid_boxes: list[tuple] = []
    h_img, w_img = 0, 0

    for layer, frame, path in rows:
        img = load_image_as_float(path, None)
        if img is None:
            continue
        gray = to_grayscale(img)
        h_img, w_img = gray.shape[:2]
        box = _color_cc_resolve_box(
            rgb01=img,
            gray=gray,
            final_roi_scale=final_roi_scale,
            cc_min_area=cc_min_area,
            cc_expand_ratio=cc_expand_ratio,
            use_color_cc_v2_geometry=True,
            min_laser_pixels=0,
            min_laser_area_ratio=0.0,
            roi_window_side=None,
        )
        if box is None:
            continue

        valid_boxes.append(box)

        if len(valid_boxes) == num_anchor_frames:
            break

    if not valid_boxes:
        return None, [], h_img, w_img

    # 两遍过滤：先算中位数，再用 1.5x 阈值（比 2.5x 更严格）
    areas = np.array([(b[2]-b[0])*(b[3]-b[1]) for b in valid_boxes], dtype=float)
    med = np.median(areas) if len(areas) > 0 else 1.0
    valid_boxes = [b for b in valid_boxes
                   if (b[2]-b[0])*(b[3]-b[1]) <= med * 1.5]

    if not valid_boxes:
        return None, [], h_img, w_img

    x0 = min(b[0] for b in valid_boxes)
    y0 = min(b[1] for b in valid_boxes)
    x1 = max(b[2] for b in valid_boxes)
    y1 = max(b[3] for b in valid_boxes)
    fixed_box = (x0, y0, x1, y1)

    return fixed_box, valid_boxes, h_img, w_img


def _crop_with_fixed_box(img_rgb01, box, roi_size):
    """用固定 box 裁剪并 resize 到 roi_size。"""
    h, w = img_rgb01.shape[:2]
    x0, y0, x1, y1 = box
    if x1 <= x0 or y1 <= y0:
        return None
    x0, y0 = _clamp(x0, 0, w), _clamp(y0, 0, h)
    x1, y1 = _clamp(x1, 0, w), _clamp(y1, 0, h)
    if x1 <= x0 or y1 <= y0:
        return None
    patch = img_rgb01[y0:y1, x0:x1]
    if patch.size == 0:
        return None
    roi = _resize_rgb_letterbox(patch, (roi_size, roi_size), pad_value=0.0)
    return roi.astype(np.float32)


def _process_one_sample(args: tuple) -> dict:
    """
    处理单个孔：计算固定 box，裁剪所有帧，存为 .pt。
    返回 {"sample_path": ..., "status": "ok"|"skipped"|"error", "detail": ...}
    """
    (sample_path, cache_path, roi_size, max_frames_per_layer,
     max_layers, skip_layers, num_anchor_frames,
     cc_min_area, cc_expand_ratio, final_roi_scale) = args

    try:
        # 1. 收集该孔所有图片，按层分组
        by_layer: dict[int, list] = defaultdict(list)
        for p in glob(os.path.join(sample_path, "*.jpg")):
            fn = os.path.basename(p)
            parts = fn.replace(".jpg", "").split("_")
            if len(parts) < 2:
                continue
            try:
                frame = int(parts[-2])
                layer = int(parts[-1])
            except ValueError:
                continue
            if layer > skip_layers:
                by_layer[layer].append((frame, p))

        layer_list = sorted(by_layer.keys())
        if max_layers and len(layer_list) > max_layers:
            layer_list = layer_list[:max_layers]

        if len(layer_list) == 0:
            return {"sample_path": sample_path, "status": "skipped",
                    "detail": "no layers after skip"}

        T = len(layer_list)
        F = max_frames_per_layer

        # 2. 累计够 20 个有效 ROI 后计算并集作为固定 box
        fixed_box, anchor_boxes, h_img, w_img = _gather_anchor_boxes(
            sample_path=sample_path,
            layer_start=skip_layers + 1,
            num_anchor_frames=num_anchor_frames,
            cc_min_area=cc_min_area,
            cc_expand_ratio=cc_expand_ratio,
            final_roi_scale=final_roi_scale,
        )
        if fixed_box is None:
            return {"sample_path": sample_path, "status": "skipped",
                    "detail": f"failed to resolve fixed box (found {len(anchor_boxes)} valid anchors < {num_anchor_frames})"}

        # 3. 用固定 box 裁剪所有帧
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
                img = load_image_as_float(p, None)
                if img is None:
                    continue
                roi = _crop_with_fixed_box(img, fixed_box, roi_size)
                if roi is None:
                    continue
                # roi is (H, W, 3) float32 [0,1], convert to (3, H, W)
                data[ti, fi] = roi.transpose(2, 0, 1)
                mask[ti, fi] = True

        # 4. 如果所有帧都裁剪失败，跳过
        if not mask.any():
            return {"sample_path": sample_path, "status": "skipped",
                    "detail": "all crops failed"}

        # 5. 写 .pt（uint8 存储，体积比 float32 小 4×）
        frames_uint8 = (data * 255).round().astype(np.uint8)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp_path = cache_path + ".tmp"
        torch.save({
            "frames": torch.from_numpy(frames_uint8),
            "mask": torch.from_numpy(mask),
            "layers": layer_list,
            "sample_path": sample_path,
            "fixed_box": fixed_box,
            "anchor_boxes": anchor_boxes,
            "img_size": (h_img, w_img),
            "_uint8": True,      # 版本标记：新格式为 uint8
            "_roi_size": roi_size,
        }, tmp_path)
        os.replace(tmp_path, cache_path)

        n_valid = int(mask.sum())
        return {"sample_path": sample_path, "status": "ok",
                "detail": f"T={T} F={F} valid={n_valid} box={fixed_box} anchors={len(anchor_boxes)}"}

    except Exception as e:
        return {"sample_path": sample_path, "status": "error",
                "detail": str(e)}


# ---------------------- 主函数 ----------------------

def main():
    parser = argparse.ArgumentParser(description="预裁剪 ROI 图片到 .pt 缓存（固定 box 策略）")
    parser.add_argument("--samples_info", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, required=True,
                        help="缓存输出目录，一孔一 .pt 文件")
    parser.add_argument("--roi_size", type=int, default=64,
                        help="最终 resize 尺寸（默认 64，大幅节省缓存体积）")
    parser.add_argument("--max_frames", type=int, default=8,
                        help="每层最多保留帧数")
    parser.add_argument("--max_layers", type=int, default=None,
                        help="每孔最多层数（None=不限）")
    parser.add_argument("--skip_layers", type=int, default=30,
                        help="跳过前 N 层（默认 30，不考虑穿透前）")
    parser.add_argument("--num_anchor_frames", type=int, default=20,
                        help="层31起取连续多少帧来算固定 box（默认 20）")
    parser.add_argument("--final_roi_scale", type=float, default=0.85,
                        help="单帧检测时的缩放系数（传给 _color_cc_box）")
    parser.add_argument("--cc_min_area", type=int, default=12,
                        help="连通域最小面积")
    parser.add_argument("--cc_expand_ratio", type=float, default=0.2,
                        help="box 扩展比例（传给 _color_cc_box）")
    parser.add_argument("--max_workers", type=int, default=None,
                        help="并行进程数，默认为 CPU 核数")
    parser.add_argument("--overwrite", action="store_true",
                        help="重新生成已存在的 .pt 文件")
    parser.add_argument("--dry_run", action="store_true",
                        help="只扫描，不写文件")
    args = parser.parse_args()

    max_workers = args.max_workers or max(1, cpu_count() - 1)

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
        rel = os.path.relpath(sample_path, os.getcwd())
        safe = rel.replace(os.sep, "_").replace("/", "_")
        cache_path = os.path.join(args.cache_dir, f"{safe}.pt")

        if os.path.exists(cache_path) and not args.overwrite:
            continue

        tasks.append((
            sample_path, cache_path,
            args.roi_size, args.max_frames, args.max_layers,
            args.skip_layers, args.num_anchor_frames,
            args.cc_min_area, args.cc_expand_ratio, args.final_roi_scale,
        ))

    print(f"[pre_crop] 样本数: {len(samples)}, 待处理: {len(tasks)}, "
          f"缓存目录: {args.cache_dir}, 并行进程: {max_workers}")
    print(f"[pre_crop] roi_size={args.roi_size}  skip_layers={args.skip_layers}  "
          f"anchor_frames={args.num_anchor_frames}  "
          f"max_frames={args.max_frames}  max_layers={args.max_layers}")

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
