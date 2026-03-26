#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化版 ROI 与 8×8 网格可视化。

功能：从 samples_info 中抽取若干样本，每样本取若干张图，在图上绘制 CenterCrop 绿色大框与 8×8 网格，
      并保存为一张拼接图（如 roi_simple.png），用于核对裁剪与网格是否与训练/推理一致。
依赖：dataset.load_image_as_float, parse_layer_from_filename；samples_info.json；图片目录。
输出：--out 指定的 PNG 文件。
主要参数：--samples_info, --num_samples, --num_images, --crop_size, --grid, --out。
示例：python visualize_roi_simple.py --samples_info ../data_drilling/samples_info.json --out roi_simple.png
"""

import os
import sys
import json
import argparse
import random
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

from dataset import load_image_as_float, parse_layer_from_filename
from glob import glob
from collections import defaultdict


def draw_center_crop_grid(img_uint8, crop_size=480, grid=(8, 8)):
    """
    只画 CenterCrop 绿色大框 + 网格
    """
    h, w = img_uint8.shape[:2]
    out = img_uint8.copy()
    
    # CenterCrop
    crop = min(crop_size, h, w)
    y0 = (h - crop) // 2
    x0 = (w - crop) // 2
    
    # 绿色大框
    cv2.rectangle(out, (x0, y0), (x0 + crop, y0 + crop), (0, 255, 0), 2)
    cv2.putText(out, f"CenterCrop({crop})", (x0 + 5, y0 + 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    # 网格
    grid_r, grid_c = grid
    ph = crop // grid_r
    pw = crop // grid_c
    for i in range(1, grid_r):
        yy = y0 + i * ph
        cv2.line(out, (x0, yy), (x0 + crop, yy), (0, 200, 200), 1)
    for j in range(1, grid_c):
        xx = x0 + j * pw
        cv2.line(out, (xx, y0), (xx, y0 + crop), (0, 200, 200), 1)
    
    # 中心红点
    cx, cy = x0 + crop // 2, y0 + crop // 2
    cv2.circle(out, (cx, cy), 4, (0, 0, 255), -1)
    
    return out


def main():
    parser = argparse.ArgumentParser(description="简化 ROI 可视化：只保留 CenterCrop")
    parser.add_argument("--samples_info", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--num_images", type=int, default=2)
    parser.add_argument("--crop_size", type=int, default=480, help="CenterCrop 尺寸")
    parser.add_argument("--grid", type=int, default=8, help="网格边长")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="roi_simple.png")
    args = parser.parse_args()
    
    if args.samples_info is None:
        args.samples_info = os.path.join(SCRIPT_DIR, "..", "data_drilling", "samples_info.json")
    args.samples_info = os.path.normpath(args.samples_info.replace("\\", os.sep))
    
    with open(args.samples_info, "r", encoding="utf-8") as f:
        samples = json.load(f)
    
    valid_paths = [s.get("sample_path", "") for s in samples 
                   if s.get("sample_path") and os.path.isdir(s.get("sample_path", ""))]
    chosen = valid_paths[:args.num_samples]
    
    images_to_draw = []
    for sample_path in chosen:
        paths = sorted(glob(os.path.join(sample_path, "*.jpg")))
        by_layer = defaultdict(list)
        for p in paths:
            layer = parse_layer_from_filename(os.path.basename(p))
            if layer is not None:
                by_layer[layer].append(p)
        layers = sorted(by_layer.keys())
        
        for i, layer in enumerate(layers[:args.num_images]):
            if not by_layer[layer]:
                continue
            img = load_image_as_float(by_layer[layer][0], target_size=None)
            if img is None:
                continue
            img_uint8 = (np.clip(img, 0, 1) * 255).astype(np.uint8)
            if img_uint8.ndim == 2:
                img_uint8 = np.stack([img_uint8] * 3, axis=-1)
            elif img_uint8.ndim == 3:
                img_uint8 = cv2.cvtColor(img_uint8, cv2.COLOR_RGB2BGR) if HAS_CV2 else img_uint8
            
            drawn = draw_center_crop_grid(img_uint8, crop_size=args.crop_size, grid=(args.grid, args.grid))
            name = os.path.basename(sample_path)[:20]
            images_to_draw.append((drawn, f"{name} L{layer}"))
    
    # 拼图
    n = len(images_to_draw)
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    cell_h, cell_w = images_to_draw[0][0].shape[:2]
    
    # 缩放以适应屏幕
    max_canvas = 2400
    total_w = cols * cell_w
    total_h = rows * cell_h
    if total_w > max_canvas or total_h > max_canvas:
        scale = max_canvas / max(total_w, total_h)
        cell_w, cell_h = int(cell_w * scale), int(cell_h * scale)
    
    canvas = np.ones((rows * cell_h, cols * cell_w, 3), dtype=np.uint8) * 240
    for idx, (img, title) in enumerate(images_to_draw):
        img = cv2.resize(img, (cell_w, cell_h), interpolation=cv2.INTER_AREA)
        r, c = idx // cols, idx % cols
        y, x = r * cell_h, c * cell_w
        canvas[y:y+cell_h, x:x+cell_w] = img
        cv2.putText(canvas, title, (x + 5, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    out_path = os.path.join(SCRIPT_DIR, args.out)
    cv2.imwrite(out_path, canvas)
    print(f"已保存: {out_path}")
    print(f"绿色=CenterCrop({args.crop_size})，网格={args.grid}x{args.grid}")


if __name__ == "__main__":
    main()
