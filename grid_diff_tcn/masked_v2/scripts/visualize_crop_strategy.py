# -*- coding: utf-8 -*-
"""
可视化脚本：展示新 ROI 裁剪策略（逐孔固定 box）
策略：
  1. 跳过前 30 层（穿透前不考虑）
  2. 从第 31 层起，取连续 20 层
  3. 对每层分别检测 box，取所有 box 的并集（最大范围）
  4. 以并集 box 中心为圆心，取 max(w,h)*1.2 作为正方形边长
  5. resize 成 64x64

使用方式：
  python visualize_crop_strategy.py [--hole <sample_path>]
"""
import argparse
import os
import sys
from glob import glob
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from grid_diff_tcn.common.image_ops import load_image_as_float

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

# 颜色定义
COLOR_MAX = (0, 200, 0)     # 绿色 — 并集 box = 最终裁剪框
COLOR_CENTER = (200, 100, 255)  # 粉色 — 中心点


def _clamp(val, lo, hi):
    return max(lo, min(hi, val))


def _to_u8(rgb01):
    return (np.clip(rgb01, 0.0, 1.0) * 255).astype(np.uint8)


def collect_frames(sample_path, layer_start, num_layers=3):
    """收集 layer_start ~ layer_start+num_layers-1 层的所有帧，按 (layer, frame) 排序。"""
    all_files = glob(os.path.join(sample_path, "*.jpg"))
    rows = []
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
        if layer_start <= layer <= layer_start + num_layers - 1:
            rows.append((layer, frame, p))
    rows.sort()
    return rows


def compute_fixed_box(frames_list, cc_min_area=12,
                      cc_expand_ratio=0.2, final_roi_scale=0.85):
    """
    核心策略：
    - 对层31起帧依次检测 box，累计够 20 个有效 ROI
    - 过滤异常大的 box（> 中位数 * 1.5）
    - 取并集作为固定裁剪框（直接取并集，不再扩）
    返回：fixed_box, list_of_valid_boxes, 原图尺寸
    """
    import cv2
    from grid_diff_tcn.common.image_ops import _color_cc_box

    valid_boxes = []
    h_img, w_img = 0, 0

    for layer, frame, path in frames_list:
        img = load_image_as_float(path, None)
        if img is None:
            continue
        gray = (0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]).astype(np.float32)
        h_img, w_img = gray.shape[:2]
        box = _color_cc_box(
            img, gray,
            final_roi_scale=final_roi_scale,
            cc_min_area=cc_min_area,
            cc_expand_ratio=cc_expand_ratio,
            use_v2_geometry=True,
        )
        if box is None:
            continue
        valid_boxes.append(box)

        if len(valid_boxes) == 20:
            break

    if not valid_boxes:
        return None, [], h_img, w_img

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
    fixed = (x0, y0, x1, y1)

    return fixed, valid_boxes, h_img, w_img


def draw_boxes(img_u8, union):
    """在图像上绘制 box 辅助线。"""
    vis = img_u8.copy()

    # 并集 box（绿色粗线）
    if union is not None:
        x0, y0, x1, y1 = union
        cv2.rectangle(vis, (x0, y0), (x1, y1), COLOR_MAX, 2)
        cx = int((x0+x1)/2)
        cy = int((y0+y1)/2)
        cv2.circle(vis, (cx, cy), 6, COLOR_CENTER, -1)

    return vis


def main():
    parser = argparse.ArgumentParser(description="可视化 ROI 裁剪策略")
    parser.add_argument("--hole", type=str,
                        default="/home/student2025/wudf2025/dinov3-main/data_drilling/train/3/35/10-1_2024_11_28_02_50_01_443",
                        help="孔的 sample_path")
    parser.add_argument("--layer_start", type=int, default=31)
    parser.add_argument("--output_dir", type=str,
                        default="/home/student2025/wudf2025/dinov3-main/grid_diff_tcn/masked_v2/visualize_crop")
    args = parser.parse_args()

    sample_path = args.hole
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    print(f"孔: {sample_path}")
    print(f"层31起累计 20 个有效 ROI 后算并集")

    # 1. 收集层31起所有帧
    frames = collect_frames(sample_path, args.layer_start)
    print(f"实际找到 {len(frames)} 帧")
    for ly, fr, p in frames[:5]:
        print(f"  layer={ly} frame={fr}")

    if not frames:
        print("未找到任何帧，请检查路径或层号范围。")
        return

    # 2. 计算固定 box（累计够 20 个有效 ROI 后取并集）
    fixed_box, anchor_boxes, h_img, w_img = compute_fixed_box(frames)

    if fixed_box is None:
        print("无法检测到足够有效 ROI，请检查图像内容。")
        return

    print(f"\n原图尺寸: {w_img}x{h_img}")
    print(f"有效 anchor 数: {len(anchor_boxes)}")
    print(f"固定 box: {fixed_box}")
    bw = fixed_box[2] - fixed_box[0]
    bh = fixed_box[3] - fixed_box[1]
    print(f"固定 box 尺寸: {bw}x{bh}")

    # 3. 生成可视化：只画并集框
    save_paths = []
    for layer, frame, path in frames:
        img_rgb = load_image_as_float(path, None)
        if img_rgb is None:
            continue
        img_u8 = _to_u8(img_rgb)

        vis = draw_boxes(img_u8, fixed_box)

        # 标注
        cv2.putText(vis, f"L{layer}|F{frame}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

        out_path = os.path.join(output_dir, f"vis_L{layer:03d}_F{frame:05d}.jpg")
        cv2.imwrite(out_path, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        save_paths.append(out_path)

    print(f"\n可视化图已保存到: {output_dir}/ ({len(save_paths)} 张)")

    # 4. 最终裁剪预览
    preview_path = os.path.join(output_dir, "final_crop_preview.jpg")
    if frames:
        _, _, first_path = frames[0]
        img_rgb = load_image_as_float(first_path, None)
        if img_rgb is not None:
            x0, y0, x1, y1 = fixed_box
            patch = img_rgb[y0:y1, x0:x1]
            patch_u8 = _to_u8(patch)
            patch_128 = cv2.resize(patch_u8, (128, 128), interpolation=cv2.INTER_CUBIC)
            cv2.imwrite(preview_path, cv2.cvtColor(patch_128, cv2.COLOR_RGB2BGR))
            print(f"最终 128x128 裁剪预览: {preview_path}")

    # 5. 拼图总览
    if HAS_MATPLOTLIB and save_paths:
        try:
            n = min(len(save_paths), 12)
            cols = 4
            rows = (n + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 4))
            axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes.flatten()
            for i, sp in enumerate(save_paths[:n]):
                img = cv2.imread(sp)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[i].imshow(img)
                axes[i].axis("off")
            for i in range(n, len(axes)):
                axes[i].axis("off")
            hole_name = os.path.basename(sample_path)
            fig.suptitle(
                f"ROI Crop Strategy | {hole_name}\n"
                f"Layers ≥{args.layer_start} | anchors={len(anchor_boxes)} | box={fixed_box}",
                fontsize=11,
            )
            overview_path = os.path.join(output_dir, "overview.jpg")
            fig.savefig(overview_path, dpi=120, bbox_inches="tight")
            print(f"总览图: {overview_path}")
            plt.close(fig)
        except Exception as e:
            print(f"matplotlib 拼图生成失败: {e}")

    print("\n策略：绿色粗线 = 并集 box = 最终裁剪框")


if __name__ == "__main__":
    main()
