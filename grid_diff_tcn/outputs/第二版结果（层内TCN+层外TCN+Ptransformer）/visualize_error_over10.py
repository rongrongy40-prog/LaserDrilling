#!/usr/bin/env python3
import argparse
import csv
import os
import sys

import numpy as np

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grid_diff_tcn.common.roi_crop_defaults import (
    DEFAULT_CC_EXPAND_RATIO,
    DEFAULT_CC_MIN_AREA,
    DEFAULT_FINAL_ROI_SCALE,
    DEFAULT_ROI_SIZE,
    DEFAULT_ROI_WINDOW_SIDE,
    DEFAULT_TARGET_WH,
    DEFAULT_USE_COLOR_CC_V2_GEOMETRY,
    norm_roi_window_side,
)
from grid_diff_tcn.common.image_ops import (
    HAS_CV2,
    color_cc_extract_gray_letterbox,
    load_image_as_float,
    to_grayscale,
    _color_cc_resolve_box,
    parse_frame_layer_from_filename,
    _crop,
    _resize_rgb_letterbox,
)


def _imwrite(path: str, rgb01: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    if HAS_CV2:
        import cv2
        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    else:
        from PIL import Image
        Image.fromarray(x).save(path)


def _draw_box(rgb01: np.ndarray, box: tuple | None, thickness: int = 2, color: tuple = (0.2, 1.0, 0.2)) -> np.ndarray:
    out = np.clip(rgb01, 0.0, 1.0).copy()
    if box is None:
        return out
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = out.shape[:2]
    x0, x1 = max(0, min(w - 1, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h - 1, y0)), max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return out
    c = np.array(color, dtype=np.float32)
    t = max(1, int(thickness))
    out[y0 : y0 + t, x0:x1] = c
    out[y1 - t : y1, x0:x1] = c
    out[y0:y1, x0 : x0 + t] = c
    out[y0:y1, x1 - t : x1] = c
    return out


def _draw_text(rgb01: np.ndarray, text: str, top: int = 10, left: int = 10, font_scale: float = 0.7, thickness: int = 2, bg: bool = True) -> np.ndarray:
    out = rgb01.copy()
    if HAS_CV2:
        import cv2
        h, w = out.shape[:2]
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        if bg:
            cv2.rectangle(out, (left, top - th - 4), (left + tw + 4, top + baseline), (0, 0, 0), -1)
            cv2.putText(out, text, (left + 2, top), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (1, 1, 1), thickness)
        else:
            cv2.putText(out, text, (left, top), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (1, 1, 1), thickness)
    return out


def _gray_to_rgb01(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim == 3:
        g = g[..., 0]
    g = np.clip(g, 0.0, 1.0)
    return np.repeat(g[:, :, None], 3, axis=2)


def _make_contact_sheet(panels, cols: int, pad: int = 6, bg: float = 0.0) -> np.ndarray:
    if not panels:
        return np.zeros((8, 8, 3), dtype=np.float32)
    cols = max(1, int(cols))
    rows = int(np.ceil(len(panels) / cols))
    ph, pw = panels[0].shape[:2]
    out = np.full((rows * ph + (rows + 1) * pad, cols * pw + (cols + 1) * pad, 3), float(bg), dtype=np.float32)
    for i, p in enumerate(panels):
        r, c = divmod(i, cols)
        y0 = pad + r * (ph + pad)
        x0 = pad + c * (pw + pad)
        out[y0 : y0 + ph, x0 : x0 + pw] = p
    return out


def _pick_evenly(paths: list, k: int) -> list:
    if not paths:
        return []
    paths = sorted(paths)
    if len(paths) <= k:
        return paths
    idx = np.linspace(0, len(paths) - 1, int(k), dtype=int)
    return [paths[i] for i in idx]


def get_layer_images(hole_dir: str, layers_to_show: list, target_size, roi_size, final_roi_scale, cc_min_area, cc_expand_ratio, roi_window_side, use_color_cc_v2_geometry):
    from glob import glob
    all_images = glob(os.path.join(hole_dir, "*.jpg"))
    
    layer_to_images = {}
    for img_path in all_images:
        frame_num, layer_num = parse_frame_layer_from_filename(img_path)
        if layer_num is not None:
            if layer_num not in layer_to_images:
                layer_to_images[layer_num] = []
            layer_to_images[layer_num].append((frame_num, img_path))
    
    for ly in layer_to_images:
        layer_to_images[ly] = sorted(layer_to_images[ly], key=lambda x: x[0])
    
    roi_window_side = norm_roi_window_side(int(roi_window_side))
    panels = []
    
    for layer_idx, layer_num in enumerate(layers_to_show):
        if layer_num not in layer_to_images:
            continue
        imgs = layer_to_images[layer_num]
        
        picks = _pick_evenly([x[1] for x in imgs], min(6, len(imgs)))
        
        for img_path in picks[:3]:
            rgb01 = load_image_as_float(img_path, tuple(target_size))
            if rgb01 is None:
                continue
            gray = to_grayscale(rgb01)
            box = _color_cc_resolve_box(
                rgb01=rgb01,
                gray=gray,
                final_roi_scale=float(final_roi_scale),
                cc_min_area=int(cc_min_area),
                cc_expand_ratio=float(cc_expand_ratio),
                use_color_cc_v2_geometry=bool(use_color_cc_v2_geometry),
                min_laser_pixels=0,
                min_laser_area_ratio=0.0,
                roi_window_side=roi_window_side,
            )
            
            if box is None:
                roi_rgb = np.zeros((int(roi_size), int(roi_size), 3), dtype=np.float32)
            else:
                rpatch = _crop(rgb01, box)
                if rpatch.size == 0:
                    roi_rgb = np.zeros((int(roi_size), int(roi_size), 3), dtype=np.float32)
                else:
                    roi_rgb = _resize_rgb_letterbox(rpatch, (int(roi_size), int(roi_size)), pad_value=0.0)
            
            color = (0.2, 1.0, 0.2)
            if layer_idx == 0:
                color = (0.2, 1.0, 0.2)
            elif layer_idx == 1:
                color = (1.0, 0.2, 0.2)
            else:
                color = (0.2, 0.5, 1.0)
            
            rgb_box = _draw_box(rgb01, box, thickness=3, color=color)
            label = f"L{layer_num}"
            rgb_box = _draw_text(rgb_box, label, top=15, left=15, font_scale=0.5, thickness=1)
            
            ph = rgb_box.shape[0]
            roi_vis = roi_rgb
            if roi_vis.shape[0] != ph:
                if HAS_CV2:
                    import cv2
                    roi_vis = cv2.resize(
                        (roi_vis * 255).astype(np.uint8),
                        (roi_vis.shape[1] * ph // roi_vis.shape[0], ph),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.float32) / 255.0
                else:
                    roi_vis = np.repeat(roi_vis[:1, :1, :], ph, axis=0)
            
            gap = np.full((ph, 6, 3), 0.0, dtype=np.float32)
            panel = np.concatenate([rgb_box, gap, roi_vis], axis=1)
            panels.append(panel)
    
    return panels


def process_hole(hole_dir: str, out_dir: str, sample_path: str, true_layer: int, pred_layer: int, target_size, roi_size, final_roi_scale, cc_min_area, cc_expand_ratio, roi_window_side, use_color_cc_v2_geometry, context_layers: int = 5):
    hole_name = os.path.basename(os.path.normpath(hole_dir))
    
    all_layers = range(max(0, true_layer - context_layers), true_layer + context_layers + 1)
    layers_to_show = [true_layer]
    
    if pred_layer != true_layer:
        layers_to_show.append(pred_layer)
    
    for ly in range(max(0, true_layer - context_layers), true_layer + context_layers + 1):
        if ly not in layers_to_show:
            layers_to_show.append(ly)
    
    layers_to_show = sorted(set(layers_to_show))
    
    panels = get_layer_images(
        hole_dir, layers_to_show, target_size, roi_size, final_roi_scale, 
        cc_min_area, cc_expand_ratio, roi_window_side, use_color_cc_v2_geometry
    )
    
    safe_name = sample_path.replace("/", "_").replace("\\", "_").replace(":", "_").replace(" ", "_")
    safe_name = safe_name[:200]
    
    if panels:
        sheet = _make_contact_sheet(panels, cols=6, pad=10, bg=0.02)
        info = f"true={true_layer}, pred={pred_layer}"
        sheet = _draw_text(sheet, info, top=20, left=20, font_scale=0.7, thickness=1)
        _imwrite(os.path.join(out_dir, f"{safe_name}__sheet.jpg"), sheet)
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="Visualize holes with error > 10 layers")
    ap.add_argument("--csv", required=True, help="CSV file with sample_path and error_layers")
    ap.add_argument("--out_dir", default="grid_diff_tcn/outputs/第二版结果（层内TCN+层外TCN+Ptransformer）/error_over10_visualization", help="Output directory")
    ap.add_argument("--context_layers", type=int, default=5, help="Context layers around penetration layer")
    ap.add_argument("--target_size", type=int, nargs=2, default=DEFAULT_TARGET_WH)
    ap.add_argument("--roi_size", type=int, default=DEFAULT_ROI_SIZE)
    ap.add_argument("--final_roi_scale", type=float, default=DEFAULT_FINAL_ROI_SCALE)
    ap.add_argument("--cc_min_area", type=int, default=DEFAULT_CC_MIN_AREA)
    ap.add_argument("--cc_expand_ratio", type=float, default=DEFAULT_CC_EXPAND_RATIO)
    ap.add_argument("--roi_window_side", type=int, default=DEFAULT_ROI_WINDOW_SIDE)
    ap.add_argument("--use_color_cc_v2_geometry", action="store_true", default=DEFAULT_USE_COLOR_CC_V2_GEOMETRY)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cases = list(reader)

    print(f"Processing {len(cases)} cases...")

    for i, row in enumerate(cases):
        sample_path = row['sample_path']
        true_layer = int(row['true_penetration_layer'])
        pred_layer = int(row['pred_penetration_layer'])
        error_layers = int(row['error_layers'])
        hole_name = os.path.basename(os.path.normpath(sample_path))
        
        if not os.path.isdir(sample_path):
            print(f"[{i+1}] Skipping (not a directory): {sample_path}")
            continue
        
        success = process_hole(
            sample_path, 
            args.out_dir, 
            sample_path,
            true_layer, 
            pred_layer, 
            args.target_size, 
            args.roi_size, 
            args.final_roi_scale, 
            args.cc_min_area, 
            args.cc_expand_ratio, 
            args.roi_window_side, 
            args.use_color_cc_v2_geometry,
            args.context_layers
        )
        if success:
            print(f"[{i+1}] {hole_name} (error={error_layers}, true={true_layer}, pred={pred_layer}) - OK")
        else:
            print(f"[{i+1}] {hole_name} (error={error_layers}) - FAILED")

    print(f"Done! Output: {args.out_dir}")


if __name__ == "__main__":
    main()
