from __future__ import annotations

import argparse
import os
import sys
from glob import glob

import numpy as np

# 允许直接运行该脚本：python grid_diff_tcn/hier/visualize_roi.py
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from grid_diff_tcn.common.roi_crop_defaults import (
    DEFAULT_CC_EXPAND_RATIO,
    DEFAULT_CC_MIN_AREA,
    DEFAULT_FINAL_ROI_SCALE,
    DEFAULT_ROI_BRIGHT_MIN_RATIO,
    DEFAULT_ROI_GRAY_P95_MIN,
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
)

# internal helper used only for visualization/scripting
from grid_diff_tcn.common.image_ops import _color_cc_resolve_box  # noqa: PLC2701


def _imwrite(path: str, rgb01: np.ndarray) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    x = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    if HAS_CV2:
        import cv2

        bgr = cv2.cvtColor(x, cv2.COLOR_RGB2BGR)
        cv2.imwrite(path, bgr)
    else:  # pragma: no cover
        from PIL import Image

        Image.fromarray(x).save(path)


def _draw_box(rgb01: np.ndarray, box: tuple[int, int, int, int] | None, thickness: int = 2) -> np.ndarray:
    out = np.clip(rgb01, 0.0, 1.0).copy()
    if box is None:
        return out
    x0, y0, x1, y1 = [int(v) for v in box]
    h, w = out.shape[:2]
    x0, x1 = max(0, min(w - 1, x0)), max(0, min(w, x1))
    y0, y1 = max(0, min(h - 1, y0)), max(0, min(h, y1))
    if x1 <= x0 or y1 <= y0:
        return out
    # lime rectangle
    c = np.array([0.2, 1.0, 0.2], dtype=np.float32)
    t = max(1, int(thickness))
    out[y0 : y0 + t, x0:x1] = c
    out[y1 - t : y1, x0:x1] = c
    out[y0:y1, x0 : x0 + t] = c
    out[y0:y1, x1 - t : x1] = c
    return out


def _gray_to_rgb01(gray: np.ndarray) -> np.ndarray:
    g = np.asarray(gray, dtype=np.float32)
    if g.ndim == 3:
        g = g[..., 0]
    g = np.clip(g, 0.0, 1.0)
    return np.repeat(g[:, :, None], 3, axis=2)


def _make_contact_sheet(panels: list[np.ndarray], cols: int, pad: int = 6, bg: float = 0.0) -> np.ndarray:
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


def _pick_evenly(paths: list[str], k: int) -> list[str]:
    if not paths:
        return []
    paths = sorted(paths)
    if len(paths) <= k:
        return paths
    idx = np.linspace(0, len(paths) - 1, int(k), dtype=int)
    return [paths[i] for i in idx]


def main() -> int:
    ap = argparse.ArgumentParser(description="批量导出多孔 ROI 裁剪拼图（原图画框 + ROI）")
    ap.add_argument("--holes_root", required=True, help="包含多个孔目录的根目录（孔目录内是 *.jpg）")
    ap.add_argument("--out_dir", default="grid_diff_tcn/roi_color_cc_examples_more", help="输出目录")
    ap.add_argument("--max_holes", type=int, default=50, help="最多处理多少个孔目录")
    ap.add_argument("--images_per_hole", type=int, default=50, help="每个孔抽多少张图做展示")
    ap.add_argument(
        "--target_size",
        type=int,
        nargs=2,
        default=DEFAULT_TARGET_WH,
        help="load 时缩放到 (W H)",
    )
    ap.add_argument("--roi_size", type=int, default=DEFAULT_ROI_SIZE, help="输出 ROI 的 letterbox 尺寸")
    ap.add_argument("--final_roi_scale", type=float, default=DEFAULT_FINAL_ROI_SCALE)
    ap.add_argument("--cc_min_area", type=int, default=DEFAULT_CC_MIN_AREA)
    ap.add_argument("--cc_expand_ratio", type=float, default=DEFAULT_CC_EXPAND_RATIO)
    ap.add_argument(
        "--roi_window_side",
        type=int,
        default=DEFAULT_ROI_WINDOW_SIDE,
        help="固定正方形边长；0 关闭（与 precompute/train 一致）",
    )
    ap.add_argument("--roi_bright_min_ratio", type=float, default=DEFAULT_ROI_BRIGHT_MIN_RATIO)
    ap.add_argument("--roi_gray_p95_min", type=float, default=DEFAULT_ROI_GRAY_P95_MIN)
    ap.add_argument("--use_color_cc_v2_geometry", action="store_true", default=DEFAULT_USE_COLOR_CC_V2_GEOMETRY)
    ap.add_argument("--no_v2_geometry", dest="use_color_cc_v2_geometry", action="store_false")
    ap.add_argument(
        "--save_per_image",
        action="store_true",
        default=False,
        help="调试用：额外保存每张图的 panel/boxed/roi（默认不保存，只出 __sheet.jpg）",
    )
    args = ap.parse_args()

    holes = [p for p in sorted(glob(os.path.join(args.holes_root, "*"))) if os.path.isdir(p)]
    holes = holes[: max(0, int(args.max_holes))]
    out_root = os.path.abspath(args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    for hi, hole_dir in enumerate(holes):
        hole_name = os.path.basename(os.path.normpath(hole_dir))
        img_paths = sorted(glob(os.path.join(hole_dir, "*.jpg")))
        picks = _pick_evenly(img_paths, int(args.images_per_hole))
        panels: list[np.ndarray] = []

        for pi, img_path in enumerate(picks):
            rgb01 = load_image_as_float(img_path, tuple(args.target_size))
            if rgb01 is None:
                continue
            gray = to_grayscale(rgb01)
            roi_window_side = norm_roi_window_side(int(args.roi_window_side))
            box = _color_cc_resolve_box(
                rgb01=rgb01,
                gray=gray,
                final_roi_scale=float(args.final_roi_scale),
                cc_min_area=int(args.cc_min_area),
                cc_expand_ratio=float(args.cc_expand_ratio),
                use_color_cc_v2_geometry=bool(args.use_color_cc_v2_geometry),
                min_laser_pixels=0,
                min_laser_area_ratio=0.0,
                roi_window_side=roi_window_side,
            )
            roi = color_cc_extract_gray_letterbox(
                rgb01=rgb01,
                gray=gray,
                roi_size=int(args.roi_size),
                final_roi_scale=float(args.final_roi_scale),
                cc_min_area=int(args.cc_min_area),
                cc_expand_ratio=float(args.cc_expand_ratio),
                min_laser_pixels=0,
                min_laser_area_ratio=0.0,
                roi_window_side=roi_window_side,
                roi_bright_min_ratio=float(args.roi_bright_min_ratio),
                roi_gray_p95_min=float(args.roi_gray_p95_min),
                use_color_cc_v2_geometry=bool(args.use_color_cc_v2_geometry),
                fixed_crop_box=None,
            )
            rgb_box = _draw_box(rgb01, box, thickness=2)
            if roi is None:
                roi_rgb = np.zeros((int(args.roi_size), int(args.roi_size), 3), dtype=np.float32)
            else:
                roi_rgb = _gray_to_rgb01(roi)

            # side-by-side panel
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
                else:  # pragma: no cover
                    roi_vis = np.repeat(roi_vis[:1, :1, :], ph, axis=0)
            gap = np.full((ph, 6, 3), 0.0, dtype=np.float32)
            panel = np.concatenate([rgb_box, gap, roi_vis], axis=1)
            panels.append(panel)

            if args.save_per_image:
                out_dir = os.path.join(out_root, f"{hi:03d}_{hole_name}")
                base = os.path.splitext(os.path.basename(img_path))[0]
                _imwrite(os.path.join(out_dir, f"{pi:02d}__{base}__panel.jpg"), panel)
                _imwrite(os.path.join(out_dir, f"{pi:02d}__{base}__boxed.jpg"), rgb_box)
                _imwrite(os.path.join(out_dir, f"{pi:02d}__{base}__roi.jpg"), roi_rgb)

        if panels:
            sheet = _make_contact_sheet(panels, cols=3, pad=10, bg=0.02)
            _imwrite(os.path.join(out_root, f"{hi:03d}_{hole_name}__sheet.jpg"), sheet)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

