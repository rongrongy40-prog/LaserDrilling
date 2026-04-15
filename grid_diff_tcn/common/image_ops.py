from __future__ import annotations

import json
import os
from glob import glob
from typing import Iterable

import numpy as np

try:
    import cv2

    HAS_CV2 = True
except ImportError:  # pragma: no cover
    HAS_CV2 = False
    from PIL import Image


# HSV 阈值（OpenCV: H∈[0,179], S/V∈[0,255]）
# 这里的“激光/亮点”在不同相机/曝光下可能偏暗或低饱和，因此用多段阈值做并集以减少漏检。
_LASER_BRIGHT_HSV_RANGES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    # 只保留黄-红色系（避免把绿色高亮吃进来）
    ((0, 70, 115), (35, 255, 255)),  # 黄/橙/红（低 H 段）
    ((160, 60, 115), (180, 255, 255)),  # 红（高 H 段，Hue wrap-around）
    # 偏暗但仍是黄-红色系：用于捕捉“暗橙色弧形火花/尾迹”
    ((0, 50, 35), (45, 255, 255)),
    ((160, 45, 35), (180, 255, 255)),
    ((0, 0, 200), (180, 110, 255)),  # 低饱和但很亮（近白/过曝）兜底
)

# “主体”颜色（以亮点为中心的局部窗口内做粗分割）
_BODY_HSV_RANGES: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = (
    # 主体也限制在黄-红色系 + 红 wrap-around
    ((0, 25, 10), (50, 255, 255)),
    ((160, 20, 10), (180, 255, 255)),
)


def parse_frame_layer_from_filename(filename: str):
    """
    从图片文件名解析拍摄序号(帧)与层号。
    约定：去掉扩展名后按 '_' 分段，倒数第二段为拍摄序号，最后一段为层号。
    例：2025_04_29_07_56_50_324_1058_157.jpg → 帧=1058，层=157
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    parts = base.split("_")
    if len(parts) < 2:
        return None, None
    try:
        frame_num = int(parts[-2])
        layer_num = int(parts[-1])
        return frame_num, layer_num
    except ValueError:
        return None, None


def load_image_as_float(path: str, target_size: tuple[int, int] | None = None):
    """
    加载单张图片为 RGB float32 [0,1]，可选缩放到 target_size=(W,H)。
    失败返回 None。
    """
    if HAS_CV2:
        img = cv2.imread(path)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
    else:  # pragma: no cover
        img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0

    if target_size is not None:
        h, w = img.shape[:2]
        tw, th = int(target_size[0]), int(target_size[1])
        if (w, h) != (tw, th):
            if HAS_CV2:
                out_a, in_a = float(th * tw), float(h * w)
                interp = cv2.INTER_AREA if out_a < in_a else cv2.INTER_CUBIC
                img = cv2.resize(img, (tw, th), interpolation=interp)
            else:  # pragma: no cover
                img = np.array(
                    Image.fromarray((img * 255).astype(np.uint8)).resize((tw, th)),
                    dtype=np.float32,
                ) / 255.0
    return img


def to_grayscale(rgb_image: np.ndarray) -> np.ndarray:
    """RGB 转灰度，输出 shape (H,W) float32。"""
    if rgb_image.ndim == 3:
        return (
            0.299 * rgb_image[:, :, 0] + 0.587 * rgb_image[:, :, 1] + 0.114 * rgb_image[:, :, 2]
        ).astype(np.float32)
    return rgb_image.astype(np.float32)


def _resize_gray(gray: np.ndarray, size_hw: tuple[int, int]) -> np.ndarray:
    th, tw = int(size_hw[0]), int(size_hw[1])
    h, w = int(gray.shape[0]), int(gray.shape[1])
    if h == th and w == tw:
        return gray.astype(np.float32)
    if HAS_CV2:
        out_a, in_a = float(th * tw), float(h * w)
        interp = cv2.INTER_CUBIC if out_a > in_a else cv2.INTER_AREA
        return cv2.resize(gray, (tw, th), interpolation=interp).astype(np.float32)
    from PIL import Image  # pragma: no cover

    arr = (np.clip(gray, 0, 1) * 255).astype(np.uint8)
    out = np.array(Image.fromarray(arr).resize((tw, th)), dtype=np.float32) / 255.0
    return out.astype(np.float32)


def _resize_gray_letterbox(gray: np.ndarray, size_hw: tuple[int, int], pad_value: float = 0.0) -> np.ndarray:
    th, tw = int(size_hw[0]), int(size_hw[1])
    h, w = int(gray.shape[0]), int(gray.shape[1])
    if h <= 0 or w <= 0:
        return np.full((th, tw), float(pad_value), dtype=np.float32)
    if h == th and w == tw:
        return gray.astype(np.float32)
    scale = min(float(th) / float(h), float(tw) / float(w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    resized = _resize_gray(gray, (nh, nw))
    out = np.full((th, tw), float(pad_value), dtype=np.float32)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _resize_rgb_letterbox(rgb: np.ndarray, size_hw: tuple[int, int], pad_value: float = 0.0) -> np.ndarray:
    th, tw = int(size_hw[0]), int(size_hw[1])
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    if h <= 0 or w <= 0:
        return np.full((th, tw, 3), float(pad_value), dtype=np.float32)
    if h == th and w == tw:
        return rgb.astype(np.float32)
    scale = min(float(th) / float(h), float(tw) / float(w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    if HAS_CV2:
        rgb_u8 = (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8)
        out_a, in_a = float(nh * nw), float(h * w)
        interp = cv2.INTER_CUBIC if out_a > in_a else cv2.INTER_AREA
        resized = cv2.resize(rgb_u8, (nw, nh), interpolation=interp).astype(np.float32) / 255.0
    else:  # pragma: no cover
        from PIL import Image

        arr = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
        resized = np.array(Image.fromarray(arr).resize((nw, nh)), dtype=np.float32) / 255.0
    out = np.full((th, tw, 3), float(pad_value), dtype=np.float32)
    y0 = (th - nh) // 2
    x0 = (tw - nw) // 2
    out[y0 : y0 + nh, x0 : x0 + nw] = resized
    return out


def _clamp_box(x0, y0, x1, y1, w: int, h: int):
    x0 = max(0, min(w - 1, int(x0)))
    y0 = max(0, min(h - 1, int(y0)))
    x1 = max(1, min(w, int(x1)))
    y1 = max(1, min(h, int(y1)))
    if x1 <= x0:
        x1 = min(w, x0 + 1)
    if y1 <= y0:
        y1 = min(h, y0 + 1)
    return int(x0), int(y0), int(x1), int(y1)


def _box_from_center(cx: float, cy: float, side: float, w: int, h: int):
    half = float(side) / 2.0
    return _clamp_box(cx - half, cy - half, cx + half, cy + half, w, h)


def _crop(gray_or_rgb: np.ndarray, box: tuple[int, int, int, int]):
    x0, y0, x1, y1 = box
    return gray_or_rgb[y0:y1, x0:x1]


def _shrink_box(box: tuple[int, int, int, int], scale: float, w: int, h: int):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    side = max(1.0, min((x1 - x0), (y1 - y0)) * float(scale))
    return _box_from_center(cx, cy, side, w, h)


def _shrink_box_outer(box: tuple[int, int, int, int], scale: float, w: int, h: int):
    x0, y0, x1, y1 = box
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    dw = float(x1 - x0)
    dh = float(y1 - y0)
    side = max(1.0, max(dw, dh) * float(scale))
    return _box_from_center(cx, cy, side, w, h)


def _union_boxes(b1, b2):
    return (
        min(int(b1[0]), int(b2[0])),
        min(int(b1[1]), int(b2[1])),
        max(int(b1[2]), int(b2[2])),
        max(int(b1[3]), int(b2[3])),
    )


def _square_box_in_image(cx: float, cy: float, side: int, w: int, h: int):
    cxi = int(round(float(np.clip(cx, 0, max(0, w - 1)))))
    cyi = int(round(float(np.clip(cy, 0, max(0, h - 1)))))
    want = max(8, int(side))
    s = min(want, w, h)
    while s >= 8:
        half = s // 2
        x0 = cxi - half
        y0 = cyi - half
        x1 = x0 + s
        y1 = y0 + s
        if x0 >= 0 and y0 >= 0 and x1 <= w and y1 <= h:
            return x0, y0, x1, y1
        s -= 1
    return None


def _laser_bright_mask_u8(rgb01: np.ndarray) -> np.ndarray | None:
    if not HAS_CV2:
        return None
    h, w = rgb01.shape[:2]
    if h < 2 or w < 2:
        return None
    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)
    bright = None
    for lo, hi in _LASER_BRIGHT_HSV_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        bright = m if bright is None else cv2.bitwise_or(bright, m)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k3)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k5)
    return bright


def _center_xy_for_fixed_roi_window(rgb01: np.ndarray, box: tuple[int, int, int, int], cc_min_area: int):
    if not HAS_CV2:
        x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    x0, y0, x1, y1 = (int(box[0]), int(box[1]), int(box[2]), int(box[3]))
    h, w = rgb01.shape[:2]
    xi0, yi0 = max(0, x0), max(0, y0)
    xi1, yi1 = min(w, x1), min(h, y1)
    bright = _laser_bright_mask_u8(rgb01)
    if bright is None or xi1 <= xi0 or yi1 <= yi0:
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    sub = bright[yi0:yi1, xi0:xi1]
    n, _, stats, cents = cv2.connectedComponentsWithStats(sub, connectivity=8)
    best_i = -1
    best_a = 0
    min_a = max(5, int(cc_min_area) // 3)
    for i in range(1, n):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > best_a:
            best_a = a
            best_i = i
    if best_i < 0 or best_a < min_a:
        return (x0 + x1) / 2.0, (y0 + y1) / 2.0
    cx_loc = float(cents[best_i, 0])
    cy_loc = float(cents[best_i, 1])
    return xi0 + cx_loc, yi0 + cy_loc


def iter_sample_jpg_sorted_by_layer_frame(sample_path: str, exclude_set: set | None) -> list[str]:
    rows: list[tuple[int, int, str]] = []
    for p in glob(os.path.join(sample_path, "*.jpg")):
        if exclude_set and os.path.normpath(p) in exclude_set:
            continue
        fn = os.path.basename(p)
        fr, ly = parse_frame_layer_from_filename(fn)
        if fr is None or ly is None:
            continue
        rows.append((int(ly), int(fr), p))
    rows.sort(key=lambda t: (t[0], t[1]))
    return [t[2] for t in rows]


def compute_hole_anchor_crop_box(
    sample_path: str,
    target_size: tuple[int, int],
    anchor_num_images: int,
    exclude_set: set | None,
    final_roi_scale: float,
    cc_min_area: int,
    cc_expand_ratio: float,
    use_color_cc_v2_geometry: bool,
    roi_window_side: int | None,
) -> tuple[int, int, int, int] | None:
    paths = iter_sample_jpg_sorted_by_layer_frame(sample_path, exclude_set)
    if not paths:
        return None
    n = max(1, int(anchor_num_images))
    take = paths[:n]
    boxes: list[tuple[int, int, int, int]] = []
    centers: list[tuple[float, float]] = []
    h_img, w_img = 0, 0
    for p in take:
        img = load_image_as_float(p, target_size)
        if img is None:
            continue
        gray = to_grayscale(img)
        h_img, w_img = int(gray.shape[0]), int(gray.shape[1])
        b = _color_cc_box(
            img,
            gray,
            float(final_roi_scale),
            cc_min_area=int(cc_min_area),
            cc_expand_ratio=float(cc_expand_ratio),
            use_v2_geometry=bool(use_color_cc_v2_geometry),
        )
        if b is None:
            continue
        x0, y0, x1, y1 = (int(b[0]), int(b[1]), int(b[2]), int(b[3]))
        boxes.append((x0, y0, x1, y1))
        centers.append(((x0 + x1) / 2.0, (y0 + y1) / 2.0))
    if not boxes:
        return None
    if roi_window_side is not None:
        cx = float(np.median([c[0] for c in centers]))
        cy = float(np.median([c[1] for c in centers]))
        sq = _square_box_in_image(cx, cy, int(roi_window_side), w_img, h_img)
        return sq
    x0 = min(b[0] for b in boxes)
    y0 = min(b[1] for b in boxes)
    x1 = max(b[2] for b in boxes)
    y1 = max(b[3] for b in boxes)
    return (x0, y0, x1, y1)


def count_laser_bright_pixels(rgb01: np.ndarray) -> int:
    m = _laser_bright_mask_u8(rgb01)
    if m is None:
        return 0
    return int(np.count_nonzero(m))


def full_image_passes_laser_gate(
    rgb01: np.ndarray,
    min_laser_pixels: int = 0,
    min_laser_area_ratio: float = 0.0,
) -> bool:
    if min_laser_pixels <= 0 and min_laser_area_ratio <= 0.0:
        return True
    h, w = rgb01.shape[:2]
    cnt = count_laser_bright_pixels(rgb01)
    if min_laser_pixels > 0 and cnt < int(min_laser_pixels):
        return False
    if min_laser_area_ratio > 0.0 and cnt < float(min_laser_area_ratio) * float(h * w):
        return False
    return True


def roi_letterboxed_quality_passes(
    rgb_patch: np.ndarray,
    gray_letterbox: np.ndarray,
    roi_size: int,
    roi_bright_min_ratio: float = 0.0,
    roi_gray_p95_min: float = 0.0,
) -> bool:
    if roi_bright_min_ratio <= 0.0 and roi_gray_p95_min <= 0.0:
        return True
    gl = np.asarray(gray_letterbox, dtype=np.float32).reshape(-1)
    if roi_gray_p95_min > 0.0:
        p95 = float(np.percentile(gl, 95))
        if p95 < float(roi_gray_p95_min):
            return False
    if roi_bright_min_ratio > 0.0:
        rgb_lb = _resize_rgb_letterbox(
            np.clip(rgb_patch, 0.0, 1.0).astype(np.float32),
            (int(roi_size), int(roi_size)),
            pad_value=0.0,
        )
        b = _laser_bright_mask_u8(rgb_lb)
        if b is None:
            return False
        r = float(np.count_nonzero(b)) / float(b.size)
        if r < float(roi_bright_min_ratio):
            return False
    return True


def load_exclude_set(exclude_json_path: str) -> set[str]:
    if not exclude_json_path or not os.path.isfile(exclude_json_path):
        return set()
    with open(exclude_json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    out: set[str] = set()
    for hole in obj.get("holes", []) or []:
        for cand in hole.get("candidates", []) or []:
            p = cand.get("image_path")
            if p:
                out.add(os.path.normpath(p))
    return out


def _color_cc_box(
    rgb01: np.ndarray,
    gray: np.ndarray,
    final_roi_scale: float,
    cc_min_area: int = 12,
    cc_expand_ratio: float = 0.2,
    use_v2_geometry: bool = True,
):
    if not HAS_CV2:
        return None
    h, w = gray.shape[:2]
    rgb_u8 = (np.clip(rgb01, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr_u8 = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr_u8, cv2.COLOR_BGR2HSV)

    # 与 _laser_bright_mask_u8 保持一致（多段并集阈值）
    bright = None
    for lo, hi in _LASER_BRIGHT_HSV_RANGES:
        m = cv2.inRange(hsv, lo, hi)
        bright = m if bright is None else cv2.bitwise_or(bright, m)
    k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    bright = cv2.morphologyEx(bright, cv2.MORPH_OPEN, k3)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, k5)

    n, _, stats, cents = cv2.connectedComponentsWithStats(bright, connectivity=8)
    best = None
    best_score = -1e18
    for i in range(1, n):
        x, y, bw, bh, area = stats[i]
        if int(area) < int(cc_min_area):
            continue
        patch = hsv[max(0, y) : min(h, y + bh), max(0, x) : min(w, x + bw)]
        if patch.size == 0:
            continue
        v_mean = float(np.mean(patch[:, :, 2]))
        s_mean = float(np.mean(patch[:, :, 1]))
        score = float(area) * 0.6 + v_mean * 0.9 + s_mean * 0.2
        if score > best_score:
            cx, cy = cents[i]
            best_score = score
            best = (float(cx), float(cy), (int(x), int(y), int(x + bw), int(y + bh)))
    if best is None:
        return None

    cx, cy, bb = best
    bx0, by0, bx1, by1 = bb

    xw = int(min(h, w) * 0.28)
    xa0, ya0 = max(0, int(cx) - xw), max(0, int(cy) - xw)
    xa1, ya1 = min(w, int(cx) + xw), min(h, int(cy) + xw)
    if xa1 <= xa0 or ya1 <= ya0:
        return None
    local = hsv[ya0:ya1, xa0:xa1]
    body = None
    for lo, hi in _BODY_HSV_RANGES:
        m = cv2.inRange(local, lo, hi)
        body = m if body is None else cv2.bitwise_or(body, m)
    body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, k5)
    body = cv2.dilate(body, k5, iterations=1)
    n2, _, st2, _ = cv2.connectedComponentsWithStats(body, connectivity=8)

    best2 = None
    best2_score = -1e18
    for i in range(1, n2):
        x, y, bw, bh, area = st2[i]
        if int(area) < 50:
            continue
        gcx = xa0 + x + bw / 2.0
        gcy = ya0 + y + bh / 2.0
        d = ((gcx - cx) ** 2 + (gcy - cy) ** 2) ** 0.5
        score = float(area) - 1.8 * float(d)
        if score > best2_score:
            best2_score = score
            best2 = (int(xa0 + x), int(ya0 + y), int(xa0 + x + bw), int(ya0 + y + bh))

    if best2 is None:
        x0, y0, x1, y1 = bx0, by0, bx1, by1
    elif use_v2_geometry:
        x0, y0, x1, y1 = _union_boxes((bx0, by0, bx1, by1), best2)
    else:
        x0, y0, x1, y1 = best2

    bw0 = max(1, x1 - x0)
    bh0 = max(1, y1 - y0)
    side_pad = max(bw0, bh0) if use_v2_geometry else max(1, min(bw0, bh0))
    pad = int(float(cc_expand_ratio) * float(side_pad))
    x0, y0 = max(0, x0 - pad), max(0, y0 - pad)
    x1, y1 = min(w, x1 + pad), min(h, y1 + pad)
    if use_v2_geometry:
        box = _shrink_box_outer((x0, y0, x1, y1), final_roi_scale, w=w, h=h)
    else:
        box = _shrink_box((x0, y0, x1, y1), final_roi_scale, w=w, h=h)
    return box


def _color_cc_resolve_box(
    rgb01: np.ndarray,
    gray: np.ndarray,
    final_roi_scale: float,
    cc_min_area: int = 12,
    cc_expand_ratio: float = 0.2,
    use_color_cc_v2_geometry: bool = True,
    min_laser_pixels: int = 0,
    min_laser_area_ratio: float = 0.0,
    roi_window_side: int | None = None,
):
    if not full_image_passes_laser_gate(rgb01, min_laser_pixels, min_laser_area_ratio):
        return None
    box = _color_cc_box(
        rgb01,
        gray,
        final_roi_scale,
        cc_min_area=cc_min_area,
        cc_expand_ratio=cc_expand_ratio,
        use_v2_geometry=use_color_cc_v2_geometry,
    )
    if box is None:
        return None
    h, w = gray.shape[:2]
    if roi_window_side is not None:
        cx, cy = _center_xy_for_fixed_roi_window(rgb01, box, cc_min_area)
        sq = _square_box_in_image(cx, cy, int(roi_window_side), w, h)
        if sq is None:
            return None
        box = sq
    return box


def color_cc_extract_gray_letterbox(
    rgb01: np.ndarray,
    gray: np.ndarray,
    roi_size: int,
    final_roi_scale: float,
    cc_min_area: int = 12,
    cc_expand_ratio: float = 0.2,
    min_laser_pixels: int = 0,
    min_laser_area_ratio: float = 0.0,
    roi_window_side: int | None = None,
    roi_bright_min_ratio: float = 0.0,
    roi_gray_p95_min: float = 0.0,
    use_color_cc_v2_geometry: bool = True,
    fixed_crop_box: tuple[int, int, int, int] | None = None,
) -> np.ndarray | None:
    if fixed_crop_box is not None:
        if not full_image_passes_laser_gate(rgb01, min_laser_pixels, min_laser_area_ratio):
            return None
        box = tuple(int(x) for x in fixed_crop_box)
    else:
        box = _color_cc_resolve_box(
            rgb01,
            gray,
            final_roi_scale,
            cc_min_area=cc_min_area,
            cc_expand_ratio=cc_expand_ratio,
            use_color_cc_v2_geometry=use_color_cc_v2_geometry,
            min_laser_pixels=min_laser_pixels,
            min_laser_area_ratio=min_laser_area_ratio,
            roi_window_side=roi_window_side,
        )
    if box is None:
        return None
    gpatch = _crop(gray, box)
    rpatch = _crop(rgb01.astype(np.float32), box)
    if gpatch.size == 0:
        return None
    roi_lb = _resize_gray_letterbox(gpatch, (int(roi_size), int(roi_size)), pad_value=0.0).astype(np.float32)
    if not roi_letterboxed_quality_passes(
        rpatch, roi_lb, int(roi_size), roi_bright_min_ratio, roi_gray_p95_min
    ):
        return None
    return roi_lb

