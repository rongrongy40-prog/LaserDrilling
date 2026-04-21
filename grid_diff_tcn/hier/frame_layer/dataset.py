# -*- coding: utf-8 -*-
"""
Hierarchical dataset for frame-level + layer-level modeling.

Returns per-sample tensors:
  - frame_data: (T, F, 192)  per-layer per-frame grid features (8*8*3: mean+std+max)
  - frame_mask: (T, F)       valid frame mask
  - seq_label:  (T,)         per-layer binary label
  - layer_list: list[int]
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from grid_diff_tcn.common.roi_crop_defaults import norm_roi_window_side
from grid_diff_tcn.common.image_ops import (
    color_cc_extract_gray_letterbox,
    load_exclude_set,
    load_image_as_float,
    parse_frame_layer_from_filename,
    to_grayscale,
    _color_cc_resolve_box,
    _crop,
    _resize_rgb_letterbox,
)

def _pool_regions_8x8(x_btf: torch.Tensor) -> torch.Tensor:
    """
    x_btf: (B,T,F,C) float
      C = 64  → 8*8 grid, 1 stat per cell
      C = 192 → 8*8 grid, 3 stats per cell (mean+std+max)
    return: (B,T,F,5) float
      5 regions = [center2x2, top, bottom, left, right]
      每个 region 对所有 stat 维度取平均
    """
    b, t, f, c = x_btf.shape
    n_cell = 8 * 8
    if c == n_cell:
        # 单统计量：直接 reshape
        g = x_btf.view(b, t, f, 8, 8)
    elif c == n_cell * 3:
        # 3 统计量 (mean,std,max)：reshape 为 (B,T,F,8,8,3)
        g = x_btf.view(b, t, f, 8, 8, 3)
    else:
        raise ValueError(f"_pool_regions_8x8: expected C=64 or 192, got C={c}")
    center = g[..., 3:5, 3:5, :].mean(dim=(-1, -2, -3))
    top    = g[..., :2, :, :].mean(dim=(-1, -2, -3))
    bottom = g[..., -2:, :, :].mean(dim=(-1, -2, -3))
    left   = g[..., :, :2, :].mean(dim=(-1, -2, -3))
    right  = g[..., :, -2:, :].mean(dim=(-1, -2, -3))
    return torch.stack([center, top, bottom, left, right], dim=-1)


def _masked_mean_var(x: torch.Tensor, m: torch.Tensor, dim: int, eps: float = 1e-6):
    """
    x: (..., F, D)
    m: (..., F) in {0,1}
    returns (mean, var): (..., D)
    """
    m = m.to(dtype=x.dtype)
    denom = m.sum(dim=dim, keepdim=True).clamp(min=1.0).unsqueeze(-1)  # align with (...,1,D)
    mean = (x * m.unsqueeze(-1)).sum(dim=dim, keepdim=True) / denom
    var = ((x - mean) ** 2 * m.unsqueeze(-1)).sum(dim=dim, keepdim=True) / denom.clamp(min=eps)
    return mean.squeeze(dim), var.squeeze(dim)


def _safe_lag1_corr(x: torch.Tensor, m: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    x: (B,T,F,D)
    m: (B,T,F) bool
    returns: (B,T,D) correlation between x[t] and x[t-1] over valid pairs
    """
    if x.size(2) <= 1:
        return torch.zeros(x.size(0), x.size(1), x.size(3), device=x.device, dtype=x.dtype)
    x0 = x[:, :, :-1, :]
    x1 = x[:, :, 1:, :]
    mp = (m[:, :, :-1] & m[:, :, 1:]).to(dtype=x.dtype)
    mean0, var0 = _masked_mean_var(x0, mp, dim=2, eps=eps)
    mean1, var1 = _masked_mean_var(x1, mp, dim=2, eps=eps)
    cov = ((x0 - mean0.unsqueeze(2)) * (x1 - mean1.unsqueeze(2)) * mp.unsqueeze(-1)).sum(dim=2)
    denom = mp.sum(dim=2).clamp(min=1.0).unsqueeze(-1)
    cov = cov / denom
    corr = cov / (var0.clamp(min=eps).sqrt() * var1.clamp(min=eps).sqrt())
    return torch.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0).clamp(-1.0, 1.0)


def _rfft_band_energy(x: torch.Tensor, m: torch.Tensor, n_bands: int = 3, eps: float = 1e-8) -> tuple[torch.Tensor, torch.Tensor]:
    """
    x: (B,T,F,D)
    m: (B,T,F) bool
    returns:
      - band_energy_ratio: (B,T,D,n_bands)
      - spectral_entropy: (B,T,D)
    """
    b, t, f, d = x.shape
    xf = x * m.to(dtype=x.dtype).unsqueeze(-1)
    # rfft along F, take power
    spec = torch.fft.rfft(xf, dim=2)
    p = (spec.real**2 + spec.imag**2)  # (B,T,Fr,D)
    # ignore DC for stability
    if p.size(2) > 1:
        p_use = p[:, :, 1:, :]
    else:
        p_use = p
    total = p_use.sum(dim=2, keepdim=False).clamp(min=eps)  # (B,T,D)
    # bands over frequency bins
    fr = int(p_use.size(2))
    if fr == 0:
        ber = torch.zeros(b, t, d, n_bands, device=x.device, dtype=x.dtype)
        se = torch.zeros(b, t, d, device=x.device, dtype=x.dtype)
        return ber, se
    edges = torch.linspace(0, fr, steps=n_bands + 1, device=x.device)
    bands = []
    for i in range(n_bands):
        lo = int(edges[i].item())
        hi = int(edges[i + 1].item())
        hi = max(hi, lo + 1)
        hi = min(hi, fr)
        seg = p_use[:, :, lo:hi, :].sum(dim=2)  # (B,T,D)
        bands.append(seg / total)
    ber = torch.stack(bands, dim=-1)  # (B,T,D,n_bands)
    # spectral entropy
    prob = p_use / total.unsqueeze(2)
    ent = -(prob.clamp(min=eps) * prob.clamp(min=eps).log()).sum(dim=2)  # (B,T,D)
    ent = ent / float(max(1, fr))  # normalize roughly
    return ber, torch.nan_to_num(ent, nan=0.0, posinf=0.0, neginf=0.0)


def compute_layer_extra_features(frame_data: torch.Tensor, frame_mask: torch.Tensor) -> torch.Tensor:
    """
    frame_data: (B,T,F,192)
    frame_mask: (B,T,F) bool
    return layer_extra: (B,T,E) float
    E = 5 regions * (mean,std,diff_l1,lag1_corr) + 5 regions * (band1,band2,band3,spec_entropy) + 1 valid_ratio
      = 5*4 + 5*4 + 1 = 41
    """
    x = frame_data
    m = frame_mask.to(dtype=torch.bool)
    regions = _pool_regions_8x8(x)  # (B,T,F,5)
    valid_ratio = m.to(dtype=regions.dtype).mean(dim=2, keepdim=False)  # (B,T)

    mean_r, var_r = _masked_mean_var(regions, m, dim=2)
    std_r = var_r.clamp(min=1e-8).sqrt()
    # diff L1 energy
    if regions.size(2) > 1:
        dr = (regions[:, :, 1:, :] - regions[:, :, :-1, :]).abs()
        md = (m[:, :, 1:] & m[:, :, :-1])
        diff_mean, _ = _masked_mean_var(dr, md, dim=2)
    else:
        diff_mean = torch.zeros_like(mean_r)
    lag1 = _safe_lag1_corr(regions, m)
    ber, sent = _rfft_band_energy(regions, m, n_bands=3)

    # flatten
    feats = [
        mean_r,
        std_r,
        diff_mean,
        lag1,
        ber.reshape(ber.size(0), ber.size(1), -1),
        sent,
        valid_ratio.unsqueeze(-1),
    ]
    out = torch.cat(feats, dim=-1)
    return torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)


def _grid_pool_single(
    gray_roi: np.ndarray,
    grid: Tuple[int, int] = (8, 8),
    pool_stats: Tuple[str, ...] = ("mean", "std", "max"),
) -> np.ndarray:
    """
    每个 grid patch 输出多个统计量，扩展特征维度。

    gray_roi: (H, W) float32 灰度图
    grid: (rows, cols) 网格划分
    pool_stats: 每个 patch 提取哪些统计量，支持 "mean", "std", "max"
    return: (grid_rows * grid_cols * len(pool_stats),) float32
    """
    h, w = gray_roi.shape[:2]
    gr, gc = int(grid[0]), int(grid[1])
    n_stats = len(pool_stats)
    n_cell = gr * gc
    feat_dim = n_cell * n_stats

    if h < gr or w < gc:
        return np.zeros((feat_dim,), dtype=np.float32)

    ph, pw = h // gr, w // gc
    out = np.zeros((feat_dim,), dtype=np.float32)
    k = 0
    for r in range(gr):
        for c in range(gc):
            patch = gray_roi[r * ph : (r + 1) * ph, c * pw : (c + 1) * pw]
            if patch.size == 0:
                out[k : k + n_stats].fill(0.0)
                k += n_stats
                continue
            vals = patch.ravel()
            for stat in pool_stats:
                if stat == "mean":
                    out[k] = float(np.mean(vals))
                elif stat == "std":
                    out[k] = float(np.std(vals))
                elif stat == "max":
                    out[k] = float(np.max(vals))
                else:
                    out[k] = 0.0
                k += 1
    return out


class HierarchicalFrameLayerDataset(Dataset):
    def __init__(
        self,
        samples_info_path: str,
        base_dir: str | None = None,
        target_size: Tuple[int, int] = (128, 128),
        roi_size: int = 96,
        grid: Tuple[int, int] = (8, 8),
        pool_stats: Tuple[str, ...] = ("mean", "std", "max"),
        max_layers: int | None = None,
        max_frames_per_layer: int = 8,
        exclude_json: str | None = None,
        final_roi_scale: float = 0.85,
        cc_min_area: int = 12,
        cc_expand_ratio: float = 0.2,
        min_laser_pixels: int = 0,
        min_laser_area_ratio: float = 0.0,
        roi_window_side: int | None = None,
        roi_bright_min_ratio: float = 0.0,
        roi_gray_p95_min: float = 0.0,
        use_color_cc_v2_geometry: bool = True,
        precomputed_dir: str | None = None,
        use_grayscale: bool = False,
        **_ignored_legacy_kwargs,
    ) -> None:
        super().__init__()
        with open(samples_info_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "Categories" in raw:
            raw = raw.get("Categories", [])
        if not isinstance(raw, list):
            raw = list(raw) if hasattr(raw, "__iter__") else []

        self.samples: List[dict] = []
        for it in raw:
            p = str(it.get("sample_path", ""))
            if base_dir and not os.path.isabs(p):
                p = os.path.join(base_dir, p)
            self.samples.append(
                {
                    "sample_path": p,
                    "is_penetrated": int(it.get("is_penetrated", 0)),
                    "penetration_layer": int(it.get("penetration_layer", -1)),
                }
            )

        self.target_size = target_size
        self.roi_size = int(roi_size)
        self.grid = grid
        self.pool_stats = tuple(pool_stats)
        self._feat_dim = int(grid[0]) * int(grid[1]) * len(self.pool_stats)
        self.max_layers = max_layers
        self.max_frames_per_layer = int(max_frames_per_layer)
        self.exclude_set = load_exclude_set(exclude_json) if exclude_json else set()

        self.final_roi_scale = float(final_roi_scale)
        self.cc_min_area = int(cc_min_area)
        self.cc_expand_ratio = float(cc_expand_ratio)
        self.min_laser_pixels = int(min_laser_pixels)
        self.min_laser_area_ratio = float(min_laser_area_ratio)
        self.roi_window_side = norm_roi_window_side(roi_window_side)
        self.roi_bright_min_ratio = float(roi_bright_min_ratio)
        self.roi_gray_p95_min = float(roi_gray_p95_min)
        self.use_color_cc_v2_geometry = bool(use_color_cc_v2_geometry)
        self.precomputed_dir = os.path.abspath(precomputed_dir) if precomputed_dir else None
        self.use_grayscale = bool(use_grayscale)

        # Precomputed name: use sample_path directly, with slashes replaced for filesystem safety
        self._precomputed_name_for_idx: List[str] = []
        for i in range(len(self.samples)):
            path = self.samples[i].get("sample_path", "") or f"sample_{i}"
            name = path.replace("/", "__").replace(os.sep, "__")
            self._precomputed_name_for_idx.append(name + ".pt")

    def __len__(self) -> int:
        return len(self.samples)

    def _build_layer_dict(self, sample_path: str) -> Dict[int, List[Tuple[int, str]]]:
        by = defaultdict(list)
        for p in glob(os.path.join(sample_path, "*.jpg")):
            n = os.path.normpath(p)
            if self.exclude_set and n in self.exclude_set:
                continue
            fn = os.path.basename(p)
            fr, ly = parse_frame_layer_from_filename(fn)
            if fr is None or ly is None:
                continue
            by[int(ly)].append((int(fr), p))
        return by

    def _select_frames(self, items: List[Tuple[int, str]]) -> List[str]:
        items = sorted(items, key=lambda x: x[0])
        paths = [p for _, p in items]
        if len(paths) <= self.max_frames_per_layer:
            return paths
        idx = np.linspace(0, len(paths) - 1, self.max_frames_per_layer, dtype=int)
        return [paths[i] for i in idx]

    def _extract_frame_feature(
        self,
        img_path: str,
    ) -> np.ndarray | None:
        img = load_image_as_float(img_path, self.target_size)
        if img is None:
            return None
        if self.use_grayscale:
            gray = to_grayscale(img)
            roi = color_cc_extract_gray_letterbox(
                rgb01=img,
                gray=gray,
                roi_size=self.roi_size,
                final_roi_scale=self.final_roi_scale,
                cc_min_area=self.cc_min_area,
                cc_expand_ratio=self.cc_expand_ratio,
                min_laser_pixels=self.min_laser_pixels,
                min_laser_area_ratio=self.min_laser_area_ratio,
                roi_window_side=self.roi_window_side,
                roi_bright_min_ratio=self.roi_bright_min_ratio,
                roi_gray_p95_min=self.roi_gray_p95_min,
                use_color_cc_v2_geometry=self.use_color_cc_v2_geometry,
                fixed_crop_box=None,
            )
        else:
            gray = to_grayscale(img)
            box = _color_cc_resolve_box(
                rgb01=img,
                gray=gray,
                final_roi_scale=self.final_roi_scale,
                cc_min_area=self.cc_min_area,
                cc_expand_ratio=self.cc_expand_ratio,
                use_color_cc_v2_geometry=self.use_color_cc_v2_geometry,
                min_laser_pixels=self.min_laser_pixels,
                min_laser_area_ratio=self.min_laser_area_ratio,
                roi_window_side=self.roi_window_side,
            )
            if box is None:
                return None
            roi = _crop(img, box)
            if roi.size == 0:
                return None
            roi = _resize_rgb_letterbox(roi, (self.roi_size, self.roi_size), pad_value=0.0)
        
        if roi is None or roi.size == 0:
            return None
        return _grid_pool_single(roi, grid=self.grid, pool_stats=self.pool_stats)

    def __getitem__(self, index: int) -> dict:
        if self.precomputed_dir:
            pt_path = os.path.join(self.precomputed_dir, self._precomputed_name_for_idx[index])
            if not os.path.isfile(pt_path):
                pt_path = os.path.join(self.precomputed_dir, f"{index}.pt")
            if os.path.isfile(pt_path):
                raw = torch.load(pt_path, map_location="cpu")
                frame_data = raw["frame_data"]
                frame_mask = raw["frame_mask"]
                # 兼容：预计算缓存与当前 max_frames_per_layer 不一致时，自动裁剪/补零
                target_f = int(self.max_frames_per_layer)
                src_f = int(frame_data.shape[1]) if frame_data.ndim >= 2 else target_f
                if src_f != target_f:
                    t = int(frame_data.shape[0]) if frame_data.ndim >= 1 else 1
                    c = int(frame_data.shape[2]) if frame_data.ndim >= 3 else self._feat_dim
                    adj_data = torch.zeros(t, target_f, c, dtype=frame_data.dtype)
                    adj_mask = torch.zeros(t, target_f, dtype=torch.bool)
                    copy_f = min(src_f, target_f)
                    adj_data[:, :copy_f] = frame_data[:, :copy_f]
                    adj_mask[:, :copy_f] = frame_mask[:, :copy_f].to(torch.bool)
                    frame_data = adj_data
                    frame_mask = adj_mask
                return {
                    "frame_data": frame_data,
                    "frame_mask": frame_mask,
                    "seq_label": raw["seq_label"],
                    "label": int(raw.get("label", 0)),
                    "penetration_layer": int(raw.get("penetration_layer", -1)),
                    "layer_list": [int(x) for x in raw.get("layer_list", [])],
                    "sample_path": raw.get("sample_path", self.samples[index].get("sample_path", "")),
                }

        sample = self.samples[index]
        sample_path = sample["sample_path"]
        by_layer = self._build_layer_dict(sample_path) if os.path.isdir(sample_path) else {}
        layer_list = sorted(by_layer.keys())
        if self.max_layers is not None and len(layer_list) > self.max_layers:
            layer_list = layer_list[: int(self.max_layers)]

        if not layer_list:
            frame_data = torch.zeros(1, self.max_frames_per_layer, self._feat_dim, dtype=torch.float32)
            frame_mask = torch.zeros(1, self.max_frames_per_layer, dtype=torch.bool)
            seq_label = torch.zeros(1, dtype=torch.long)
            return {
                "frame_data": frame_data,
                "frame_mask": frame_mask,
                "seq_label": seq_label,
                "label": sample["is_penetrated"],
                "penetration_layer": sample["penetration_layer"],
                "layer_list": [0],
                "sample_path": sample_path,
            }

        t = len(layer_list)
        f = self.max_frames_per_layer
        data = np.zeros((t, f, self._feat_dim), dtype=np.float32)
        mask = np.zeros((t, f), dtype=np.bool_)

        for ti, ly in enumerate(layer_list):
            picks = self._select_frames(by_layer.get(ly, []))
            for fi, p in enumerate(picks[:f]):
                feat = self._extract_frame_feature(p)
                if feat is None:
                    continue
                data[ti, fi] = feat
                mask[ti, fi] = True

        seq_label = np.zeros((t,), dtype=np.int64)
        if int(sample["is_penetrated"]) == 1 and int(sample["penetration_layer"]) in layer_list:
            pos = layer_list.index(int(sample["penetration_layer"]))
            seq_label[pos:] = 1

        return {
            "frame_data": torch.from_numpy(data).float(),
            "frame_mask": torch.from_numpy(mask),
            "seq_label": torch.from_numpy(seq_label).long(),
            "label": int(sample["is_penetrated"]),
            "penetration_layer": int(sample["penetration_layer"]),
            "layer_list": layer_list,
            "sample_path": sample_path,
        }


def collate_hierarchical_batch(batch: List[dict]) -> dict:
    max_t = max(int(b["frame_data"].shape[0]) for b in batch) if batch else 1
    f = int(batch[0]["frame_data"].shape[1]) if batch else 1
    c = int(batch[0]["frame_data"].shape[2]) if batch else 192
    bsz = len(batch)

    x = torch.zeros(bsz, max_t, f, c, dtype=torch.float32)
    m = torch.zeros(bsz, max_t, f, dtype=torch.bool)
    y = torch.zeros(bsz, max_t, dtype=torch.long)
    layer_mask = torch.zeros(bsz, max_t, dtype=torch.bool)

    labels, pen_layers, layer_lists, paths = [], [], [], []
    for i, s in enumerate(batch):
        fd = s["frame_data"]
        t_actual = int(fd.shape[0])

        # Align seq_label T to frame_data T (precomputed cache may have mismatch)
        seq_lbl = s["seq_label"]
        if seq_lbl.ndim == 1:
            if seq_lbl.shape[0] < t_actual:
                seq_lbl = F.pad(seq_lbl, (0, t_actual - seq_lbl.shape[0]))
            elif seq_lbl.shape[0] > t_actual:
                seq_lbl = seq_lbl[:t_actual]

        x[i, :t_actual] = fd
        m[i, :t_actual] = s["frame_mask"]
        y[i, :t_actual] = seq_lbl
        layer_mask[i, :t_actual] = True
        labels.append(int(s["label"]))
        pen_layers.append(int(s["penetration_layer"]))
        layer_lists.append(s["layer_list"])
        paths.append(s["sample_path"])

    out = {
        "frame_data": x,
        "frame_mask": m,
        "seq_label": y,
        "layer_mask": layer_mask,
        "label": torch.tensor(labels, dtype=torch.long),
        "penetration_layer": torch.tensor(pen_layers, dtype=torch.long),
        "layer_list": layer_lists,
        "sample_path": paths,
    }
    try:
        out["layer_extra"] = compute_layer_extra_features(x, m)
    except Exception:
        # 特征为辅助项；若某环境 torch.fft 不可用或其他异常，退化为不提供
        pass
    return out

