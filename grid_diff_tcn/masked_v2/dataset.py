# -*- coding: utf-8 -*-
"""
Dataset for masked image modeling.
Loads raw ROI images and applies center masking during training.
"""

from __future__ import annotations

import json
import os
from collections import defaultdict
from glob import glob
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as FF
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


def _default_transform(
    roi: np.ndarray,
    target_size: int = 224,
) -> torch.Tensor:
    """
    Convert ROI array to (3, H, W) float tensor in [0, 1].
    """
    if roi.dtype != np.float32 and roi.dtype != np.float64:
        roi = roi.astype(np.float32) / 255.0
    else:
        roi = roi.astype(np.float32)
    if roi.max() > 1.0:
        roi = roi / 255.0

    h, w = roi.shape[:2]
    if h != target_size or w != target_size:
        import cv2
        roi = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(roi).permute(2, 0, 1).float()
    return tensor


def _extract_roi_one(
    img_path: str,
    roi_size: int,
    final_roi_scale: float,
    cc_min_area: int,
    cc_expand_ratio: float,
    min_laser_pixels: int,
    min_laser_area_ratio: float,
    roi_window_side: int,
    roi_bright_min_ratio: float,
    roi_gray_p95_min: float,
    use_color_cc_v2_geometry: bool,
    use_grayscale: bool,
) -> np.ndarray | None:
    # Load at roi_size directly: load+resize (24ms) is cheaper than
    # full-res load (24ms) + full-res box (125ms). Benchmark proves this is fastest.
    img = load_image_as_float(img_path, (roi_size, roi_size))
    if img is None:
        return None
    gray = to_grayscale(img)
    if use_grayscale:
        return color_cc_extract_gray_letterbox(
            rgb01=img, gray=gray, roi_size=roi_size,
            final_roi_scale=final_roi_scale, cc_min_area=cc_min_area,
            cc_expand_ratio=cc_expand_ratio,
            min_laser_pixels=min_laser_pixels,
            min_laser_area_ratio=min_laser_area_ratio,
            roi_window_side=roi_window_side,
            roi_bright_min_ratio=roi_bright_min_ratio,
            roi_gray_p95_min=roi_gray_p95_min,
            use_color_cc_v2_geometry=use_color_cc_v2_geometry,
            fixed_crop_box=None,
        )
    else:
        box = _color_cc_resolve_box(
            rgb01=img, gray=gray, final_roi_scale=final_roi_scale,
            cc_min_area=cc_min_area, cc_expand_ratio=cc_expand_ratio,
            use_color_cc_v2_geometry=use_color_cc_v2_geometry,
            min_laser_pixels=min_laser_pixels,
            min_laser_area_ratio=min_laser_area_ratio,
            roi_window_side=roi_window_side,
        )
        if box is None:
            return None
        # img is already at roi_size, box coords are in same space
        roi = _crop(img, box)
        if roi.size == 0:
            return None
        return _resize_rgb_letterbox(roi, (roi_size, roi_size), pad_value=0.0)


class MaskedDrillingDataset(Dataset):
    """
    Dataset for masked image modeling.
    
    Loads raw ROI images and returns them with center masking applied.
    For stage 1 (pre-training): returns full images for pixel reconstruction
    For stage 2 (fine-tuning): can work with pre-extracted features or images
    
    Set preload=True to load all ROI tensors into RAM at init, 
    eliminating disk I/O during training (~10x speedup).
    """
    
    def __init__(
        self,
        samples_info_path: str,
        roi_size: int = 224,
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
        use_grayscale: bool = False,
        preload: bool = False,
        preload_workers: int = 8,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        with open(samples_info_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "Categories" in raw:
            raw = raw.get("Categories", [])
        
        self.samples: List[dict] = []
        for it in raw:
            p = str(it.get("sample_path", ""))
            self.samples.append({
                "sample_path": p,
                "is_penetrated": int(it.get("is_penetrated", 0)),
                "penetration_layer": int(it.get("penetration_layer", -1)),
            })

        # Debug: limit number of samples
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
            print(f"[MaskedDrillingDataset] limited to {len(self.samples)} samples (max_samples={max_samples})")

        self.roi_size = int(roi_size)
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
        self.use_grayscale = bool(use_grayscale)
        
        # Preload: only scan directories into memory (fast, ~16s for 1015 samples).
        # Actual ROI extraction is parallelized by DataLoader num_workers during training.
        self.preload = preload
        self._sample_layer_dicts: List[Dict[int, List[Tuple[int, str]]]] = []
        self._sample_layer_lists: List[List[int]] = []
        
        if self.preload:
            from tqdm import tqdm
            
            print(f"[MaskedDrillingDataset] Preloading directory structure ({len(self.samples)} samples)...")
            for si, sample in enumerate(tqdm(self.samples, desc="Scanning dirs")):
                sample_path = sample["sample_path"]
                layer_dict = self._build_layer_dict(sample_path) if os.path.isdir(sample_path) else {}
                layer_list = sorted(layer_dict.keys())
                if self.max_layers is not None and len(layer_list) > self.max_layers:
                    layer_list = layer_list[:int(self.max_layers)]
                self._sample_layer_dicts.append(layer_dict)
                self._sample_layer_lists.append(layer_list)
            print(f"[MaskedDrillingDataset] Scan done.")
        else:
            self._preloaded = [None] * len(self.samples)
    
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
    
    def _extract_roi(self, img_path: str) -> np.ndarray | None:
        """Extract ROI from image path."""
        img = load_image_as_float(img_path, (self.roi_size, self.roi_size))
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
        
        return roi
    
    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        sample_path = sample["sample_path"]
        
        # Use pre-scanned layer dicts if available (avoids redundant dir scans per epoch)
        if self.preload and index < len(self._sample_layer_lists):
            layer_list = self._sample_layer_lists[index]
            by_layer = self._sample_layer_dicts[index] if index < len(self._sample_layer_dicts) else {}
        else:
            by_layer = self._build_layer_dict(sample_path) if os.path.isdir(sample_path) else {}
            layer_list = sorted(by_layer.keys())
            if self.max_layers is not None and len(layer_list) > self.max_layers:
                layer_list = layer_list[:int(self.max_layers)]
        
        if not layer_list:
            frame_data = torch.zeros(
                self.max_frames_per_layer, 3, self.roi_size, self.roi_size,
                dtype=torch.float32
            )
            frame_mask = torch.zeros(self.max_frames_per_layer, dtype=torch.bool)
            seq_label = torch.zeros(1, dtype=torch.long)
            return {
                "frame_data": frame_data,
                "frame_mask": frame_mask,
                "seq_label": seq_label,
                "label": sample["is_penetrated"],
                "penetration_layer": sample["penetration_layer"],
                "layer_list": [0],
                "sample_path": sample_path,
                "dataset_idx": index,
            }
        
        t = len(layer_list)
        f = self.max_frames_per_layer
        
        data = torch.zeros(t, f, 3, self.roi_size, self.roi_size, dtype=torch.float32)
        mask = torch.zeros(t, f, dtype=torch.bool)
        
        for ti, ly in enumerate(layer_list):
            picks = self._select_frames(by_layer.get(ly, []))
            for fi, p in enumerate(picks[:f]):
                roi = self._extract_roi(p)
                if roi is None:
                    continue
                tensor = _default_transform(roi, target_size=self.roi_size)
                data[ti, fi] = tensor
                mask[ti, fi] = True
        
        t = len(layer_list)
        seq_label = torch.zeros(t, dtype=torch.int64)
        if int(sample["is_penetrated"]) == 1 and int(sample["penetration_layer"]) in layer_list:
            pos = layer_list.index(int(sample["penetration_layer"]))
            seq_label[pos:] = 1
        
        return {
            "frame_data": data,
            "frame_mask": mask,
            "seq_label": seq_label,
            "label": sample["is_penetrated"],
            "penetration_layer": sample["penetration_layer"],
            "layer_list": layer_list,
            "sample_path": sample_path,
            "dataset_idx": index,
        }


def collate_masked_batch(batch: list) -> dict:
    """
    Collate function for masked dataset.

    Handles both raw image input (T,F,3,H,W) and precomputed features (T,F,feat_dim).
    Detected automatically based on the dimensionality of the first item.
    """
    ndims = batch[0]["frame_data"].dim()
    is_feature = (ndims == 3)  # (T, F, feat_dim) vs (T, F, 3, H, W)

    max_t = max(item["frame_data"].shape[0] for item in batch)
    max_f = max(item["frame_data"].shape[1] for item in batch)
    B = len(batch)

    if is_feature:
        # Precomputed features: (T, F, feat_dim)
        feat_dim = batch[0]["frame_data"].shape[2]
        frame_data = torch.zeros(B, max_t, max_f, feat_dim, dtype=torch.float32)
    else:
        # Raw images: (T, F, 3, H, W)
        C = 3
        H = batch[0]["frame_data"].shape[-2]
        W = batch[0]["frame_data"].shape[-1]
        frame_data = torch.zeros(B, max_t, max_f, C, H, W, dtype=torch.float32)

    frame_mask = torch.zeros(len(batch), max_t, max_f, dtype=torch.bool)
    seq_labels = []
    labels = []
    penetration_layers = []
    layer_lists = []
    sample_paths = []

    for bi, item in enumerate(batch):
        src_t, src_f = item["frame_data"].shape[:2]
        frame_data[bi, :src_t, :src_f] = item["frame_data"]
        frame_mask[bi, :src_t, :src_f] = item["frame_mask"]
        seq_labels.append(item["seq_label"])
        labels.append(item["label"])
        penetration_layers.append(item["penetration_layer"])
        layer_lists.append(item["layer_list"])
        sample_paths.append(item["sample_path"])

    seq_labels_tensor = torch.full((len(batch), max_t), -100, dtype=torch.int64)
    for bi, sl in enumerate(seq_labels):
        src_t = len(sl)
        seq_labels_tensor[bi, :src_t] = sl

    dataset_indices = torch.tensor(
        [item.get("dataset_idx", bi) for bi, item in enumerate(batch)],
        dtype=torch.long
    )

    return {
        "frame_data": frame_data,
        "frame_mask": frame_mask,
        "seq_label": seq_labels_tensor,
        "labels": torch.tensor(labels, dtype=torch.long),
        "penetration_layers": torch.tensor(penetration_layers, dtype=torch.int64),
        "layer_lists": layer_lists,
        "sample_paths": sample_paths,
        "dataset_indices": dataset_indices,
    }


# ---------------------------------------------------------------------------
# CropCacheDataset: 读取预裁剪的 .pt 缓存文件训练。
# 训练时直接 torch.load，绕过所有图像处理，无 I/O 开销。
#
# 可选支持预计算特征目录（来自 extract.py）：
#   --precomputed_dir data_drilling/features_cache
#   加载 .pt 中 "features": (T, F, feat_dim) 替代原始帧作为 model 输入，
#   同时配合 --use_cached_features 让模型跳过 DINOv3 forward。
# ---------------------------------------------------------------------------

class CropCacheDataset(Dataset):
    """
    读取 pre_crop.py 生成的 .pt 缓存文件。

    .pt 文件结构（由 pre_crop.py 生成）：
        frames: (T, F, 3, H, W) float32, 0-1 归一化
        mask:   (T, F) bool
        layers: list[int]，每层实际编号
        sample_path: str

    __getitem__ 返回的字典与 MaskedDrillingDataset 格式完全一致，
    collate_fn 可共用 collate_masked_batch。

    当 precomputed_dir 设定时，返回预计算 DINOv3 特征
    (T, F, feat_dim)，配合 use_cached_features=True 使用。
    """

    def __init__(
        self,
        samples_info_path: str,
        cache_dir: str,
        roi_size: int = 224,
        max_layers: int | None = None,
        max_frames_per_layer: int = 8,
        preload: bool = False,
        precomputed_dir: str | None = None,
        max_samples: int | None = None,
    ) -> None:
        super().__init__()
        with open(samples_info_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict) and "Categories" in raw:
            raw = raw.get("Categories", [])

        self.samples: List[dict] = []
        for it in raw:
            p = str(it.get("sample_path", ""))
            self.samples.append({
                "sample_path": p,
                "is_penetrated": int(it.get("is_penetrated", 0)),
                "penetration_layer": int(it.get("penetration_layer", -1)),
            })

        # Debug: limit number of samples
        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.cache_dir = str(cache_dir)
        self.roi_size = int(roi_size)
        self.max_layers = max_layers
        self.max_frames_per_layer = int(max_frames_per_layer)
        self.preload = preload
        self.precomputed_dir = precomputed_dir

        # 建立 sample_path -> roi_cache_path 映射（与 pre_crop.py 一致）
        self._cache_map: dict[str, str] = {}
        for s in self.samples:
            sp = s["sample_path"]
            if not sp:
                continue
            rel = os.path.relpath(sp, os.getcwd())
            safe = rel.replace(os.sep, "_").replace("/", "_")
            self._cache_map[sp] = os.path.join(self.cache_dir, f"{safe}.pt")

        # 建立 sample_path -> precomputed_feature_path 映射（与 extract.py 一致）
        self._feat_map: dict[str, str] = {}
        if self.precomputed_dir:
            for s in self.samples:
                sp = s["sample_path"]
                if not sp:
                    continue
                key = sp.replace(os.sep, "__").replace("/", "__").replace(".", "_")
                self._feat_map[sp] = os.path.join(self.precomputed_dir, f"{key}.pt")

            # 过滤：只保留有预计算特征的样本（use_cached_features 模式）
            self.samples = [s for s in self.samples
                            if self._feat_map.get(s["sample_path"])
                            and os.path.exists(self._feat_map[s["sample_path"]])]
            print(f"[CropCacheDataset] Filtered to {len(self.samples)} samples with cached features "
                  f"(from {self.precomputed_dir})")

        # 过滤：只保留有有效 ROI 缓存的样本（缺失或加载失败的 .pt 直接删除）
        skipped = []
        valid_samples = []
        if self.preload:
            # preload 模式：预加载时直接验证，合法则保留，失败则跳过
            from tqdm import tqdm
            self._preloaded = [None] * len(self.samples)
            self._feat_preloaded = [None] * len(self.samples)
            print(f"[CropCacheDataset] Preloading & validating {len(self.samples)} .pt files...")
            for si, s in enumerate(tqdm(self.samples, desc="Loading cache")):
                sp = s["sample_path"]
                cp = self._cache_map.get(sp)
                if cp and os.path.exists(cp):
                    try:
                        self._preloaded[si] = torch.load(cp, map_location="cpu", weights_only=False)
                        valid_samples.append(s)
                    except Exception:
                        skipped.append(sp)
                        self._preloaded[si] = None
                else:
                    skipped.append(sp)
                if self.precomputed_dir:
                    fp = self._feat_map.get(sp)
                    if fp and os.path.exists(fp):
                        try:
                            self._feat_preloaded[si] = torch.load(fp, map_location="cpu", weights_only=False)
                        except Exception:
                            self._feat_preloaded[si] = None
            print("[CropCacheDataset] Preload done.")
        else:
            # 非 preload 模式：逐个检查文件是否存在（不完整加载）
            self._preloaded = [None] * len(self.samples)
            self._feat_preloaded = [None] * len(self.samples)
            from tqdm import tqdm
            for s in tqdm(self.samples, desc="Checking cache"):
                sp = s["sample_path"]
                cp = self._cache_map.get(sp)
                if cp and os.path.exists(cp):
                    valid_samples.append(s)
                else:
                    skipped.append(sp)
        if skipped:
            print(f"[CropCacheDataset] Skipped {len(skipped)} samples with missing/unreadable "
                  f"cache files: {skipped}")
        self.samples = valid_samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        sample_path = sample["sample_path"]

        if self.preload and self._preloaded[index] is not None:
            cached = self._preloaded[index]
        else:
            cp = self._cache_map.get(sample_path)
            if cp and os.path.exists(cp):
                try:
                    cached = torch.load(cp, map_location="cpu", weights_only=False)
                except Exception:
                    cached = None
            else:
                cached = None

        # 预计算特征（来自 extract.py）
        feat_cached = None
        if self.precomputed_dir:
            if self.preload and self._feat_preloaded[index] is not None:
                feat_cached = self._feat_preloaded[index]
            else:
                fp = self._feat_map.get(sample_path)
                if fp and os.path.exists(fp):
                    try:
                        feat_cached = torch.load(fp, map_location="cpu", weights_only=False)
                    except Exception:
                        feat_cached = None

        if cached is None:
            # 缓存不存在，返回零张量（与 MaskedDrillingDataset 空样本格式一致）
            # 形状: (T, F, 3, H, W) 其中 T=1, F=max_frames_per_layer
            frame_data = torch.zeros(
                1, self.max_frames_per_layer, 3, self.roi_size, self.roi_size,
                dtype=torch.float32
            )
            frame_mask = torch.zeros(1, self.max_frames_per_layer, dtype=torch.bool)
            seq_label = torch.zeros(1, dtype=torch.int64)
            layer_list = [0]
        else:
            # cached["frames"]: (T, F, 3, H, W), cached["mask"]: (T, F)
            raw_frames = cached["frames"]       # (T, F, 3, H, W)
            raw_mask: torch.Tensor = cached["mask"]  # (T, F) bool
            layer_list: list[int] = cached.get("layers", list(range(raw_frames.shape[0])))

            # Handle new uint8+64x64 cache format: convert to float32 and resize
            if cached.get("_uint8", False):
                stored_roi_size = cached.get("_roi_size", 64)
                raw_frames = raw_frames.float() / 255.0  # uint8 → float32 [0,1]
                if stored_roi_size != self.roi_size:
                    T, F, C, H, W = raw_frames.shape
                    flat = raw_frames.permute(0, 1, 3, 4, 2).reshape(T * F, C, H, W)
                    flat = FF.interpolate(flat, size=(self.roi_size, self.roi_size),
                                         mode="bilinear", align_corners=False)
                    raw_frames = flat.reshape(T, F, 3, self.roi_size, self.roi_size)

            if self.max_layers and len(layer_list) > self.max_layers:
                layer_list = layer_list[:self.max_layers]
                raw_frames = raw_frames[:self.max_layers]
                raw_mask = raw_mask[:self.max_layers]

            # 如果有预计算特征，用 (T, F, feat_dim) 替换原始帧
            if feat_cached is not None:
                # 支持两种格式：extract.py 输出的 "features" 和 precompute.py 输出的 "frame_data"
                if "features" in feat_cached:
                    frame_data = feat_cached["features"]   # (T, F, feat_dim) from extract.py
                else:
                    frame_data = feat_cached["frame_data"]  # (T, F, feat_dim) from precompute.py
            else:
                frame_data = raw_frames  # already (T, F, 3, H, W)
            frame_mask = raw_mask

            # seq_label: first penetration layer and all subsequent layers = 1
            t = len(layer_list)
            seq_label = torch.zeros(t, dtype=torch.int64)
            if (int(sample["is_penetrated"]) == 1
                    and int(sample["penetration_layer"]) in layer_list):
                pos = layer_list.index(int(sample["penetration_layer"]))
                seq_label[pos:] = 1

        return {
            "frame_data": frame_data,
            "frame_mask": frame_mask,
            "seq_label": seq_label,
            "label": sample["is_penetrated"],
            "penetration_layer": sample["penetration_layer"],
            "layer_list": layer_list,
            "sample_path": sample_path,
            "dataset_idx": index,
        }