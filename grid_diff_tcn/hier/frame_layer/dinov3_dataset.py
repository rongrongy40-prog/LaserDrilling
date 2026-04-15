# -*- coding: utf-8 -*-
"""
Hierarchical dataset using DINOv3 features instead of hand-crafted grid features.
Each frame ROI is passed through DinoV3FeatureExtractor to produce a 768-dim (ViT-B)
CLS token feature, then fed into the HierarchicalGridDiffProbTransformer.
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
from grid_diff_tcn.hier.frame_layer.dataset import compute_layer_extra_features


def _default_dinov3_transform(
    roi: np.ndarray, target_size: int = 224
) -> torch.Tensor:
    """
    Convert an ROI array to a (3, H, W) float tensor in [0, 1],
    resized to target_size (making dimensions divisible by 16).

    Args:
        roi: (H, W, 3) uint8 or float32, assumed to be RGB
        target_size: final width/height after resize (must be divisible by 16)

    Returns:
        (3, target_size, target_size) float32 tensor in [0, 1]
    """
    if roi.dtype != np.float32 and roi.dtype != np.float64:
        roi = roi.astype(np.float32) / 255.0
    else:
        roi = roi.astype(np.float32)
    if roi.max() > 1.0:
        roi = roi / 255.0

    # target dimensions must be divisible by patch_size=16
    h, w = roi.shape[:2]
    target_h = target_size
    target_w = target_size

    if h != target_h or w != target_w:
        import cv2

        roi = cv2.resize(roi, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

    tensor = torch.from_numpy(roi).permute(2, 0, 1).float()  # (3, H, W)
    return tensor


class HierarchicalDinoV3Dataset(Dataset):
    """
    Hierarchical dataset that extracts DINOv3 features per frame.

    Replaces the hand-crafted 8x8 grid features (192-dim) with DINOv3 ViT-B
    CLS token features (768-dim). All other dataset mechanics
    (layer sampling, frame selection, precomputed cache, etc.) are identical
    to the parent HierarchicalFrameLayerDataset.

    Args:
        samples_info_path: path to samples_info JSON
        dinov3_extractor: an initialized DinoV3FeatureExtractor (default None,
            only needed when not using precomputed_dir)
        dinov3_feat_dim: dimension of DINOv3 features (default 768 for ViT-B)
        roi_size: size of ROI crop (default 224, recommended for DINOv3)
        target_size: resize target (default (224, 224))
        max_layers, max_frames_per_layer, penetration_radius, etc.:
            see HierarchicalFrameLayerDataset
        precomputed_dir: if provided, load cached features from this directory
            instead of running DINOv3 at data loading time
        **kwargs: passed through to HierarchicalFrameLayerDataset init
    """

    def __init__(
        self,
        samples_info_path: str,
        dinov3_extractor: torch.nn.Module | None = None,
        dinov3_feat_dim: int = 768,
        roi_size: int = 224,
        target_size: Tuple[int, int] = (224, 224),
        max_layers: int | None = None,
        max_frames_per_layer: int = 8,
        penetration_radius: int = 2,
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
        _dinov3_target_size: int = 224,
        **_ignored_kwargs,
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
            self.samples.append(
                {
                    "sample_path": p,
                    "is_penetrated": int(it.get("is_penetrated", 0)),
                    "penetration_layer": int(it.get("penetration_layer", -1)),
                }
            )

        self._feat_dim = int(dinov3_feat_dim)
        self.roi_size = int(roi_size)
        self.target_size = target_size
        self.max_layers = max_layers
        self.max_frames_per_layer = int(max_frames_per_layer)
        self.penetration_radius = int(max(0, penetration_radius))
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
        self._dinov3_target_size = int(_dinov3_target_size)

        # DINOv3 extractor (can be None when using precomputed_dir)
        self._dinov3_extractor = dinov3_extractor
        if self._dinov3_extractor is not None:
            self._dinov3_extractor.eval()
            for param in self._dinov3_extractor.parameters():
                param.requires_grad = False

        # Precomputed name: use sample_path directly, with slashes replaced for filesystem safety
        self._precomputed_name_for_idx: List[str] = []
        for i in range(len(self.samples)):
            path = self.samples[i].get("sample_path", "") or f"sample_{i}"
            name = path.replace("/", "__").replace(os.sep, "__")
            self._precomputed_name_for_idx.append(name + ".pt")

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def feat_dim(self) -> int:
        return self._feat_dim

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

    def _extract_frame_feature(self, img_path: str) -> np.ndarray | None:
        """
        Extract a DINOv3 feature from a single frame image path.
        Returns a (dinov3_feat_dim,) numpy array or None on failure.
        """
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

        # Convert to tensor and extract DINOv3 feature
        tensor = _default_dinov3_transform(roi, target_size=self._dinov3_target_size)
        tensor = tensor.unsqueeze(0)  # (1, 3, H, W)

        with torch.inference_mode():
            feat = self._dinov3_extractor(tensor)
            if isinstance(feat, tuple):
                feat = torch.cat(feat, dim=-1)
            feat = feat.squeeze(0).cpu().numpy()

        return feat.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        # --- try precomputed cache first ---
        if self.precomputed_dir:
            pt_path = os.path.join(self.precomputed_dir, self._precomputed_name_for_idx[index])
            if not os.path.isfile(pt_path):
                pt_path = os.path.join(self.precomputed_dir, f"{index}.pt")
            if os.path.isfile(pt_path):
                raw = torch.load(pt_path, map_location="cpu")
                frame_data = raw["frame_data"]
                frame_mask = raw["frame_mask"]

                # Align seq_label T to frame_data T (cache may have mismatch)
                seq_label = raw["seq_label"]
                if seq_label.ndim == 1:
                    src_t = int(frame_data.shape[0])
                    if seq_label.shape[0] < src_t:
                        seq_label = F.pad(seq_label, (0, src_t - seq_label.shape[0]))
                    elif seq_label.shape[0] > src_t:
                        seq_label = seq_label[:src_t]
                    seq_label = seq_label.clone()

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
                    "seq_label": seq_label,
                    "label": int(raw.get("label", 0)),
                    "penetration_layer": int(raw.get("penetration_layer", -1)),
                    "layer_list": [int(x) for x in raw.get("layer_list", [])],
                    "sample_path": raw.get("sample_path", self.samples[index].get("sample_path", "")),
                }

        # --- extract on the fly ---
        if self._dinov3_extractor is None:
            raise RuntimeError(
                "HierarchicalDinoV3Dataset: dinov3_extractor is None and no "
                "precomputed_dir is set. Cannot extract features on the fly."
            )

        sample = self.samples[index]
        sample_path = sample["sample_path"]
        by_layer = self._build_layer_dict(sample_path) if os.path.isdir(sample_path) else {}
        layer_list = sorted(by_layer.keys())
        if self.max_layers is not None and len(layer_list) > self.max_layers:
            layer_list = layer_list[: int(self.max_layers)]

        if not layer_list:
            frame_data = torch.zeros(
                1, self.max_frames_per_layer, self._feat_dim, dtype=torch.float32
            )
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
            r = self.penetration_radius
            seq_label[max(0, pos - r) : min(t, pos + r + 1)] = 1

        return {
            "frame_data": torch.from_numpy(data).float(),
            "frame_mask": torch.from_numpy(mask),
            "seq_label": torch.from_numpy(seq_label).long(),
            "label": int(sample["is_penetrated"]),
            "penetration_layer": int(sample["penetration_layer"]),
            "layer_list": layer_list,
            "sample_path": sample_path,
        }
