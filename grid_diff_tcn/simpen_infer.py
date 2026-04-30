# -*- coding: utf-8 -*-
"""
Per-frame similarity-based penetration detection using SimPenDec/stage1.pt encoder.

New pipeline:
  1. Load the DINOv3 encoder from SimPenDec/stage1.pt (Stage 1 checkpoint).
  2. Build a per-frame feature library:
       - For each penetrated training sample, at the penetration layer,
         collect ALL valid frames (not mean-pooled).
       - Each frame → one 384-dim DINOv3 feature vector.
       - Store: {feature, sample_path, pen_layer, frame_num}
  3. For each test sample (CAUSAL, frame-by-frame):
       - Process frames in order (within layer by frame_num, layers by layer_num).
       - Each frame: search library → get max similarity.
       - Maintain a rolling baseline (recent N frame similarities).
       - Penetration detected when: max_sim > baseline_mean + k*baseline_std + lower.
       - Stop at first penetration frame. Find its layer. Layer error = |pred - true|.
  4. Output: JSON with per-frame details + layer error.

ROI extraction is identical to pre_crop.py (same parameters, same parsing).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict, deque
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent      # grid_diff_tcn/ → dinov3-main/

# ---------------------------------------------------------------------------
# ROI extraction — identical to pre_crop.py
# ---------------------------------------------------------------------------

from grid_diff_tcn.common.image_ops import (
    load_image_as_float,
    to_grayscale,
    _color_cc_resolve_box,
    _crop,
    _resize_rgb_letterbox,
    parse_frame_layer_from_filename,
)
from grid_diff_tcn.common.roi_crop_defaults import (
    DEFAULT_ROI_SIZE,
    DEFAULT_FINAL_ROI_SCALE,
    DEFAULT_CC_MIN_AREA,
    DEFAULT_CC_EXPAND_RATIO,
    DEFAULT_MIN_LASER_PIXELS,
    DEFAULT_MIN_LASER_AREA_RATIO,
    DEFAULT_USE_COLOR_CC_V2_GEOMETRY,
    DEFAULT_ROI_WINDOW_SIDE,
    norm_roi_window_side,
)


def _crop_one_image(
    img_path: str,
    roi_size: int,
    final_roi_scale: float,
    cc_min_area: int,
    cc_expand_ratio: float,
    min_laser_pixels: int,
    min_laser_area_ratio: float,
    roi_window_side: int | None,
    use_color_cc_v2_geometry: bool,
) -> np.ndarray | None:
    """Crop ROI from a single image — mirrors pre_crop.py exactly."""
    img = load_image_as_float(img_path, (roi_size, roi_size))
    if img is None:
        return None
    gray = to_grayscale(img)
    box = _color_cc_resolve_box(
        rgb01=img, gray=gray,
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


# ---------------------------------------------------------------------------
# 1. Load encoder
# ---------------------------------------------------------------------------

def load_encoder_from_simpen(
    checkpoint_path: str,
    dinov3_model: str = "vit_small",
    dinov3_feat_dim: int = 384,
    dinov3_roi_size: int = 224,
):
    """Load DinoV3FeatureExtractor from a SimPenDec Stage-1 checkpoint."""
    from grid_diff_tcn.hier.frame_layer.dinov3_features import DinoV3FeatureExtractor

    print(f"[Encoder] Loading: {checkpoint_path}")
    extractor = DinoV3FeatureExtractor(
        model_name=dinov3_model, pretrained=False,
        image_size=dinov3_roi_size, device="cpu",
    )
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd = ckpt["state_dict"]
    enc_sd = {k[18:]: v for k, v in sd.items() if k.startswith("dinov3_extractor.")}
    extractor.load_state_dict(enc_sd, strict=False)
    extractor.eval()
    print(f"[Encoder] DINOv3 ({dinov3_model}) encoder loaded, feat_dim={dinov3_feat_dim}")
    return extractor


# ---------------------------------------------------------------------------
# 2. Per-frame feature library
# ---------------------------------------------------------------------------

@dataclass
class FrameEntry:
    """A single frame's feature vector with metadata."""
    feature: torch.Tensor  # (feat_dim,)
    sample_path: str
    pen_layer: int
    frame_num: int


def _scan_frames_from_dir(sample_path: str):
    """
    Scan a sample directory, return frames grouped by layer and sorted by frame number.

    Returns:
        frames_by_layer: dict[int, list[(frame_num, path)]]
        all_layers: sorted list of layer numbers
        layer_of_frame: dict[(layer, frame_num)] → frame_num  (reverse lookup)
    """
    frames_by_layer = defaultdict(list)
    for p in glob(os.path.join(sample_path, "*.jpg")):
        fn = os.path.basename(p)
        fr, ly = parse_frame_layer_from_filename(fn)
        if fr is None or ly is None:
            continue
        frames_by_layer[int(ly)].append((int(fr), p))

    all_layers = sorted(frames_by_layer.keys())
    for ly in frames_by_layer:
        frames_by_layer[ly].sort(key=lambda x: x[0])
    return frames_by_layer, all_layers


def build_feature_library(
    samples_info_path: str,
    encoder: torch.nn.Module,
    device: str = "cuda",
    roi_size: int | None = None,
    chunk_size: int = 32,
    final_roi_scale: float | None = None,
    cc_min_area: int | None = None,
    cc_expand_ratio: float | None = None,
    min_laser_pixels: int | None = None,
    min_laser_area_ratio: float | None = None,
    roi_window_side: int | None = None,
    use_color_cc_v2_geometry: bool | None = None,
) -> list[FrameEntry]:
    """
    Build a per-frame feature library from penetrated training samples.

    For each penetrated sample:
      1. Scan the directory, find the penetration layer.
      2. Collect ALL valid frames from the penetration layer (not mean-pooled).
      3. Each frame → one DINOv3 feature vector → one FrameEntry.

    Returns list of FrameEntry objects.
    """
    roi_size = roi_size if roi_size is not None else DEFAULT_ROI_SIZE
    final_roi_scale = final_roi_scale if final_roi_scale is not None else DEFAULT_FINAL_ROI_SCALE
    cc_min_area = cc_min_area if cc_min_area is not None else DEFAULT_CC_MIN_AREA
    cc_expand_ratio = cc_expand_ratio if cc_expand_ratio is not None else DEFAULT_CC_EXPAND_RATIO
    min_laser_pixels = min_laser_pixels if min_laser_pixels is not None else DEFAULT_MIN_LASER_PIXELS
    min_laser_area_ratio = min_laser_area_ratio if min_laser_area_ratio is not None else DEFAULT_MIN_LASER_AREA_RATIO
    roi_window_side = norm_roi_window_side(roi_window_side)
    use_color_cc_v2_geometry = (
        use_color_cc_v2_geometry if use_color_cc_v2_geometry is not None
        else DEFAULT_USE_COLOR_CC_V2_GEOMETRY
    )

    with open(samples_info_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("Categories", [])
    samples = [{
        "sample_path": str(it.get("sample_path", "")),
        "is_penetrated": int(it.get("is_penetrated", 0)),
        "penetration_layer": int(it.get("penetration_layer", -1)),
    } for it in raw]

    pen_samples = [s for s in samples if s["is_penetrated"] == 1]
    print(f"[Library] {len(pen_samples)}/{len(samples)} penetrated samples for library")
    if not pen_samples:
        raise ValueError("No penetrated samples found!")

    all_entries: list[FrameEntry] = []

    for sample in tqdm(pen_samples, desc="Building per-frame library"):
        sp = sample["sample_path"]
        if not os.path.isdir(sp):
            continue
        frames_by_layer, all_layers = _scan_frames_from_dir(sp)
        pen_layer = sample["penetration_layer"]
        if pen_layer not in frames_by_layer:
            print(f"[Library] WARNING: pen_layer {pen_layer} not found in {sp}")
            continue

        frames = frames_by_layer[pen_layer]  # sorted by frame_num

        for frame_num, path in frames:
            roi = _crop_one_image(
                path, roi_size,
                final_roi_scale, cc_min_area, cc_expand_ratio,
                min_laser_pixels, min_laser_area_ratio,
                roi_window_side, use_color_cc_v2_geometry,
            )
            if roi is None:
                continue

            tensor = torch.from_numpy(roi.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder(tensor).cpu().squeeze(0)  # (feat_dim,)
            entry = FrameEntry(
                feature=feat, sample_path=sp,
                pen_layer=pen_layer, frame_num=frame_num,
            )
            all_entries.append(entry)

    print(f"[Library] Built with {len(all_entries)} per-frame entries "
          f"({len(pen_samples)} samples × ~{len(all_entries)//max(len(pen_samples),1)} frames/samp avg)")
    return all_entries


# ---------------------------------------------------------------------------
# 3. FAISS index for per-frame library
# ---------------------------------------------------------------------------

def build_faiss_index(
    entries: list[FrameEntry],
    use_faiss: bool = True,
    nprobe: int = 8,
) -> tuple[object, int]:
    """
    Build FAISS index over per-frame features (L2-normalized, inner product = cosine sim).

    For small libraries (< 2048 frames): IndexFlatIP (exact).
    For large libraries: IndexIVFPQ (approximate, fast).
    """
    if not use_faiss or len(entries) < 10:
        return None, len(entries)

    try:
        import faiss
    except Exception:
        print("[FAISS] not available, using brute-force fallback")
        return None, len(entries)

    mat = torch.stack([e.feature for e in entries]).numpy().astype(np.float32)
    d, n = mat.shape[1], mat.shape[0]

    # Normalize for cosine similarity
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    mat = mat / (norms + 1e-8)

    if n <= 2048:
        idx = faiss.IndexFlatIP(d)
        idx.add(mat)
        print(f"[FAISS] IndexFlatIP({n} frames, d={d})")
        return idx, n

    nb = max(64, min(256, n // 16))
    m_pq = min(48, d)
    quantizer = faiss.IndexFlatIP(d)
    idx = faiss.IndexIVFPQ(quantizer, d, nb, 8, m_pq)
    idx.train(mat)
    idx.add(mat)
    idx.nprobe = nprobe
    print(f"[FAISS] IndexIVFPQ({n} frames, d={d}, nb={nb}, m={m_pq}, nprobe={nprobe})")
    return idx, n


def search_library(
    index: object,
    feat_np: np.ndarray,
    k: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Search the library for top-k most similar frames.

    Args:
        index: FAISS index or None
        feat_np: (feat_dim,) query, should already be normalized

    Returns:
        (topk_scores shape=(k,), topk_indices shape=(k,))
    """
    import faiss
    q = feat_np.reshape(1, -1).astype(np.float32)
    if index is not None:
        D, I = index.search(q, k)
        return D[0], I[0]
    else:
        # Brute-force fallback using torch
        pass  # handled at call site


# ---------------------------------------------------------------------------
# 4. Per-frame causal inference
# ---------------------------------------------------------------------------

def infer_on_testset(
    samples_info_path: str,
    encoder: torch.nn.Module,
    entries: list[FrameEntry],
    device: str = "cuda",
    roi_size: int | None = None,
    chunk_size: int = 32,
    skip_first_layers: int = 30,
    baseline_window: int = 20,
    spike_k: float = 1.0,
    spike_lower: float = 0.05,
    warmup_frames: int = 10,
    use_faiss: bool = True,
    faiss_nprobe: int = 8,
    return_all_sim_history: bool = False,
    final_roi_scale: float | None = None,
    cc_min_area: int | None = None,
    cc_expand_ratio: float | None = None,
    min_laser_pixels: int | None = None,
    min_laser_area_ratio: float | None = None,
    roi_window_side: int | None = None,
    use_color_cc_v2_geometry: bool | None = None,
) -> tuple[list[dict], list[float]] | list[dict]:
    """
    Per-frame causal inference with spike detection.

    Pipeline:
      For each test sample:
        1. Scan frames, process IN ORDER (by layer, then by frame number).
           Skip layers <= skip_first_layers.
        2. Each frame: ROI crop → DINOv3 → search library → max_sim.
        3. Maintain a rolling baseline of recent frame max_sims.
        4. After warmup_frames: penetration detected when:
               max_sim > baseline_mean + spike_k * baseline_std + spike_lower
        5. Stop at first penetration frame. Find its layer.
        6. Layer error = |pred_layer - true_layer|.

    Args:
        baseline_window: number of recent frame similarities to use for baseline
        spike_k: multiplier on baseline_std in spike detection
        spike_lower: absolute lower bound on spike threshold
        warmup_frames: minimum frames before spike detection activates

    Returns:
        results list of dicts, and optionally all_sim_history (if return_all_sim_history).
    """
    roi_size = roi_size if roi_size is not None else DEFAULT_ROI_SIZE
    final_roi_scale = final_roi_scale if final_roi_scale is not None else DEFAULT_FINAL_ROI_SCALE
    cc_min_area = cc_min_area if cc_min_area is not None else DEFAULT_CC_MIN_AREA
    cc_expand_ratio = cc_expand_ratio if cc_expand_ratio is not None else DEFAULT_CC_EXPAND_RATIO
    min_laser_pixels = min_laser_pixels if min_laser_pixels is not None else DEFAULT_MIN_LASER_PIXELS
    min_laser_area_ratio = min_laser_area_ratio if min_laser_area_ratio is not None else DEFAULT_MIN_LASER_AREA_RATIO
    roi_window_side = norm_roi_window_side(roi_window_side)
    use_color_cc_v2_geometry = (
        use_color_cc_v2_geometry if use_color_cc_v2_geometry is not None
        else DEFAULT_USE_COLOR_CC_V2_GEOMETRY
    )

    with open(samples_info_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict):
        raw = raw.get("Categories", [])
    samples = [{
        "sample_path": str(it.get("sample_path", "")),
        "is_penetrated": int(it.get("is_penetrated", 0)),
        "penetration_layer": int(it.get("penetration_layer", -1)),
    } for it in raw]

    if len(entries) == 0:
        raise ValueError("Library is empty!")

    # Stack all features for brute-force fallback
    lib_mat = torch.stack([e.feature for e in entries])  # (N, feat_dim)
    lib_norm = F.normalize(lib_mat, dim=1)  # (N, feat_dim)

    index, lib_size = build_faiss_index(entries, use_faiss=use_faiss, nprobe=faiss_nprobe)

    # Flattened ordered frame sequence for this sample:
    # list of (layer, frame_num, path)
    def get_ordered_frames(frames_by_layer, all_layers):
        result = []
        for ly in all_layers:
            for fr, path in frames_by_layer[ly]:
                result.append((ly, fr, path))
        return result

    print(f"[Infer] {len(samples)} samples, lib={lib_size} frames, "
          f"skip_layers<={skip_first_layers}")
    print(f"[Infer] Spike: sim > mean(baseline={baseline_window}) + {spike_k}*std + {spike_lower}")
    print(f"[Infer] warmup_frames={warmup_frames}, faiss={'yes' if index is not None else 'no'}")

    results = []
    global_sim_history: deque[float] = deque(maxlen=500)

    for sample in tqdm(samples, desc="Inference"):
        sp = sample["sample_path"]
        if not os.path.isdir(sp):
            print(f"[Infer] WARNING: dir not found: {sp}")
            results.append(_make_result(sample, None, None, None, None, {}, None))
            continue

        frames_by_layer, all_layers = _scan_frames_from_dir(sp)

        # Build ordered frame list, skipping early layers
        ordered_frames = []
        for ly in all_layers:
            if ly <= skip_first_layers:
                continue
            for fr, path in frames_by_layer[ly]:
                ordered_frames.append((ly, fr, path))

        if not ordered_frames:
            results.append(_make_result(sample, None, None, None, None, {}, None))
            continue

        # Per-frame causal search
        frame_sims: list[dict] = []   # {layer, frame_num, max_sim}
        baseline: deque[float] = deque(maxlen=baseline_window)
        pen_detected = False
        pen_frame_layer = None
        pen_frame_num = None
        pen_max_sim = None

        for layer, frame_num, path in ordered_frames:
            roi = _crop_one_image(
                path, roi_size,
                final_roi_scale, cc_min_area, cc_expand_ratio,
                min_laser_pixels, min_laser_area_ratio,
                roi_window_side, use_color_cc_v2_geometry,
            )
            if roi is None:
                continue

            tensor = torch.from_numpy(roi.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                feat = encoder(tensor).cpu().squeeze(0)  # (feat_dim,)

            # Normalize
            feat_n = F.normalize(feat.unsqueeze(0), dim=1).squeeze(0)  # (feat_dim,)
            feat_np = feat_n.numpy().astype(np.float32)

            # Search library
            if index is not None:
                import faiss
                D, I = index.search(feat_np.reshape(1, -1), 1)
                max_sim = float(D[0, 0])
            else:
                scores = np.dot(lib_norm.numpy(), feat_np)
                max_sim = float(scores.max())

            frame_sims.append({
                "layer": layer, "frame_num": frame_num,
                "max_sim": round(max_sim, 4),
            })
            baseline.append(max_sim)
            global_sim_history.append(max_sim)

            # Spike detection after warmup
            if len(baseline) < warmup_frames:
                continue

            bm = sum(baseline) / len(baseline)
            bvar = sum((x - bm) ** 2 for x in baseline) / len(baseline)
            bstd = math.sqrt(bvar)
            threshold = bm + spike_k * bstd + spike_lower

            if max_sim > threshold:
                pen_detected = True
                pen_frame_layer = layer
                pen_frame_num = frame_num
                pen_max_sim = max_sim
                break  # ← CAUSAL STOP at first penetration frame

        # Post-process
        overall_max_sim = max((f["max_sim"] for f in frame_sims), default=None)
        true_layer = sample["penetration_layer"]

        if pen_detected:
            pred_layer = pen_frame_layer
            error = abs(pred_layer - true_layer) if pred_layer is not None else None
        else:
            # No spike detected: penetration_frame=None
            pred_layer = None
            error = None

        results.append(_make_result(
            sample, pred_layer, error, overall_max_sim,
            pen_max_sim,
            {f"{f['layer']}_{f['frame_num']}": f["max_sim"] for f in frame_sims},
            pen_frame_layer,
        ))

    if return_all_sim_history:
        return results, list(global_sim_history)
    return results


def _make_result(sample, pred_layer, error, max_sim, pen_sim,
                 frame_sims, pen_frame_layer):
    """Helper to build a result dict."""
    return {
        "sample_path": sample["sample_path"],
        "is_penetrated": bool(sample["is_penetrated"]),
        "true_layer": sample["penetration_layer"],
        "pred_layer": pred_layer,
        "error": error,
        "max_similarity": round(max_sim, 4) if max_sim is not None else None,
        "penetration_sim": round(pen_sim, 4) if pen_sim is not None else None,
        "pred_penetrated": pred_layer is not None,
        "penetration_frame_layer": pen_frame_layer,
        "frame_similarities": frame_sims,  # {f"{layer}_{frame}": sim}
    }


# ---------------------------------------------------------------------------
# 5. Evaluation & metrics
# ---------------------------------------------------------------------------

def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {"error": "No results"}

    pen = [r for r in results if r["is_penetrated"]]
    non_pen = [r for r in results if not r["is_penetrated"]]
    pen_with_pred = [r for r in pen if r["pred_layer"] is not None]
    pen_with_err = [r for r in pen_with_pred if r["error"] is not None]

    mae = acc3 = acc5 = acc10 = None
    if pen_with_err:
        errs = [r["error"] for r in pen_with_err]
        mae = round(sum(errs) / len(errs), 2)
        acc3 = round(sum(1 for e in errs if e <= 3) / len(errs), 4)
        acc5 = round(sum(1 for e in errs if e <= 5) / len(errs), 4)
        acc10 = round(sum(1 for e in errs if e <= 10) / len(errs), 4)

    pen_det = round(sum(1 for r in pen if r["pred_penetrated"]) / len(pen), 4) if pen else None
    non_pen_det = round(sum(1 for r in non_pen if not r["pred_penetrated"]) / len(non_pen), 4) if non_pen else None

    return {
        "total": total,
        "num_penetrated": len(pen),
        "num_non_penetrated": len(non_pen),
        "num_with_pred": len(pen_with_pred),
        "mae": mae,
        "accuracy_within_3": acc3,
        "accuracy_within_5": acc5,
        "accuracy_within_10": acc10,
        "pen_detection_accuracy": pen_det,
        "non_pen_detection_accuracy": non_pen_det,
    }


def print_metrics(summary: dict, **kwargs):
    print("\n" + "=" * 50)
    print("Inference Results (per-frame spike detection)")
    print("=" * 50)
    print(f"Total samples:          {summary.get('total', 'N/A')}")
    print(f"Penetrated:             {summary.get('num_penetrated', 'N/A')}")
    print(f"Non-penetrated:         {summary.get('num_non_penetrated', 'N/A')}")
    print(f"With layer prediction:   {summary.get('num_with_pred', 'N/A')}")
    print("-" * 50)
    print(f"MAE (layer error):     {summary.get('mae', 'N/A')}")
    print(f"Accuracy <=3 layers:    {summary.get('accuracy_within_3', 'N/A')}")
    print(f"Accuracy <=5 layers:    {summary.get('accuracy_within_5', 'N/A')}")
    print(f"Accuracy <=10 layers:   {summary.get('accuracy_within_10', 'N/A')}")
    print("-" * 50)
    print(f"Pen detection acc:      {summary.get('pen_detection_accuracy', 'N/A')}")
    print(f"Non-pen detection acc:  {summary.get('non_pen_detection_accuracy', 'N/A')}")
    print("=" * 50)
    extras = [f"{k}={v}" for k, v in kwargs.items() if v is not None]
    if extras:
        print(f"({', '.join(extras)})")


# ---------------------------------------------------------------------------
# 6. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Per-frame penetration detection via spike detection + FAISS"
    )
    parser.add_argument("--checkpoint", type=str, default="SimPenDec/stage1.pt")
    parser.add_argument("--train_samples_info", type=str,
                        default="data_drilling/samples_info_train_split.json")
    parser.add_argument("--test_samples_info", type=str,
                        default="data_drilling/samples_info_test.json")
    parser.add_argument("--output_json", type=str,
                        default="grid_diff_tcn/simpen_inference_results.json")
    # Encoder
    parser.add_argument("--dinov3_model", type=str, default="vit_small")
    parser.add_argument("--dinov3_feat_dim", type=int, default=384)
    # ROI — must match pre_crop.py
    parser.add_argument("--roi_size", type=int, default=DEFAULT_ROI_SIZE)
    parser.add_argument("--final_roi_scale", type=float, default=DEFAULT_FINAL_ROI_SCALE)
    parser.add_argument("--cc_min_area", type=int, default=DEFAULT_CC_MIN_AREA)
    parser.add_argument("--cc_expand_ratio", type=float, default=DEFAULT_CC_EXPAND_RATIO)
    parser.add_argument("--min_laser_pixels", type=int, default=DEFAULT_MIN_LASER_PIXELS)
    parser.add_argument("--min_laser_area_ratio", type=float, default=DEFAULT_MIN_LASER_AREA_RATIO)
    parser.add_argument("--roi_window_side", type=int, default=DEFAULT_ROI_WINDOW_SIDE)
    parser.add_argument("--use_color_cc_v2_geometry", type=lambda x: x.lower() == "true",
                        default=DEFAULT_USE_COLOR_CC_V2_GEOMETRY)
    # Inference
    parser.add_argument("--dinov3_chunk_size", type=int, default=32)
    parser.add_argument("--skip_first_layers", type=int, default=30,
                        help="Skip layers <= N (pen never occurs early)")
    parser.add_argument("--baseline_window", type=int, default=20,
                        help="Number of recent frame sims for baseline mean/std")
    parser.add_argument("--spike_k", type=float, default=1.0,
                        help="k in spike_threshold = mean + k*std + lower")
    parser.add_argument("--spike_lower", type=float, default=0.05,
                        help="Absolute lower bound on spike threshold")
    parser.add_argument("--warmup_frames", type=int, default=10,
                        help="Minimum frames before spike detection activates")
    parser.add_argument("--use_faiss", type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--faiss_nprobe", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    repo_root = SCRIPT_DIR.parent
    checkpoint = str(repo_root / args.checkpoint)
    train_info = str(repo_root / args.train_samples_info)
    test_info = str(repo_root / args.test_samples_info)
    output_json = str(repo_root / args.output_json)
    roi_window_side = norm_roi_window_side(args.roi_window_side)

    print("=" * 60)
    print("Per-Frame Penetration Detection")
    print("=" * 60)
    print(f"Checkpoint:      {checkpoint}")
    print(f"Train info:      {train_info}")
    print(f"Test info:       {test_info}")
    print(f"Output:          {output_json}")
    print("-" * 60)
    print(f"ROI (pre_crop.py): roi_size={args.roi_size}, "
          f"final_roi_scale={args.final_roi_scale}")
    print("-" * 60)
    print(f"Spike detection:")
    print(f"  baseline_window={args.baseline_window}, "
          f"spike_k={args.spike_k}, spike_lower={args.spike_lower}")
    print(f"  warmup_frames={args.warmup_frames}, skip_layers<={args.skip_first_layers}")
    print(f"  FAISS: {args.use_faiss}, nprobe={args.faiss_nprobe}")
    print("=" * 60)

    # Load encoder
    encoder = load_encoder_from_simpen(
        checkpoint, dinov3_model=args.dinov3_model,
        dinov3_feat_dim=args.dinov3_feat_dim, dinov3_roi_size=args.roi_size,
    )
    encoder.to(args.device)
    encoder.eval()

    # Build per-frame library
    entries = build_feature_library(
        samples_info_path=train_info,
        encoder=encoder, device=args.device,
        roi_size=args.roi_size, chunk_size=args.dinov3_chunk_size,
        final_roi_scale=args.final_roi_scale,
        cc_min_area=args.cc_min_area, cc_expand_ratio=args.cc_expand_ratio,
        min_laser_pixels=args.min_laser_pixels,
        min_laser_area_ratio=args.min_laser_area_ratio,
        roi_window_side=args.roi_window_side,
        use_color_cc_v2_geometry=args.use_color_cc_v2_geometry,
    )

    # Inference
    results, sim_history = infer_on_testset(
        samples_info_path=test_info,
        encoder=encoder, entries=entries, device=args.device,
        roi_size=args.roi_size, chunk_size=args.dinov3_chunk_size,
        skip_first_layers=args.skip_first_layers,
        baseline_window=args.baseline_window,
        spike_k=args.spike_k, spike_lower=args.spike_lower,
        warmup_frames=args.warmup_frames,
        use_faiss=args.use_faiss, faiss_nprobe=args.faiss_nprobe,
        return_all_sim_history=True,
        final_roi_scale=args.final_roi_scale,
        cc_min_area=args.cc_min_area, cc_expand_ratio=args.cc_expand_ratio,
        min_laser_pixels=args.min_laser_pixels,
        min_laser_area_ratio=args.min_laser_area_ratio,
        roi_window_side=args.roi_window_side,
        use_color_cc_v2_geometry=args.use_color_cc_v2_geometry,
    )

    summary = compute_metrics(results)
    print_metrics(summary,
                 baseline_window=args.baseline_window,
                 spike_k=args.spike_k, spike_lower=args.spike_lower,
                 warmup=args.warmup_frames,
                 skip=args.skip_first_layers)

    output = {
        "config": {k: getattr(args, k, None) for k in [
            "checkpoint", "train_samples_info", "test_samples_info",
            "roi_size", "final_roi_scale", "cc_min_area", "cc_expand_ratio",
            "min_laser_pixels", "min_laser_area_ratio",
            "roi_window_side", "use_color_cc_v2_geometry",
            "dinov3_model", "dinov3_feat_dim",
            "skip_first_layers", "baseline_window", "spike_k", "spike_lower",
            "warmup_frames", "use_faiss", "faiss_nprobe",
        ]},
        "summary": summary,
        "results": results,
    }
    output["config"]["library_frames"] = len(entries)

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to: {output_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

