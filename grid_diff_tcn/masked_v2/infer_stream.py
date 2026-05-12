# -*- coding: utf-8 -*-
"""
Streaming inference for masked_v2 — no batch padding, supports early stopping.

Pipeline
--------
Layer by layer:
  1. Scan sample directory → sorted layer list (no loading yet)
  2. For each layer:
       a. Load frames for this layer only
       b. ROI crop + transform → (F, 3, H, W) tensor
       c. DINOv3 encoder (Stage 1 fine-tuned) → (F, feat_dim) features
       d. HierarchicalGridDiffProbTransformer.forward_step()
          → logits_step, prob_full so far
       e. S3WD / argmax decision on accumulated probability curve
       f. If decision confirmed AND past lock_layers: return immediately

Modes
-----
  Single sample : --sample_dir  <path>         (real-time / production use)
  Multi-sample  : --samples_info <samples.json> (evaluation / batch use)
"""

import argparse
import json
import os
import sys
import time as time_module
from glob import glob as _glob
from typing import Dict, List, Optional, Tuple

import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.insert(0, _REPO_ROOT)

import torch
from tqdm import tqdm

from grid_diff_tcn.common.image_ops import (
    color_cc_extract_gray_letterbox,
    load_image_as_float,
    parse_frame_layer_from_filename,
    to_grayscale,
    _color_cc_resolve_box,
    _crop,
    _resize_rgb_letterbox,
)
from grid_diff_tcn.hier.frame_layer.dinov3_features import DINOV3_FEAT_DIMS, DinoV3FeatureExtractor
from grid_diff_tcn.masked_v2.model import MaskedPixelModel, load_masked_model


# ---------------------------------------------------------------------------
# ROI extraction helpers (copied from dataset.py to avoid importing the whole dataset)
# ---------------------------------------------------------------------------

def _default_transform(roi, target_size: int = 224) -> torch.Tensor:
    """Convert ROI array to (3, H, W) float32 tensor in [0, 1]."""
    import cv2
    if roi.dtype != np.float32 and roi.dtype != np.float64:
        roi = roi.astype(np.float32) / 255.0
    else:
        roi = roi.astype(np.float32)
    if roi.max() > 1.0:
        roi = roi / 255.0
    h, w = roi.shape[:2]
    if h != target_size or w != target_size:
        roi = cv2.resize(roi, (target_size, target_size), interpolation=cv2.INTER_LINEAR)
    if roi.ndim == 2:
        roi = np.stack([roi] * 3, axis=-1)
    tensor = torch.from_numpy(roi).permute(2, 0, 1)  # (3, H, W)
    return tensor


def _load_exclude_set(path: Optional[str]) -> set:
    if not path or not os.path.exists(path):
        return set()
    with open(path, "r", encoding="utf-8") as f:
        return set(json.load(f))


def _build_layer_dict(
    sample_path: str,
    roi_size: int = 224,
    max_frames_per_layer: int = 8,
    use_grayscale: bool = False,
    final_roi_scale: float = 0.85,
    cc_min_area: int = 12,
    cc_expand_ratio: float = 0.2,
    min_laser_pixels: int = 0,
    min_laser_area_ratio: float = 0.0,
    roi_window_side: Optional[int] = None,
    roi_bright_min_ratio: float = 0.0,
    roi_gray_p95_min: float = 0.0,
    use_color_cc_v2_geometry: bool = True,
) -> Dict[int, List[Tuple[int, str]]]:
    """Scan sample directory, return {layer_num: [(frame_num, img_path), ...]}."""
    from collections import defaultdict
    by = defaultdict(list)
    for p in _glob(os.path.join(sample_path, "*.jpg")):
        fn = os.path.basename(p)
        fr, ly = parse_frame_layer_from_filename(fn)
        if fr is None or ly is None:
            continue
        by[int(ly)].append((int(fr), p))
    return by


def _select_frames(items: List[Tuple[int, str]], max_frames: int) -> List[str]:
    """Select up to max_frames evenly-spaced frame paths from a layer."""
    items = sorted(items, key=lambda x: x[0])
    paths = [p for _, p in items]
    if len(paths) <= max_frames:
        return paths
    idx = list(range(0, len(paths), len(paths) // max_frames))[:max_frames]
    return [paths[int(i)] for i in idx]


def _extract_roi_single(
    img_path: str,
    roi_size: int = 224,
    use_grayscale: bool = False,
    final_roi_scale: float = 0.85,
    cc_min_area: int = 12,
    cc_expand_ratio: float = 0.2,
    min_laser_pixels: int = 0,
    min_laser_area_ratio: float = 0.0,
    roi_window_side: Optional[int] = None,
    roi_bright_min_ratio: float = 0.0,
    roi_gray_p95_min: float = 0.0,
    use_color_cc_v2_geometry: bool = True,
):
    """Extract ROI from a single image path. Returns None if extraction fails."""
    img = load_image_as_float(img_path, (roi_size, roi_size))
    if img is None:
        return None

    gray = to_grayscale(img)

    if use_grayscale:
        roi = color_cc_extract_gray_letterbox(
            rgb01=img,
            gray=gray,
            roi_size=roi_size,
            final_roi_scale=final_roi_scale,
            cc_min_area=cc_min_area,
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
            rgb01=img,
            gray=gray,
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

    return roi


# ---------------------------------------------------------------------------
# S3WD decision (same logic as infer_simple.py / train.py)
# ---------------------------------------------------------------------------

def s3wd_decision(
    prob_t: torch.Tensor,
    layers: List[int],
    lock_layers: int,
    wait: int = 3,
    threshold: float = 0.6,
    accept: float = 1.0,
) -> Tuple[int, str]:
    """
    S3WD scan from front to back.

    Returns
        (pred_layer, decision_source):
            "s3wd"            — S3WD scan confirmed a decision
            "argmax_fallback"  — scan found no position, fell back to argmax
            "invalid"         — all positions have zero probability
            "lock_layers"     — never passed lock_layers
    """
    t = len(prob_t)
    consecutive_high = 0

    for ti in range(t):
        if ti < lock_layers:
            continue
        p = prob_t[ti].item()
        if p >= accept:
            return (layers[ti] if ti < len(layers) else layers[-1], "s3wd")
        if p >= threshold:
            consecutive_high += 1
            if consecutive_high >= wait:
                return (layers[ti] if ti < len(layers) else layers[-1], "s3wd")
        else:
            consecutive_high = 0

    # Fallback: argmax over all valid positions (skip lock_layers)
    valid_probs = [(ti, prob_t[ti].item()) for ti in range(t) if ti >= lock_layers]
    if not valid_probs:
        return (-1, "lock_layers")
    best_ti = max(valid_probs, key=lambda x: x[1])[0]
    return (layers[best_ti] if best_ti < len(layers) else layers[-1], "argmax_fallback")


# ---------------------------------------------------------------------------
# Per-sample streaming inference
# ---------------------------------------------------------------------------

def infer_single_sample(
    sample_path: str,
    classifier: torch.nn.Module,
    encoder: torch.nn.Module,
    feat_dim: int,
    roi_size: int = 224,
    max_frames_per_layer: int = 8,
    lock_layers: int = 30,
    s3wd_wait: int = 3,
    s3wd_threshold: float = 0.6,
    s3wd_accept: float = 0.7,
    device: torch.device = torch.device("cpu"),
    roi_kwargs: Optional[dict] = None,
    early_stop: bool = True,
    verbose: bool = False,
) -> dict:
    """
    Stream-inference a single well directory.

    Returns
        dict with keys:
            pred_layer        — predicted penetration layer (-1 if none)
            decision_source    — how decision was made
            num_layers_processed — number of layers processed before return
            total_layers      — total layers in sample
            prob_curve         — list of (layer_idx, physical_layer, prob)
            processed_layers  — list of processed physical layer numbers
    """
    roi_kwargs = roi_kwargs or {}

    # ---- 1. Scan directory, get sorted layer list ----
    t0 = time_module.perf_counter()
    layer_dict = _build_layer_dict(
        sample_path,
        max_frames_per_layer=max_frames_per_layer,
        **roi_kwargs,
    )
    layer_list = sorted(layer_dict.keys())
    t_scan = time_module.perf_counter()

    if not layer_list:
        return {
            "pred_layer": -1,
            "decision_source": "no_layers",
            "num_layers_processed": 0,
            "total_layers": 0,
            "prob_curve": [],
            "processed_layers": [],
        }

    # ---- 2. Reset streaming state ----
    classifier.reset_hidden()
    if hasattr(classifier, "frame_encoder") and hasattr(classifier.frame_encoder, "reset_hidden"):
        classifier.frame_encoder.reset_hidden()

    # Accumulation lists
    all_probs: List[float] = []
    processed_layers: List[int] = []
    decision_source = "none"
    pred_layer = -1
    num_processed = 0

    # ---- 3. Stream through layers ----
    for ti, ly in enumerate(layer_list):
        picks = _select_frames(layer_dict.get(ly, []), max_frames_per_layer)

        # ---- 3a. Load and extract ROI for this layer's frames ----
        frame_tensors = []
        frame_masks = []
        for fp in picks:
            roi = _extract_roi_single(fp, roi_size=roi_size, **roi_kwargs)
            if roi is None:
                frame_masks.append(False)
                continue
            tensor = _default_transform(roi, target_size=roi_size)
            frame_tensors.append(tensor)
            frame_masks.append(True)

        # All frames failed → step with zero input
        if not frame_tensors:
            feat_t = torch.zeros(1, 1, feat_dim, device=device, dtype=torch.float32)
            mask_t = torch.zeros(1, 1, dtype=torch.bool, device=device)
        else:
            # Stack: (F, 3, H, W) → reshape for encoder
            frames_batch = torch.stack(frame_tensors, dim=0).to(device)  # (F, 3, H, W)
            F = frames_batch.shape[0]
            # frame_mask must match feat_encoded length — all True since failed frames already filtered
            mask_t = torch.ones(1, 1, F, dtype=torch.bool, device=device)
            # encoder expects (B, T, F, 3, H, W) — here B=1, T=1
            frames_batch = frames_batch.unsqueeze(0).unsqueeze(0)  # (1, 1, F, 3, H, W)
            feat_flat = frames_batch.reshape(-1, 3, roi_size, roi_size)  # (F, 3, H, W)

            # Extract features through Stage 1 fine-tuned encoder
            with torch.inference_mode():
                feat_encoded = encoder(feat_flat)  # (F, feat_dim)
            feat_encoded = feat_encoded.unsqueeze(0).unsqueeze(0)  # (1, 1, F, feat_dim)

        # ---- 3b. Streaming forward through classifier ----
        # lock_layers masking is applied inside forward_step (mirrors infer_simple.py line 222-223)
        with torch.no_grad():
            result = classifier.forward_step(feat_encoded, frame_mask_step=mask_t, lock_layers=lock_layers)
        prob_full = result["prob_full"].squeeze(0)  # (t,) — all layers so far

        # prob_full length may be < ti+1 on early steps; pad if needed
        current_prob = prob_full[-1].item()
        all_probs.append(current_prob)
        processed_layers.append(ly)
        num_processed += 1

        if verbose:
            print(f"  [Layer {ly:>4}]  prob={current_prob:.4f}  "
                  f"(processed {ti+1})")

        # ---- 3c. S3WD decision on accumulated curve ----
        if ti >= lock_layers:
            prob_tensor = torch.tensor(all_probs, dtype=torch.float32)
            pred_layer, decision_source = s3wd_decision(
                prob_tensor,
                processed_layers,
                lock_layers=0,  # already filtered by ti >= lock_layers
                wait=s3wd_wait,
                threshold=s3wd_threshold,
                accept=s3wd_accept,
            )

            if decision_source == "s3wd" and early_stop:
                if verbose:
                    print(f"  [Decision] S3WD confirmed at layer {ly}, pred={pred_layer}")
                break
            elif decision_source == "argmax_fallback":
                # Continue scanning — argmax can still change as we accumulate evidence
                pass

    # ---- 4. Final decision if not yet decided ----
    if decision_source in ("none", "lock_layers"):
        if len(all_probs) > lock_layers:
            prob_tensor = torch.tensor(all_probs, dtype=torch.float32)
            pred_layer, decision_source = s3wd_decision(
                prob_tensor,
                processed_layers,
                lock_layers=0,
                wait=s3wd_wait,
                threshold=s3wd_threshold,
                accept=s3wd_accept,
            )
        else:
            decision_source = "lock_layers"

    prob_curve = [
        (ti, processed_layers[ti], all_probs[ti])
        for ti in range(len(all_probs))
    ]

    t_end = time_module.perf_counter()
    t_model = t_end - t_scan
    t_total = t_end - t0

    return {
        "pred_layer": pred_layer,
        "decision_source": decision_source,
        "num_layers_processed": num_processed,
        "total_layers": len(layer_list),
        "prob_curve": prob_curve,
        "processed_layers": processed_layers,
        "time_model_sec": round(t_model, 3),
        "time_total_sec": round(t_total, 3),
    }


# ---------------------------------------------------------------------------
# Multi-sample batch inference
# ---------------------------------------------------------------------------

def infer_multi_samples(
    samples_info_path: str,
    classifier: torch.nn.Module,
    encoder: torch.nn.Module,
    feat_dim: int,
    roi_size: int = 224,
    max_frames_per_layer: int = 8,
    lock_layers: int = 30,
    s3wd_wait: int = 3,
    s3wd_threshold: float = 0.6,
    s3wd_accept: float = 0.7,
    device: torch.device = torch.device("cpu"),
    roi_kwargs: Optional[dict] = None,
    early_stop: bool = True,
    max_samples: Optional[int] = None,
    output_csv: Optional[str] = None,
    verbose: bool = False,
) -> List[dict]:
    """Run streaming inference on all samples in a samples_info JSON."""
    t0 = time_module.perf_counter()
    with open(samples_info_path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and "Categories" in raw:
        items = raw.get("Categories", [])
    else:
        items = raw if isinstance(raw, list) else []

    results = []
    iterator = tqdm(items, desc="[Streaming Inference]")
    if max_samples is not None:
        iterator = tqdm(items[:max_samples], desc="[Streaming Inference]")

    for sample in iterator:
        sample_path = sample.get("sample_path", "")
        if not sample_path or not os.path.isdir(sample_path):
            results.append({
                "sample_path": sample_path,
                "pred_layer": -1,
                "decision_source": "path_not_found",
                "true_layer": sample.get("penetration_layer", -1),
                "error": -1,
                "num_layers_processed": 0,
                "total_layers": 0,
            })
            continue

        result = infer_single_sample(
            sample_path=sample_path,
            classifier=classifier,
            encoder=encoder,
            feat_dim=feat_dim,
            roi_size=roi_size,
            max_frames_per_layer=max_frames_per_layer,
            lock_layers=lock_layers,
            s3wd_wait=s3wd_wait,
            s3wd_threshold=s3wd_threshold,
            s3wd_accept=s3wd_accept,
            device=device,
            roi_kwargs=roi_kwargs,
            early_stop=early_stop,
            verbose=verbose,
        )

        true_layer = sample.get("penetration_layer", -1)
        is_penetrated = sample.get("is_penetrated", 0)
        if true_layer >= 0 and result["pred_layer"] >= 0 and result["pred_layer"] in result["processed_layers"]:
            error = abs(result["processed_layers"].index(result["pred_layer"]) -
                        result["processed_layers"].index(true_layer))
        else:
            error = -1

        result["sample_path"] = sample_path
        result["true_layer"] = true_layer
        result["is_penetrated"] = is_penetrated
        result["error"] = error
        results.append(result)

    # ---- Compute metrics ----
    t1 = time_module.perf_counter()
    valid_results = [r for r in results if r.get("is_penetrated") == 1 and r.get("error", -1) >= 0]
    if valid_results:
        total = len(valid_results)
        within_3 = sum(1 for r in valid_results if r["error"] <= 3)
        within_5 = sum(1 for r in valid_results if r["error"] <= 5)
        over_10 = sum(1 for r in valid_results if r["error"] > 10)
        print(f"\n[Metrics] n={total}  "
              f"within_3={within_3}/{total} ({100*within_3/total:.1f}%)  "
              f"within_5={within_5}/{total} ({100*within_5/total:.1f}%)  "
              f"over_10={over_10}/{total} ({100*over_10/total:.1f}%)")
    print(f"[Time] {len(results)} samples in {t1-t0:.1f}s  "
          f"({(t1-t0)/max(len(results),1):.2f}s / sample avg)")

    # ---- Save CSV ----
    if output_csv:
        import csv as csvlib
        rows = []
        for r in results:
            rows.append({
                "sample_path": r["sample_path"],
                "pred_layer": r["pred_layer"],
                "true_layer": r.get("true_layer", -1),
                "error": r.get("error", -1),
                "decision_source": r["decision_source"],
                "num_layers_processed": r["num_layers_processed"],
                "total_layers": r["total_layers"],
                "is_penetrated": r.get("is_penetrated", 0),
                "time_model_sec": r.get("time_model_sec", -1),
                "time_total_sec": r.get("time_total_sec", -1),
            })
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csvlib.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"[Streaming Inference] Results saved to {output_csv}")

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser(
        description="Streaming inference for masked_v2 — no batch padding, supports early stopping.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- Model ----
    p.add_argument("--stage2_checkpoint", type=str, required=True,
                   help="Path to Stage 2 model checkpoint (.pt)")
    p.add_argument("--stage1_checkpoint", type=str, default=None,
                   help="Path to Stage 1 encoder checkpoint. "
                        "If omitted, uses encoder from --stage2_checkpoint")
    p.add_argument("--dinov3_model", type=str, default="vit_small",
                   choices=["vit_small", "vit_base"],
                   help="DINOv3 model variant")
    p.add_argument("--dinov3_feat_dim", type=int, default=384,
                   help="DINOv3 feature dimension")
    p.add_argument("--dinov3_roi_size", type=int, default=224,
                   help="ROI image size fed to DINOv3")
    p.add_argument("--dinov3_chunk_size", type=int, default=4,
                   help="How many frames to process through encoder at once")

    # ---- Inference mode (mutually exclusive) ----
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--sample_dir", type=str,
                  help="Single well directory (production / real-time inference)")
    g.add_argument("--samples_info", type=str,
                  help="Path to samples_info.json for batch evaluation")

    # ---- S3WD parameters ----
    p.add_argument("--lock_layers", type=int, default=30,
                   help="Force prediction = 0 for first N layers (safety lock)")
    p.add_argument("--s3wd_wait", type=int, default=3,
                   help="S3WD: consecutive frames above threshold before confirming")
    p.add_argument("--s3wd_threshold", type=float, default=0.6,
                   help="S3WD: probability threshold")
    p.add_argument("--s3wd_accept", type=float, default=0.7,
                   help="S3WD: immediate decision threshold (1.0=disabled)")

    # ---- ROI extraction ----
    p.add_argument("--max_frames_per_layer", type=int, default=8,
                   help="Max frames to sample per layer")
    p.add_argument("--use_grayscale", action="store_true",
                   help="Use grayscale ROI extraction (slower, more robust)")
    p.add_argument("--final_roi_scale", type=float, default=0.85,
                   help="Final ROI scale factor")
    p.add_argument("--cc_min_area", type=int, default=12,
                   help="Min connected-component area for ROI detection")
    p.add_argument("--cc_expand_ratio", type=float, default=0.2,
                   help="ROI expansion ratio")
    p.add_argument("--roi_window_side", type=int, default=None,
                   help="Fixed ROI window side (None=auto)")
    p.add_argument("--roi_bright_min_ratio", type=float, default=0.0,
                   help="Min bright pixel ratio in ROI window")
    p.add_argument("--roi_gray_p95_min", type=float, default=0.0,
                   help="Min gray p95 for ROI validity")
    p.add_argument("--use_color_cc_v2_geometry", type=int, default=1,
                   help="Use color_cc_v2 geometry for ROI detection")
    p.add_argument("--exclude_json", type=str, default=None,
                   help="Path to JSON file listing images to exclude")

    # ---- Misc ----
    p.add_argument("--device", type=str, default="cuda",
                   help="Device (cuda / cpu)")
    p.add_argument("--early_stop", action="store_true", default=True,
                   help="Stop processing layers once S3WD confirms a decision")
    p.add_argument("--no_early_stop", dest="early_stop", action="store_false",
                   help="Process all layers even after decision is made")
    p.add_argument("--max_samples", type=int, default=None,
                   help="Limit number of samples in batch mode")
    p.add_argument("--output_csv", type=str, default=None,
                   help="Path to save CSV results (batch mode)")
    p.add_argument("--output_probs_csv", type=str, default=None,
                   help="Path to save per-layer probability curves (CSV)")
    p.add_argument("--verbose", action="store_true",
                   help="Print per-layer probability during inference")
    p.add_argument("--print_prob_curve", action="store_true",
                   help="Print final probability curve after inference")

    return p.parse_args()


def main():
    args = _parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[Streaming Inference] Device: {device}")

    # ---- Build encoder + classifier ----
    # The encoder = DINOv3 (from Stage 1 checkpoint)
    # The classifier = HierarchicalGridDiffProbTransformer (from Stage 2 checkpoint)
    # We load Stage 1 encoder into a MaskedPixelModel, then extract its components.

    print(f"[Streaming Inference] Loading Stage 1 encoder: {args.stage1_checkpoint or args.stage2_checkpoint}")
    enc_ckpt = args.stage1_checkpoint or args.stage2_checkpoint
    encoder_model = load_masked_model(
        checkpoint_path=args.stage2_checkpoint,
        stage=2,
        encoder_checkpoint=enc_ckpt,
        dinov3_model=args.dinov3_model,
        dinov3_feat_dim=args.dinov3_feat_dim,
        dinov3_roi_size=args.dinov3_roi_size,
        dinov3_chunk_size=args.dinov3_chunk_size,
        use_cached_features=False,
    )
    encoder_model = encoder_model.to(device)
    encoder_model.eval()

    # Determine feature dimension from encoder model
    feat_dim = DINOV3_FEAT_DIMS.get(args.dinov3_model, args.dinov3_feat_dim)

    # The encoder that produces DINOv3 features for the classifier
    encoder = encoder_model.dinov3_extractor  # DinoV3FeatureExtractor

    # The Stage 2 classifier (HierarchicalGridDiffProbTransformer)
    classifier = encoder_model.classifier
    classifier = classifier.to(device)
    classifier.eval()

    print(f"  encoder (DINOv3): {type(encoder).__name__}")
    print(f"  feat_dim:        {feat_dim}")
    print(f"  classifier:       {type(classifier).__name__}")

    # ---- ROI extraction kwargs ----
    roi_kwargs = {
        "use_grayscale": args.use_grayscale,
        "final_roi_scale": args.final_roi_scale,
        "cc_min_area": args.cc_min_area,
        "cc_expand_ratio": args.cc_expand_ratio,
        "roi_window_side": args.roi_window_side,
        "roi_bright_min_ratio": args.roi_bright_min_ratio,
        "roi_gray_p95_min": args.roi_gray_p95_min,
        "use_color_cc_v2_geometry": bool(args.use_color_cc_v2_geometry),
    }

    # ---- Single-sample mode ----
    if args.sample_dir:
        print(f"\n[Single Sample] {args.sample_dir}")
        result = infer_single_sample(
            sample_path=args.sample_dir,
            classifier=classifier,
            encoder=encoder,
            feat_dim=feat_dim,
            roi_size=args.dinov3_roi_size,
            max_frames_per_layer=args.max_frames_per_layer,
            lock_layers=args.lock_layers,
            s3wd_wait=args.s3wd_wait,
            s3wd_threshold=args.s3wd_threshold,
            s3wd_accept=args.s3wd_accept,
            device=device,
            roi_kwargs=roi_kwargs,
            early_stop=args.early_stop,
            verbose=args.verbose,
        )

        print(f"\n[Result]")
        print(f"  Predicted layer  : {result['pred_layer']}")
        print(f"  Decision source : {result['decision_source']}")
        print(f"  Layers processed: {result['num_layers_processed']} / {result['total_layers']}")
        print(f"  Model time      : {result['time_model_sec']}s")
        print(f"  Total time      : {result['time_total_sec']}s")

        if args.print_prob_curve or args.verbose:
            print(f"\n[Probability Curve]")
            print(f"  {'Idx':>4}  {'Layer':>6}  {'Prob':>8}  {'穿透':>4}")
            print(f"  {'-'*4}  {'-'*6}  {'-'*8}  {'-'*4}")
            for ti, ly, prob in result["prob_curve"]:
                penetrated = "✓" if prob >= args.s3wd_accept else ""
                print(f"  {ti:>4}  {ly:>6}  {prob:>8.4f}  {penetrated:>4}")

        if args.output_probs_csv:
            import csv as csvlib
            with open(args.output_probs_csv, "w", newline="", encoding="utf-8") as f:
                writer = csvlib.DictWriter(f, fieldnames=[
                    "layer_idx", "physical_layer", "prob", "penetrated"])
                writer.writeheader()
                for ti, ly, prob in result["prob_curve"]:
                    writer.writerow({
                        "layer_idx": ti,
                        "physical_layer": ly,
                        "prob": f"{prob:.6f}",
                        "penetrated": 1 if prob >= args.s3wd_accept else 0,
                    })
            print(f"\n[Prob curve saved to] {args.output_probs_csv}")

        return

    # ---- Multi-sample mode ----
    if args.samples_info:
        print(f"\n[Batch Mode] {args.samples_info}")
        results = infer_multi_samples(
            samples_info_path=args.samples_info,
            classifier=classifier,
            encoder=encoder,
            feat_dim=feat_dim,
            roi_size=args.dinov3_roi_size,
            max_frames_per_layer=args.max_frames_per_layer,
            lock_layers=args.lock_layers,
            s3wd_wait=args.s3wd_wait,
            s3wd_threshold=args.s3wd_threshold,
            s3wd_accept=args.s3wd_accept,
            device=device,
            roi_kwargs=roi_kwargs,
            early_stop=args.early_stop,
            max_samples=args.max_samples,
            output_csv=args.output_csv,
            verbose=args.verbose,
        )

        if args.output_probs_csv:
            import csv as csvlib
            rows = []
            for r in results:
                for ti, ly, prob in r.get("prob_curve", []):
                    rows.append({
                        "sample": r["sample_path"],
                        "layer_idx": ti,
                        "physical_layer": ly,
                        "prob": prob,
                        "pred_layer": r["pred_layer"],
                        "decision_source": r["decision_source"],
                    })
            with open(args.output_probs_csv, "w", newline="", encoding="utf-8") as f:
                writer = csvlib.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(rows)
            print(f"[Prob curves saved to] {args.output_probs_csv}")
        return


if __name__ == "__main__":
    main()
