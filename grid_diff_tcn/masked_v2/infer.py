# -*- coding: utf-8 -*-
"""
Inference script for masked_v2 model (Stage 2).

Two data modes:
  1. Cached ROI (default): loads pre-cropped .pt files from roi_cache/ via samples_info allowlist.
  2. Online cropping: runs ROI extraction on raw images (MaskedDrillingDataset).

Inference pipeline mirrors train.py evaluate_stage2():
  - Run model forward to collect raw per-frame probabilities.
  - Grid-search S3WD (wait, threshold) on validation set to find best params.
  - Apply best params on test set and output per-well CSV.

CSV output columns: hole_path, true_layer, pred_layer, error
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from grid_diff_tcn.masked_v2.model import MaskedPixelModel, load_masked_model
from grid_diff_tcn.masked_v2.dataset import MaskedDrillingDataset, collate_masked_batch


# ---------------------------------------------------------------------------
# ROICacheDataset — direct .pt loader from roi_cache/
# ---------------------------------------------------------------------------

def _sample_path_to_cache_path(sample_path: str, cache_dir: str) -> str:
    """
    Derive the expected .pt file path from a sample_path.
    Uses the same naming convention as pre_crop.py:
      rel = os.path.relpath(sample_path, cwd)
      safe = rel.replace(os.sep, "_").replace("/", "_")
      cache_path = cache_dir / f"{safe}.pt"
    """
    try:
        rel = os.path.relpath(sample_path, os.getcwd())
    except ValueError:
        rel = sample_path
    safe = rel.replace(os.sep, "_").replace("/", "_")
    return os.path.join(cache_dir, f"{safe}.pt")


class ROICacheDataset(Dataset):
    """
    Loads pre-cropped .pt files from roi_cache/.

    .pt structure (from pre_crop.py):
        frames: (T, F, 3, H, W) float32, 0-1 normalized
        mask:   (T, F) bool
        layers: list[int]
        sample_path: str (original source path)

    If samples_info_path is provided, it acts as an allowlist: only samples
    listed in the JSON are returned.

    File lookup is LAZY: the expected .pt path is derived from sample_path
    at __getitem__ time, so init is fast (no full-cache scan).

    __getitem__ returns the same dict format as MaskedDrillingDataset,
    so collate_fn = collate_masked_batch works unchanged.
    """

    def __init__(
        self,
        cache_dir: str,
        samples_info_path: Optional[str] = None,
        roi_size: int = 224,
        max_layers: Optional[int] = None,
        max_frames_per_layer: int = 8,
        preload: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.cache_dir = str(cache_dir)
        self.roi_size = int(roi_size)
        self.max_layers = max_layers
        self.max_frames_per_layer = int(max_frames_per_layer)
        self.preload = preload

        # Load samples_info as allowlist
        if samples_info_path and os.path.exists(samples_info_path):
            with open(samples_info_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            if isinstance(raw, dict) and "Categories" in raw:
                raw = raw.get("Categories", [])

            self.samples = [
                {
                    "sample_path": str(it.get("sample_path", "")),
                    "is_penetrated": int(it.get("is_penetrated", 0)),
                    "penetration_layer": int(it.get("penetration_layer", -1)),
                }
                for it in raw
            ]
        else:
            self.samples = []

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        # Precompute expected .pt paths (lazy — only check existence at getitem time)
        self._cache_map: Dict[str, str] = {
            s["sample_path"]: _sample_path_to_cache_path(s["sample_path"], cache_dir)
            for s in self.samples
            if s["sample_path"]
        }
        found = sum(
            1 for s in self.samples
            if s["sample_path"] and os.path.exists(self._cache_map.get(s["sample_path"], ""))
        )
        print(f"[ROICacheDataset] cache_dir={cache_dir}  "
              f"allowlist_samples={len(self.samples)}  "
              f"cache_files_found={found}")

        # Preload: load .pt into RAM only for matched samples
        self._preloaded: List[Optional[dict]] = [None] * len(self.samples)
        if self.preload:
            print(f"[ROICacheDataset] Preloading {len(self.samples)} .pt files into RAM...")
            for si, s in enumerate(self.samples):
                cp = self._cache_map.get(s["sample_path"])
                if cp and os.path.exists(cp):
                    try:
                        self._preloaded[si] = torch.load(
                            cp, map_location="cpu", weights_only=False)
                    except Exception:
                        pass
            print("[ROICacheDataset] Preload done.")

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

        if cached is None:
            frame_data = torch.zeros(
                1, self.max_frames_per_layer, 3, self.roi_size, self.roi_size,
                dtype=torch.float32,
            )  # (T=1, F, 3, H, W)
            frame_mask = torch.zeros(1, self.max_frames_per_layer, dtype=torch.bool)
            seq_label = torch.zeros(1, dtype=torch.int64)
            layer_list = [0]
        else:
            raw_frames = cached["frames"]          # (T, F, 3, H, W)
            raw_mask: torch.Tensor = cached["mask"]  # (T, F) bool
            layer_list: List[int] = cached.get("layers", list(range(raw_frames.shape[0])))

            # Truncate T first (if max_layers set), then pad F to max_frames_per_layer
            if self.max_layers and len(layer_list) > self.max_layers:
                layer_list = layer_list[: self.max_layers]
                raw_frames = raw_frames[: self.max_layers]
                raw_mask = raw_mask[: self.max_layers]

            # Ensure (T, F, 3, H, W) with F == max_frames_per_layer
            T, F, C, H, W = raw_frames.shape
            if F < self.max_frames_per_layer:
                pad_f = self.max_frames_per_layer - F
                raw_frames = torch.nn.functional.pad(raw_frames, (0, 0, 0, 0, 0, 0, 0, pad_f))
                raw_mask = torch.nn.functional.pad(raw_mask, (0, pad_f), value=False)

            frame_data = raw_frames
            frame_mask = raw_mask

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
        }


# ---------------------------------------------------------------------------
# S3WD decision & evaluation logic (mirrors train.py evaluate_stage2)
# ---------------------------------------------------------------------------

def s3wd_decision(
    prob_t: torch.Tensor,
    mask_t: torch.Tensor,
    pen_layer: int,
    layers: List[int],
    wait: int,
    thresh: float,
    accept: float = 0.5,
    lock_layers: int = 30,
) -> int:
    """
    S3WD: require `wait` consecutive valid timesteps above `thresh`.
    Returns the predicted layer index (0-based in layers), or -1 if not penetrated.
    The `accept` param controls the minimum probability to accept a decision
    (if the max valid probability is below accept, return -1).
    """
    if pen_layer < 0 or pen_layer not in layers:
        return -1
    pen_idx = layers.index(pen_layer)
    if pen_idx < lock_layers:
        return -1
    consecutive_high = 0
    best_valid_prob = 0.0
    best_layer = -1
    for ti in range(len(prob_t)):
        if not mask_t[ti]:
            continue
        p = float(prob_t[ti])
        if p > best_valid_prob:
            best_valid_prob = p
            best_layer = ti
        if p >= thresh:
            consecutive_high += 1
            if consecutive_high >= wait:
                if best_valid_prob >= accept:
                    return best_layer
                else:
                    return -1
        else:
            consecutive_high = 0
    return -1


def compute_metrics(
    preds: List[int],
    labels: List[int],
    penetration_layers: List[int],
    layer_lists: List[List[int]],
    lock_layers: int = 30,
) -> dict:
    """
    Compute penetration layer error distribution (mirrors train.py compute_metrics).
    Only samples with label==1, valid pen_layer in layer_lists, and valid pred
    contribute to the layer error stats.
    """
    within_3 = within_5 = over_10 = total = 0

    for pred, label, pen_layer, layers in zip(preds, labels, penetration_layers, layer_lists):
        if label != 1:
            continue
        if pen_layer < 0 or pen_layer not in layers:
            continue
        pen_idx = layers.index(pen_layer)
        if pen_idx < lock_layers:
            continue
        if pred < 0:
            # No penetration predicted — count as >10 error
            total += 1
            over_10 += 1
            continue
        if pred not in layers:
            total += 1
            over_10 += 1
            continue

        true_idx = layers.index(pen_layer)
        pred_idx = layers.index(pred)
        error = abs(pred_idx - true_idx)

        total += 1
        if error <= 3:
            within_3 += 1
        if error <= 5:
            within_5 += 1
        if error > 10:
            over_10 += 1

    if total == 0:
        return {"pct_within_3": 0.0, "pct_within_5": 0.0, "pct_over_10": 0.0, "total": 0}

    return {
        "pct_within_3": within_3 / total,
        "pct_within_5": within_5 / total,
        "pct_over_10": over_10 / total,
        "total": total,
    }


def grid_search_s3wd(
    model: MaskedPixelModel,
    loader: DataLoader,
    device: torch.device,
    lock_layers: int = 30,
) -> tuple[dict, List[dict]]:
    """
    Run model forward on the dataset, collect per-frame probs, then grid-search
    S3WD (wait, threshold, accept) to maximise pct_within_3 + pct_within_5.

    Returns:
        best_params: dict with best wait, threshold, accept
        all_results: list of metric dicts per param combo
    """
    model.eval()
    all_labels: List[int] = []
    all_pen_layers: List[int] = []
    all_layer_lists: List[List[int]] = []
    all_frame_probs: List[tuple] = []  # (prob_t_cpu, mask_t_cpu)

    with torch.inference_mode():
        for batch in tqdm(loader, desc="[GridSearch] Collecting probs"):
            frame_data = batch["frame_data"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            labels = batch["labels"]
            pen_layers = batch["penetration_layers"]
            layer_lists_batch = batch["layer_lists"]

            result = model.forward(frame_data, frame_mask=frame_mask)
            logits = result["logits"]

            # Apply lock_layers
            if lock_layers > 0:
                logits_locked = logits.clone()
                logits_locked[:, 1, :lock_layers] = float("-inf")
            else:
                logits_locked = logits

            probs = F.softmax(logits_locked, dim=1)[:, 1]  # (B, T)

            for bi in range(logits.shape[0]):
                prob_t = probs[bi].cpu()
                mask_bi = frame_mask[bi]                     # (T, F)
                mask_t = mask_bi.any(dim=1)                  # (T,)
                all_frame_probs.append((prob_t, mask_t))
                all_labels.append(int(labels[bi].item()))
                all_pen_layers.append(int(pen_layers[bi].item()))
                all_layer_lists.append(layer_lists_batch[bi])

    # Grid search
    thresholds = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
    accept_vals = [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
    wait_vals = [3, 4, 5, 6, 7, 8, 9, 10]

    best_metric = -1.0
    best_params = {}
    all_results = []

    for wait in wait_vals:
        for thresh in thresholds:
            for accept in accept_vals:
                preds = []
                for (prob_t, mask_t), label, pen_layer, layers in zip(
                        all_frame_probs, all_labels, all_pen_layers, all_layer_lists):
                    if label != 1:
                        preds.append(-1)
                        continue
                    pred_layer = s3wd_decision(
                        prob_t, mask_t, pen_layer, layers,
                        wait=wait, thresh=thresh, accept=accept, lock_layers=lock_layers,
                    )
                    preds.append(pred_layer)

                metrics = compute_metrics(
                    preds, all_labels, all_pen_layers, all_layer_lists,
                    lock_layers=lock_layers,
                )
                combined = metrics["pct_within_3"] + metrics["pct_within_5"]
                metrics["params"] = {"wait": wait, "threshold": thresh, "accept": accept}
                all_results.append(metrics)

                if combined >= best_metric:
                    best_metric = combined
                    best_params = {"wait": wait, "threshold": thresh, "accept": accept}

    print(f"\n[S3WD GridSearch] Best: wait={best_params['wait']}, "
          f"thresh={best_params['threshold']:.2f}, accept={best_params['accept']:.2f}, "
          f"<=3+<=5={best_metric*100:.1f}%")
    if all_results:
        top3 = sorted(all_results, key=lambda x: -(x["pct_within_3"] + x["pct_within_5"]))[:3]
        for r in top3:
            print(f"  wait={r['params']['wait']}, thresh={r['params']['threshold']:.2f}, "
                  f"accept={r['params']['accept']:.2f}  "
                  f"<=3:{r['pct_within_3']*100:.1f}%  <=5:{r['pct_within_5']*100:.1f}%  "
                  f">=10:{r['pct_over_10']*100:.1f}%  n={r['total']}")

    return best_params, all_results


def run_inference(
    model: MaskedPixelModel,
    loader: DataLoader,
    dataset: Dataset,
    device: torch.device,
    decision_method: str = "s3wd",
    s3wd_wait: int = 5,
    s3wd_thresh: float = 0.6,
    s3wd_accept: float = 0.3,
    lock_layers: int = 30,
    best_s3wd_params: Optional[dict] = None,
) -> tuple[List[dict], dict]:
    """
    Run inference on a dataset.

    If decision_method == "s3wd" and best_s3wd_params is provided,
    those params override s3wd_wait / s3wd_thresh / s3wd_accept.

    If decision_method == "learned", the LearnedDecisionHead is used directly
    and decision_idx (rounded) becomes the prediction — no threshold tuning needed.

    Returns:
        csv_rows: list of dicts with hole_path, true_layer, pred_layer, error
        metrics: dict with aggregate metrics
    """
    model.eval()

    all_labels: List[int] = []
    all_pen_layers: List[int] = []
    all_layer_lists: List[List[int]] = []
    all_sample_paths: List[str] = []
    all_frame_probs: List[tuple] = []
    all_decision_idx_raw: List[float] = []  # raw float predictions from learned head

    with torch.inference_mode():
        for batch in tqdm(loader, desc="[Inference] Running model"):
            frame_data = batch["frame_data"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            labels = batch["labels"]
            pen_layers = batch["penetration_layers"]
            layer_lists_batch = batch["layer_lists"]
            sample_paths_batch = batch["sample_paths"]

            result = model.forward(
                frame_data, frame_mask=frame_mask,
                return_decision_idx=use_learned_decision,
            )
            logits = result["logits"]

            if lock_layers > 0:
                logits_locked = logits.clone()
                logits_locked[:, 1, :lock_layers] = float("-inf")
            else:
                logits_locked = logits

            probs = F.softmax(logits_locked, dim=1)[:, 1]  # (B, T)

            # Collect learned decision indices if available
            decision_idx_batch = None
            if use_learned_decision:
                decision_idx_batch = result.get("decision_idx")
                if decision_idx_batch is not None:
                    decision_idx_batch = decision_idx_batch.cpu()

            for bi in range(logits.shape[0]):
                prob_t = probs[bi].cpu()
                mask_bi = frame_mask[bi]
                mask_t = mask_bi.any(dim=1)
                all_frame_probs.append((prob_t, mask_t))
                all_labels.append(int(labels[bi].item()))
                all_pen_layers.append(int(pen_layers[bi].item()))
                all_layer_lists.append(layer_lists_batch[bi])
                all_sample_paths.append(sample_paths_batch[bi])
                if decision_idx_batch is not None:
                    all_decision_idx_raw.append(float(decision_idx_batch[bi].item()))
                else:
                    all_decision_idx_raw.append(0.0)

    # Decide effective params
    if decision_method == "s3wd" and best_s3wd_params:
        eff_wait = best_s3wd_params.get("wait", s3wd_wait)
        eff_thresh = best_s3wd_params.get("threshold", s3wd_thresh)
        eff_accept = best_s3wd_params.get("accept", s3wd_accept)
    else:
        eff_wait = s3wd_wait
        eff_thresh = s3wd_thresh
        eff_accept = s3wd_accept

    preds = []
    for si, ((prob_t, mask_t), label, pen_layer, layers, raw_idx) in enumerate(zip(
            all_frame_probs, all_labels, all_pen_layers, all_layer_lists, all_decision_idx_raw)):
        if decision_method == "learned":
            # LearnedDecisionHead: 直接用 round(raw_idx) 作为预测
            # lock_layers 安全锁：如果真实穿透层在 lock_layers 之前，预测 -1
            if label == 1 and pen_layer in layers:
                pen_idx = layers.index(pen_layer)
                if pen_idx >= lock_layers:
                    pred_layer_idx = round(raw_idx)
                    if 0 <= pred_layer_idx < len(layers):
                        pred_layer = layers[pred_layer_idx]
                    else:
                        # 越界：clamp 到有效范围
                        clamped = max(0, min(pred_layer_idx, len(layers) - 1))
                        pred_layer = layers[clamped]
                else:
                    pred_layer = -1
            else:
                pred_layer = -1
        elif decision_method == "s3wd":
            pred_layer = s3wd_decision(
                prob_t, mask_t, pen_layer, layers,
                wait=eff_wait, thresh=eff_thresh, accept=eff_accept, lock_layers=lock_layers,
            )
        elif decision_method == "topkmedian":
            valid_probs = prob_t[mask_t]
            valid_idx = torch.where(mask_t)[0]
            if valid_probs.numel() == 0:
                pred_layer = -1
            else:
                k_actual = min(9, valid_probs.numel())
                topk_probs, topk_indices = valid_probs.topk(k_actual)
                if topk_probs.median() >= eff_thresh:
                    pred_layer = valid_idx[topk_indices[topk_indices == topk_probs.argmax()]].item()
                else:
                    pred_layer = -1
        else:  # "threshold" — argmax
            valid_probs = prob_t[mask_t]
            valid_idx = torch.where(mask_t)[0]
            if valid_probs.numel() == 0:
                pred_layer = -1
            else:
                max_idx = valid_probs.argmax()
                pred_layer = valid_idx[max_idx].item()
                if valid_probs.max() < eff_thresh:
                    pred_layer = -1
        preds.append(pred_layer)

    # Build CSV rows
    csv_rows = []
    for sp, true_layer, pred_layer, layers, raw_idx in zip(
            all_sample_paths, all_pen_layers, preds, all_layer_lists, all_decision_idx_raw):
        if true_layer >= 0 and pred_layer >= 0 and pred_layer in layers:
            true_idx = layers.index(true_layer)
            pred_idx = layers.index(pred_layer)
            error = abs(pred_idx - true_idx)
        else:
            error = -1
        csv_rows.append({
            "hole_path": sp,
            "true_layer": true_layer,
            "pred_layer": pred_layer,
            "error": error,
            "raw_decision_idx": round(raw_idx, 3),
        })

    # Compute aggregate metrics
    metrics = compute_metrics(
        preds, all_labels, all_pen_layers, all_layer_lists,
        lock_layers=lock_layers,
    )
    metrics["decision_method"] = decision_method
    if decision_method == "s3wd":
        metrics["s3wd_params"] = {"wait": eff_wait, "threshold": eff_thresh, "accept": eff_accept}

    return csv_rows, metrics


# ---------------------------------------------------------------------------
# Streaming inference: early stopping on penetration detection
# ---------------------------------------------------------------------------

def _streaming_s3wd_step(
    prob_t: float,
    thresh: float,
    accept: float,
    wait: int,
    consecutive_high: int,
    best_prob: float,
    best_idx: int,
) -> tuple[int, int, float, int]:
    """
    One step of S3WD streaming logic.
    Returns (new_consecutive_high, new_best_prob, new_best_idx, decision).

    decision: -1 = no decision yet, >= 0 = predicted layer index (0-based)
    """
    if prob_t >= thresh:
        consecutive_high += 1
        if consecutive_high >= wait:
            if best_prob >= accept:
                return consecutive_high, best_prob, best_idx, best_idx
            else:
                return consecutive_high, best_prob, best_idx, -1
    else:
        consecutive_high = 0
    return consecutive_high, best_prob, best_idx, -1


def _streaming_decision_step(
    prob_step: float,
    decision_idx: float,
    thresh: float,
    decision_method: str,
) -> int:
    """
    Streaming decision for one step.
    decision_method "learned": use decision_idx round
    decision_method "threshold": use prob_step > thresh
    Returns: -1 = no decision, >= 0 = predicted layer index
    """
    if decision_method == "learned":
        pred = round(decision_idx)
        if 0 <= pred:
            return pred
        return -1
    else:  # "threshold"
        if prob_step >= thresh:
            return 0  # predicted layer 0 at this step
        return -1


def run_streaming_inference(
    model: torch.nn.Module,
    dataset: Dataset,
    device: torch.device,
    lock_layers: int = 30,
    decision_method: str = "learned",
    stop_thresh: float = 0.5,
    stop_wait: int = 3,
    max_inference_layers: int | None = None,
    return_frame_probs: bool = False,
) -> tuple[List[dict], dict, List[tuple] | None]:
    """
    Streaming inference: process layers one at a time with early stopping.

    As soon as penetration is detected, stop reading further layers.
    This mirrors the real drilling process where you stop when penetration occurs.

    Args:
        model: MaskedPixelModel (Stage 2)
        dataset: ROICacheDataset or MaskedDrillingDataset
        device
        lock_layers: layer index below which we don't predict penetration
        decision_method: "learned" | "threshold"
        stop_thresh: penetration probability threshold for early stopping
        stop_wait: number of consecutive steps above stop_thresh to confirm
        max_inference_layers: cap on how many layers to process (for non-penetrated samples)
        return_frame_probs: if True, also return frame probabilities for analysis

    Returns:
        csv_rows, metrics, frame_probs_list (or None)
    """
    model.eval()

    # Get feature extractor for cached feature mode
    has_features = hasattr(model, "get_features")

    all_labels: List[int] = []
    all_pen_layers: List[int] = []
    all_layer_lists: List[List[int]] = []
    all_sample_paths: List[str] = []
    all_frame_probs: List[tuple] = []
    all_decision_idx_raw: List[float] = []
    early_stop_counts: List[int] = []  # layers processed before early stop
    total_layers_processed: List[int] = []  # total layers (for comparison)

    stop_wait = int(stop_wait)

    with torch.inference_mode():
        for si in tqdm(range(len(dataset)), desc="[Streaming Inference]"):
            sample = dataset[si]

            frame_data: torch.Tensor = sample["frame_data"]      # (1, T, F, 3, H, W) or (T, F, 3, H, W)
            frame_mask: torch.Tensor = sample["frame_mask"]       # (1, T, F) or (T, F)
            label = int(sample["label"])
            pen_layer = int(sample["penetration_layer"])
            layer_list: List[int] = sample["layer_list"]
            sample_path = str(sample.get("sample_path", ""))

            # Normalize to (T, F, ...) — handle both batched and unbatched
            if frame_data.dim() == 5:  # (1, T, F, 3, H, W) or (1, T, F, 3, H, W) with T=1
                if frame_data.shape[0] == 1:
                    frame_data = frame_data.squeeze(0)            # (T, F, 3, H, W)
                    frame_mask = frame_mask.squeeze(0)            # (T, F)
            T = frame_data.shape[0]
            F = frame_data.shape[1]

            # If model uses cached features (pre-extracted), extract now
            # Otherwise model.forward will handle it internally
            features: torch.Tensor | None = None
            if has_features and model.use_cached_features:
                # frame_data is (T, F, C) cached features
                features = frame_data.unsqueeze(0).to(device)     # (1, T, F, C)
            elif has_features and isinstance(frame_data, torch.Tensor) and frame_data.dim() == 3:
                # frame_data is (T, F, C) already features
                features = frame_data.unsqueeze(0).to(device)
            else:
                # frame_data is (T, F, 3, H, W) raw images → forward handles internally
                features = None

            # Reset classifier streaming state
            model.classifier.reset_hidden()

            consecutive_high = 0
            best_prob = 0.0
            best_idx = 0
            stop_decided = False
            stop_idx = -1  # layer index at which we stopped
            early_stop_at = -1  # 0-based layer index where early stop triggered

            all_probs_this_sample: List[float] = []
            raw_idx_this_sample: List[float] = []

            max_steps = min(T, max_inference_layers) if max_inference_layers else T

            for ti in range(max_steps):
                # Extract feature for this layer
                if features is not None:
                    # features: (1, T, F, C)
                    feat_t = features[:, ti:ti+1, :, :]          # (1, 1, F, C)
                    mask_t = frame_mask[ti:ti+1].unsqueeze(0)     # (1, 1, F)
                else:
                    # frame_data: (T, F, 3, H, W)
                    img_t = frame_data[ti:ti+1].unsqueeze(0)     # (1, 1, F, 3, H, W)
                    mask_t = frame_mask[ti:ti+1].unsqueeze(0)    # (1, 1, F)
                    feat_t = img_t                                 # (1, 1, F, 3, H, W)

                feat_t_dev = feat_t.to(device)
                mask_t_dev = mask_t.to(device)

                # Streaming forward step
                result = model.classifier.forward_step(feat_t_dev, frame_mask_step=mask_t_dev)
                prob_step = float(result["prob_step"].cpu().item())  # (B, 1) -> scalar
                decision_idx_step = float(result["decision_idx"].cpu().item())

                all_probs_this_sample.append(prob_step)
                raw_idx_this_sample.append(decision_idx_step)

                # S3WD-style early stopping check
                if prob_step >= stop_thresh:
                    consecutive_high += 1
                    if prob_step > best_prob:
                        best_prob = prob_step
                        best_idx = ti
                    if consecutive_high >= stop_wait:
                        early_stop_at = ti
                        stop_idx = best_idx
                        stop_decided = True
                        break
                else:
                    consecutive_high = 0

            # For non-penetrated or lock_layers samples: predict -1
            if label == 0 or (label == 1 and (pen_layer < 0 or pen_layer not in layer_list)):
                final_pred_layer = -1
                final_pred_idx = -1.0
            elif label == 1 and pen_layer in layer_list:
                pen_idx = layer_list.index(pen_layer)
                if pen_idx < lock_layers:
                    final_pred_layer = -1
                    final_pred_idx = -1.0
                else:
                    if stop_decided:
                        final_pred_layer = layer_list[stop_idx] if stop_idx < len(layer_list) else layer_list[-1]
                        final_pred_idx = float(stop_idx)
                    else:
                        # No early stop triggered — use learned decision idx for full sequence
                        full_probs = torch.tensor(all_probs_this_sample)
                        full_logits = torch.stack([
                            torch.zeros_like(full_probs),
                            full_probs,
                        ], dim=1).unsqueeze(0)  # (1, 2, T)
                        full_z = torch.zeros(1, len(all_probs_this_sample), 128)
                        try:
                            decision_idx_full = float(
                                model.classifier.decision_head(
                                    full_z, full_logits, None
                                ).cpu().item()
                            )
                        except Exception:
                            decision_idx_full = raw_idx_this_sample[-1] if raw_idx_this_sample else 0.0
                        pred_layer_idx = round(decision_idx_full)
                        pred_layer_idx = max(0, min(pred_layer_idx, len(layer_list) - 1))
                        final_pred_layer = layer_list[pred_layer_idx]
                        final_pred_idx = decision_idx_full

            all_labels.append(label)
            all_pen_layers.append(pen_layer)
            all_layer_lists.append(layer_list)
            all_sample_paths.append(sample_path)
            all_frame_probs.append((torch.tensor(all_probs_this_sample), torch.ones(len(all_probs_this_sample), dtype=torch.bool)))
            all_decision_idx_raw.append(final_pred_idx)
            early_stop_counts.append(early_stop_at + 1 if early_stop_at >= 0 else -1)
            total_layers_processed.append(max_steps)

    # Build CSV rows
    csv_rows = []
    for idx in range(len(all_sample_paths)):
        sp = all_sample_paths[idx]
        true_label = all_labels[idx]
        pen_l = all_pen_layers[idx]
        layers = all_layer_lists[idx]
        raw_idx = all_decision_idx_raw[idx]
        es_count = early_stop_counts[idx]
        tot_layers = total_layers_processed[idx]

        # Determine pred_layer from stored data
        if true_label == 0 or (true_label == 1 and (pen_l < 0 or pen_l not in layers)):
            pred_layer = -1
            error = -1
        elif true_label == 1 and pen_l in layers:
            pen_idx = layers.index(pen_l)
            if pen_idx < lock_layers:
                pred_layer = -1
                error = -1
            else:
                # raw_idx is the 0-based layer index predicted by the model
                if raw_idx >= 0:
                    pred_layer_idx_clamped = max(0, min(int(round(raw_idx)), len(layers) - 1))
                    pred_layer = layers[pred_layer_idx_clamped]
                    true_idx = layers.index(pen_l)
                    error = abs(pred_layer_idx_clamped - true_idx)
                else:
                    pred_layer = -1
                    error = -1
        else:
            pred_layer = -1
            error = -1

        csv_rows.append({
            "hole_path": sp,
            "true_layer": pen_l,
            "pred_layer": pred_layer,
            "error": error,
            "raw_decision_idx": round(raw_idx, 3),
            "early_stop_layer": es_count,
            "total_layers": tot_layers,
        })

    metrics = compute_metrics(
        [r["pred_layer"] for r in csv_rows],
        all_labels,
        all_pen_layers,
        all_layer_lists,
        lock_layers=lock_layers,
    )
    metrics["decision_method"] = decision_method + "_streaming"
    metrics["early_stop_rate"] = sum(1 for e in early_stop_counts if e >= 0) / max(len(early_stop_counts), 1)
    metrics["avg_layers_processed"] = sum(t for t in total_layers_processed) / max(len(total_layers_processed), 1)
    metrics["avg_early_stop_layers"] = sum(e for e in early_stop_counts if e >= 0) / max(sum(1 for e in early_stop_counts if e >= 0), 1)

    return csv_rows, metrics, all_frame_probs if return_frame_probs else None


def print_metrics(metrics: dict, prefix: str = "[Inference]"):
    n = metrics.get("total", 0)
    dm = metrics.get("decision_method", "?")
    print(f"{prefix} Decision={dm}  n={n}")
    if n > 0:
        print(f"  pct_within_3: {metrics['pct_within_3']*100:.1f}%")
        print(f"  pct_within_5: {metrics['pct_within_5']*100:.1f}%")
        print(f"  pct_over_10:  {metrics['pct_over_10']*100:.1f}%")
        if dm == "s3wd":
            p = metrics.get("s3wd_params", {})
            print(f"  (S3WD params: wait={p.get('wait','?')}, "
                  f"thresh={p.get('threshold','?')}, accept={p.get('accept','?')})")
        elif dm == "learned":
            if "decision_head_params" in metrics:
                print(f"  (LearnedDecisionHead: no threshold tuning needed)")
    else:
        print("  (no valid samples)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_dataset(
    args: argparse.Namespace,
    samples_info_path: Optional[str],
    mode: str = "cache",
    device: Optional[torch.device] = None,
) -> tuple[Optional[Dataset], DataLoader]:
    """
    Build dataset and DataLoader for the given mode.

    mode:
      "cache"  — ROICacheDataset (pre-cropped .pt files)
      "online" — MaskedDrillingDataset (online ROI extraction)
    """
    collate_fn = collate_masked_batch
    roi_size = args.dinov3_roi_size
    max_frames = args.max_frames_per_layer
    max_layers = args.max_layers
    max_samples = args.max_samples
    preload = args.preload
    num_workers = args.num_workers

    if mode == "cache":
        dataset = ROICacheDataset(
            cache_dir=args.roi_cache_dir,
            samples_info_path=samples_info_path,
            roi_size=roi_size,
            max_layers=max_layers,
            max_frames_per_layer=max_frames,
            preload=preload,
            max_samples=max_samples,
        )
    else:
        dataset = MaskedDrillingDataset(
            samples_info_path=samples_info_path,
            roi_size=roi_size,
            max_layers=max_layers,
            max_frames_per_layer=max_frames,
            preload=preload,
            max_samples=max_samples,
        )

    # Avoid multiprocessing issues on some envs; only use workers with CUDA
    use_cuda = device is not None and device.type == "cuda"
    loader_workers = num_workers if use_cuda else 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=loader_workers,
        collate_fn=collate_fn,
        pin_memory=use_cuda,
        prefetch_factor=2 if loader_workers > 0 else None,
    )
    return dataset, loader


def main():
    parser = argparse.ArgumentParser(
        description="Inference for masked_v2 Stage 2 model. "
                    "Supports cached ROI (default) and online ROI extraction.",
    )

    # --- Model ---
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to Stage 2 checkpoint (.pt)")
    parser.add_argument("--dinov3_model", type=str, default="vit_small")
    parser.add_argument("--dinov3_feat_dim", type=int, default=384)
    parser.add_argument("--dinov3_roi_size", type=int, default=224)
    parser.add_argument("--dinov3_chunk_size", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--nhead", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--freeze_encoder",
                        type=lambda x: x.lower() == "true", default=True)
    parser.add_argument("--mask_ratio", type=float, default=0.75)
    parser.add_argument("--mask_shape", type=str, default="circle")
    parser.add_argument("--precomputed_dir", type=str, default=None)
    parser.add_argument("--use_cached_features", action="store_true")

    # --- Data ---
    parser.add_argument("--roi_cache_dir", type=str,
                        default="data_drilling/roi_cache",
                        help="Directory with pre-cropped .pt ROI cache files")
    parser.add_argument("--samples_info", type=str,
                        default="data_drilling/samples_info_test.json",
                        help="Sample metadata for test set (JSON array)")
    parser.add_argument("--val_samples_info", type=str,
                        default="data_drilling/samples_info_val.json",
                        help="Sample metadata for validation set (used for S3WD grid search)")
    parser.add_argument("--online_crop", action="store_true",
                        help="Use online ROI extraction (MaskedDrillingDataset) instead of cached .pt files")
    parser.add_argument("--max_layers", type=int, default=None)
    parser.add_argument("--max_frames_per_layer", type=int, default=8)
    parser.add_argument("--preload", action="store_true",
                        help="Preload .pt files into RAM (cached mode) or scan dirs (online mode)")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4)

    # --- Inference ---
    parser.add_argument("--run_val", action="store_true",
                        help="Run validation set first for S3WD grid search, then test set")
    parser.add_argument("--decision_method", type=str, default="s3wd",
                        choices=["s3wd", "topkmedian", "threshold", "learned"])
    parser.add_argument("--lock_layers", type=int, default=30,
                        help="Layers before this index are forced to non-penetrated. Default 30.")
    parser.add_argument("--s3wd_wait", type=int, default=5)
    parser.add_argument("--s3wd_thresh", type=float, default=0.6,
                        help="S3WD probability threshold (used when --run_val is NOT set)")
    parser.add_argument("--s3wd_accept", type=float, default=0.3,
                        help="S3WD accept threshold — min probability to confirm a decision "
                             "(used when --run_val is NOT set)")
    parser.add_argument("--skip_grid_search", action="store_true",
                        help="Skip S3WD grid search on validation set (use default s3wd_wait/s3wd_thresh/s3wd_accept)")

    # --- Streaming inference (early stopping) ---
    parser.add_argument("--streaming", action="store_true",
                        help="Enable streaming inference: process layers one-by-one with early stopping. "
                             "As soon as penetration is detected, stop reading further layers.")
    parser.add_argument("--stop_thresh", type=float, default=0.5,
                        help="Penetration probability threshold for streaming early stop. Default 0.5.")
    parser.add_argument("--stop_wait", type=int, default=3,
                        help="Consecutive steps above stop_thresh to confirm early stop. Default 3.")
    parser.add_argument("--stop_accept", type=float, default=0.3,
                        help="Minimum best probability to confirm early stop. Default 0.3.")
    parser.add_argument("--max_inference_layers", type=int, default=None,
                        help="Cap on layers processed per well in streaming mode (for non-penetrated). Default: all layers.")

    # --- Output ---
    parser.add_argument("--output_csv", type=str, default="inference_results.csv",
                        help="Output CSV path with per-well results")
    parser.add_argument("--grid_search_output", type=str, default="grid_search_results.json",
                        help="Output JSON for grid search results (when --run_val is used)")
    parser.add_argument("--save_predictions_json", type=str, default=None,
                        help="Optional: save raw predictions as JSON")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")

    # --- Load model ---
    model_kwargs = dict(
        dinov3_model=args.dinov3_model,
        dinov3_feat_dim=args.dinov3_feat_dim,
        dinov3_roi_size=args.dinov3_roi_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_transformer_layers=args.num_layers,
        freeze_encoder=args.freeze_encoder,
        mask_ratio=args.mask_ratio,
        stage=2,
    )
    if args.use_cached_features and args.precomputed_dir:
        model_kwargs["use_cached_features"] = True

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = MaskedPixelModel(**model_kwargs).to(device)

    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"], strict=False)
    print(f"[Main] Loaded checkpoint from {args.checkpoint}")

    data_mode = "online" if args.online_crop else "cache"

    # -------------------------------------------------------------------------
    # Phase 1: Validation set — S3WD grid search
    # -------------------------------------------------------------------------
    best_s3wd_params: Optional[dict] = None
    val_metrics: Optional[dict] = None

    if args.run_val and os.path.exists(args.val_samples_info):
        print(f"\n{'='*60}")
        print(f"[Phase 1] Validation inference + S3WD grid search")
        print(f"{'='*60}")

        val_dataset, val_loader = build_dataset(args, args.val_samples_info, mode=data_mode, device=device)

        if args.skip_grid_search:
            best_s3wd_params = {"wait": args.s3wd_wait, "threshold": args.s3wd_thresh, "accept": args.s3wd_accept}
            print(f"[Phase 1] skip_grid_search=True — using provided params: "
                  f"wait={best_s3wd_params['wait']}, thresh={best_s3wd_params['threshold']}, "
                  f"accept={best_s3wd_params['accept']}")
            # Still run inference to get metrics
            val_rows, val_metrics = run_inference(
                model, val_loader, val_dataset, device,
                decision_method=args.decision_method,
                s3wd_wait=best_s3wd_params["wait"],
                s3wd_thresh=best_s3wd_params["threshold"],
                s3wd_accept=best_s3wd_params["accept"],
                lock_layers=args.lock_layers,
                best_s3wd_params=None,
                use_learned_decision=args.decision_method == "learned",
            )
        else:
            best_s3wd_params, all_grid_results = grid_search_s3wd(
                model, val_loader, device,
                lock_layers=args.lock_layers,
            )
            # Save grid search results
            grid_output = {
                "best_params": best_s3wd_params,
                "all_results": all_grid_results,
            }
            with open(args.grid_search_output, "w") as f:
                json.dump(grid_output, f, indent=2)
            print(f"[Phase 1] Grid search results saved to {args.grid_search_output}")

            # Run final val inference with best params
            val_rows, val_metrics = run_inference(
                model, val_loader, val_dataset, device,
                decision_method="s3wd",
                lock_layers=args.lock_layers,
                best_s3wd_params=best_s3wd_params,
                use_learned_decision=args.decision_method == "learned",
            )

        val_metrics["decision_method"] = args.decision_method
        print_metrics(val_metrics, prefix="[Phase 1 Val]")

        # Save val CSV (optional)
        val_csv_path = args.output_csv.replace(".csv", "_val.csv")
        with open(val_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f, fieldnames=["hole_path", "true_layer", "pred_layer", "error", "raw_decision_idx"])
            writer.writeheader()
            writer.writerows(val_rows)
        print(f"[Phase 1] Val CSV saved to {val_csv_path}")
    else:
        # No validation — use default or CLI params; try checkpoint s3wd_params first
            if args.decision_method == "s3wd":
                ckpt_s3wd = checkpoint.get("s3wd_params")
                if ckpt_s3wd:
                    best_s3wd_params = ckpt_s3wd
                    print(f"[Main] Loaded s3wd params from checkpoint: "
                          f"wait={best_s3wd_params['wait']}, thresh={best_s3wd_params['threshold']}, "
                          f"accept={best_s3wd_params.get('accept','?')}")
                else:
                    best_s3wd_params = {
                        "wait": args.s3wd_wait,
                        "threshold": args.s3wd_thresh,
                        "accept": args.s3wd_accept,
                    }
                    print(f"[Main] No checkpoint s3wd params — using CLI: "
                          f"wait={best_s3wd_params['wait']}, thresh={best_s3wd_params['threshold']}, "
                          f"accept={best_s3wd_params['accept']}")

    # -------------------------------------------------------------------------
    # Phase 2: Test set inference
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"[Phase 2] Test set inference")
    print(f"{'='*60}")

    if not os.path.exists(args.samples_info):
        print(f"[Phase 2] ERROR: samples_info not found at {args.samples_info}")
        sys.exit(1)

    test_dataset, test_loader = build_dataset(args, args.samples_info, mode=data_mode, device=device)

    if test_dataset is not None and len(test_dataset) == 0:
        print(f"[Phase 2] ERROR: test dataset is empty. Check --samples_info and --roi_cache_dir")
        sys.exit(1)

    print(f"[Phase 2] Test dataset size: {len(test_dataset)}")

    # Decide effective params
    if best_s3wd_params is not None:
        eff_method = "s3wd"
        eff_s3wd_params = best_s3wd_params
    else:
        eff_method = args.decision_method
        eff_s3wd_params = None

    if args.streaming:
        print(f"[Phase 2] STREAMING mode — early stopping enabled")
        print(f"  stop_thresh={args.stop_thresh}, stop_wait={args.stop_wait}, "
              f"stop_accept={args.stop_accept}, max_layers={args.max_inference_layers}")
        test_rows, test_metrics, _ = run_streaming_inference(
            model, test_dataset, device,
            lock_layers=args.lock_layers,
            decision_method=eff_method if eff_method in ("learned", "threshold") else "learned",
            stop_thresh=args.stop_thresh,
            stop_wait=args.stop_wait,
            max_inference_layers=args.max_inference_layers,
            return_frame_probs=False,
        )
        csv_fieldnames = ["hole_path", "true_layer", "pred_layer", "error",
                          "raw_decision_idx", "early_stop_layer", "total_layers"]
    else:
        test_rows, test_metrics = run_inference(
            model, test_loader, test_dataset, device,
            decision_method=eff_method,
            s3wd_wait=args.s3wd_wait,
            s3wd_thresh=args.s3wd_thresh,
            s3wd_accept=args.s3wd_accept,
            lock_layers=args.lock_layers,
            best_s3wd_params=eff_s3wd_params,
            use_learned_decision=args.decision_method == "learned",
        )
        csv_fieldnames = ["hole_path", "true_layer", "pred_layer", "error", "raw_decision_idx"]
    test_metrics["decision_method"] = eff_method + ("_streaming" if args.streaming else "")

    print_metrics(test_metrics, prefix="[Phase 2 Test]")

    # Streaming-specific stats
    if args.streaming:
        print(f"  early_stop_rate: {test_metrics.get('early_stop_rate', 0)*100:.1f}%")
        print(f"  avg_layers_processed: {test_metrics.get('avg_layers_processed', 0):.1f}")
        print(f"  avg_early_stop_layers: {test_metrics.get('avg_early_stop_layers', 0):.1f}")

    # Save test CSV
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fieldnames)
        writer.writeheader()
        writer.writerows(test_rows)
    print(f"\n[Phase 2] Test CSV saved to {args.output_csv}")

    # Optional: save predictions as JSON
    if args.save_predictions_json:
        with open(args.save_predictions_json, "w") as f:
            json.dump(test_rows, f, indent=2, ensure_ascii=False)
        print(f"[Phase 2] Raw predictions JSON saved to {args.save_predictions_json}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
