#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 2 推理脚本 — 完全复刻 train.py 的 evaluate_stage2() 逻辑。

两种推理模式：
  --mode cached  : 使用预裁剪的 ROI 缓存 + 预计算的 DINOv3 特征（最快）
  --mode online  : 在线裁剪 ROI + 在线用 DINOv3 提特征（最灵活）
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import List

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from grid_diff_tcn.masked_v2.dataset import MaskedDrillingDataset, collate_masked_batch
from grid_diff_tcn.masked_v2.extract import build_feature_extractor, _extract_single
from grid_diff_tcn.masked_v2.model import MaskedPixelModel, load_masked_model


# ---------------------------------------------------------------------------
# 核心评估函数 — 从 train.py 原样复制
# ---------------------------------------------------------------------------

def s3wd_decision(
    prob_t: torch.Tensor,
    mask_t: torch.Tensor,
    layers: list[int],
    wait: int = 3,
    threshold: float = 0.6,
    accept: float = 1.0,
) -> tuple[int, str]:
    """
    S3WD decision for a single sample.

    Returns:
        (pred_layer, source):
            source == "s3wd"           — S3WD scan found a decision
            source == "argmax_fallback" — S3WD scan completed without finding
                                          a decision; used argmax over valid positions
            source == "invalid"         — no valid positions (mask all False or sum==0)
    """
    t = len(prob_t)
    consecutive_high = 0
    for ti in range(t):
        if not mask_t[ti]:
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
    # S3WD scan completed without a decision → fallback to argmax
    valid = mask_t
    if valid.sum() == 0:
        return (-1, "invalid")
    valid_prob = prob_t.clone()
    valid_prob[~valid] = float("-inf")
    argmax_ti = valid_prob.argmax().item()
    return (layers[argmax_ti] if argmax_ti < len(layers) else layers[-1], "argmax_fallback")


def _s3wd_fixed_eval(
    all_frame_probs: list,
    all_labels: list,
    all_pen_layers: list,
    all_layer_lists: list,
    lock_layers: int,
    wait: int = 3,
    threshold: float = 0.6,
    accept: float = 1.0,
) -> tuple[float, dict, list, list]:
    """
    Fixed-parameter S3WD evaluation.

    Returns:
        combined_metric: pct_within_3 + pct_within_5 on S3WD preds
        params_dict: S3WD hyperparams and per-class percentages
        s3wd_preds: S3WD predictions (same length as inputs)
        sources: list of "s3wd" or "argmax" per sample, indicating which
                 method contributed the final prediction
    """
    s3wd_preds = []
    sources = []
    for ((prob_t, mask_t), label, pen_layer, layers) in zip(
            all_frame_probs, all_labels, all_pen_layers, all_layer_lists):
        if label != 1:
            s3wd_preds.append(-1)
            sources.append("argmax")  # excluded anyway
            continue
        if pen_layer < 0 or pen_layer not in layers:
            s3wd_preds.append(-1)
            sources.append("argmax")  # excluded anyway
            continue
        pen_idx = layers.index(pen_layer)
        if pen_idx < lock_layers:
            s3wd_preds.append(-1)
            sources.append("argmax")  # excluded anyway
            continue
        pred_layer, src = s3wd_decision(prob_t, mask_t, layers, wait=wait, threshold=threshold, accept=accept)
        s3wd_preds.append(pred_layer)
        sources.append(src)
    m = compute_metrics(s3wd_preds, all_labels, all_pen_layers, all_layer_lists, lock_layers=lock_layers)
    combined = m["pct_within_3"] + m["pct_within_5"]
    return combined, {
        "wait": wait, "threshold": threshold, "accept": accept,
        "pct_within_3": m["pct_within_3"],
        "pct_within_5": m["pct_within_5"],
        "pct_over_10": m["pct_over_10"],
        "total": m["total"],
    }, s3wd_preds, sources


def compute_metrics(
    preds: list,
    labels: list,
    penetration_layers: list,
    layer_lists: list,
    lock_layers: int = 30,
) -> dict:
    """Compute evaluation metrics. Mirrors train.py compute_metrics."""
    within_3 = within_5 = over_10 = total = 0
    for pred, label, pen_layer, layers in zip(preds, labels, penetration_layers, layer_lists):
        if label != 1:
            continue
        if pen_layer < 0 or pen_layer not in layers:
            continue
        pen_idx = layers.index(pen_layer)
        if pen_idx < lock_layers:
            continue
        if pred < 0 or pred not in layers:
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


# ---------------------------------------------------------------------------
# 推理核心 — 逐字复刻 train.py evaluate_stage2()
# ---------------------------------------------------------------------------

def run_inference(
    model: MaskedPixelModel,
    loader: DataLoader,
    device: torch.device,
    decision_method: str = "s3wd",
    s3wd_wait: int = 3,
    s3wd_threshold: float = 0.6,
    s3wd_accept: float = 1.0,
    lock_layers: int = 30,
    use_learned: bool = False,
    debug: bool = False,
) -> tuple[List[dict], dict]:
    """
    Batch inference — 完全复刻 train.py evaluate_stage2()。

    decision_method:
      - "s3wd" (default): S3WD post-processing on locked softmax probability
        (sequential scan for N consecutive frames above threshold; argmax is fallback)
      - "learned": TemporalDecisionHead — first decision_prob > 0.5 determines prediction

    Returns:
        csv_rows: [{hole_path, true_layer, pred_layer, error, pen_layer, layer_list_str}]
        metrics: dict with aggregate metrics
    """
    model.eval()
    use_learned = (decision_method == "learned")

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_pen_layers: List[int] = []
    all_layer_lists: List[List[int]] = []
    all_sample_paths: List[str] = []
    all_frame_probs: List[tuple] = []   # only used by s3wd

    with torch.inference_mode():
        for batch in tqdm(loader, desc="[Inference] Running model"):
            frame_data = batch["frame_data"].to(device)
            frame_mask = batch["frame_mask"].to(device)
            labels = batch["labels"]
            pen_layers = batch["penetration_layers"]
            layer_lists_batch = batch["layer_lists"]
            sample_paths_batch = batch["sample_paths"]

            # ---- model forward — 复刻 train.py evaluate_stage2() ----
            result = model.forward(
                frame_data, frame_mask=frame_mask,
                return_decision_idx=use_learned,
            )
            logits = result["logits"]  # (B, 2, T)

            # ---- lock_layers — 复刻 train.py 第978-979行 ----
            if lock_layers > 0:
                logits[:, 1, :lock_layers] = float("-inf")

            # ---- Prediction — 复刻 train.py evaluate_stage2() 分支 ----
            if use_learned:
                decision_idx = result.get("decision_idx")
                for bi in range(logits.shape[0]):
                    if decision_idx is not None and labels[bi].item() == 1:
                        ll = layer_lists_batch[bi]
                        raw = int(round(decision_idx[bi].item()))
                        idx = max(0, min(raw, len(ll) - 1))
                        all_preds.append(ll[idx])
                    else:
                        all_preds.append(-1)
            else:
                # ---- s3wd path — 复刻 train.py 第1010-1022行 ----
                probs = F.softmax(logits, dim=1)[:, 1]  # (B, T) — 和 train.py 完全一致
                for bi, ll in enumerate(layer_lists_batch):
                    pt = probs[bi]
                    mask_t = frame_mask[bi].any(dim=1)
                    all_frame_probs.append((pt.cpu(), mask_t.cpu()))
                    if mask_t.sum() == 0:
                        all_preds.append(-1)
                        continue
                    valid = pt[mask_t]
                    valid_idx = torch.where(mask_t)[0]
                    argmax_ti = valid_idx[valid.argmax()].item()
                    all_preds.append(ll[argmax_ti] if argmax_ti < len(ll) else ll[-1])

            # ---- 收集所有样本信息 ----
            all_labels.extend(labels.cpu().tolist())
            all_pen_layers.extend(pen_layers.cpu().tolist())
            all_layer_lists.extend(layer_lists_batch)
            all_sample_paths.extend(sample_paths_batch)

    # ---- Metrics — argmax baseline (overwritten by S3WD post-processing below) ----
    # Mirrors train.py: compute_metrics(all_preds) first, then S3WD, then compute_metrics again
    metrics = compute_metrics(
        all_preds, all_labels, all_pen_layers, all_layer_lists,
        lock_layers=lock_layers,
    )

    # ---- S3WD post-processing — 复刻 train.py evaluate_stage2() 的流程 ----
    # 1. _s3wd_fixed_eval builds its own preds list from all_frame_probs
    # 2. Then re-call compute_metrics on that list (mirrors train.py lines 1029 + 1051)
    # 3. CSV uses the same S3WD-modified preds
    if use_learned:
        metrics["learned_decision"] = True
        metrics["best_s3wd_params"] = {
            "mode": "learned_decision",
            "pct_within_3": metrics["pct_within_3"],
            "pct_within_5": metrics["pct_within_5"],
            "pct_over_10": metrics["pct_over_10"],
            "total": metrics["total"],
        }
    else:
        # _s3wd_fixed_eval returns (combined_metric, params_dict, s3wd_preds, sources)
        best_metric, best_params, s3wd_preds, sources = _s3wd_fixed_eval(
            all_frame_probs, all_labels, all_pen_layers, all_layer_lists, lock_layers,
            wait=s3wd_wait, threshold=s3wd_threshold, accept=s3wd_accept,
        )
        # Re-compute metrics on S3WD preds — matches train.py's final compute_metrics call
        metrics = compute_metrics(
            s3wd_preds, all_labels, all_pen_layers, all_layer_lists,
            lock_layers=lock_layers,
        )
        metrics["best_s3wd_metric"] = best_metric
        metrics["best_s3wd_params"] = best_params
        metrics["learned_decision"] = False
        # CSV uses S3WD preds with source annotation
        csv_preds = s3wd_preds
        csv_rows = []
        for sp, true_layer, pred_layer, layers, src in zip(
                all_sample_paths, all_pen_layers, csv_preds, all_layer_lists, sources):
            if true_layer >= 0 and pred_layer >= 0 and pred_layer in layers:
                error = abs(layers.index(pred_layer) - layers.index(true_layer))
            else:
                error = -1
            csv_rows.append({
                "hole_path": sp,
                "true_layer": true_layer,
                "pred_layer": pred_layer,
                "error": error,
                "decision_source": src,
            })
        metrics["csv_rows"] = csv_rows
        return csv_rows, metrics

    # ---- Build CSV rows (learned mode) ----
    csv_rows = []
    for sp, true_layer, pred_layer, layers in zip(
            all_sample_paths, all_pen_layers, all_preds, all_layer_lists):
        if true_layer >= 0 and pred_layer >= 0 and pred_layer in layers:
            error = abs(layers.index(pred_layer) - layers.index(true_layer))
        else:
            error = -1
        csv_rows.append({
            "hole_path": sp,
            "true_layer": true_layer,
            "pred_layer": pred_layer,
            "error": error,
            "decision_source": "learned",
        })

    metrics["csv_rows"] = csv_rows
    return csv_rows, metrics


def print_metrics(metrics: dict, prefix: str = "[Inference]") -> None:
    n = metrics.get("total", 0)
    dm = "learned" if metrics.get("learned_decision") else "s3wd"
    print(f"{prefix} Decision={dm}  n={n}")
    if n > 0:
        print(f"  pct_within_3: {metrics['pct_within_3']*100:.1f}%")
        print(f"  pct_within_5: {metrics['pct_within_5']*100:.1f}%")
        print(f"  pct_over_10:  {metrics['pct_over_10']*100:.1f}%")
        p = metrics.get("best_s3wd_params", {})
        if dm == "s3wd":
            print(f"  (S3WD params: wait={p.get('wait','?')}, "
                  f"thresh={p.get('threshold','?')}, accept={p.get('accept','?')})")
    else:
        print("  (no valid samples — check lock_layers or label filtering)")


# ---------------------------------------------------------------------------
# 在线特征提取模式
# ---------------------------------------------------------------------------

def run_inference_online(
    samples_info: str,
    device: torch.device,
    model_checkpoint: str,
    encoder_checkpoint: str | None,
    dinov3_model: str,
    dinov3_feat_dim: int,
    dinov3_roi_size: int,
    dinov3_chunk_size: int,
    decision_method: str,
    s3wd_wait: int,
    s3wd_threshold: float,
    s3wd_accept: float,
    lock_layers: int,
    max_layers: int | None,
    max_frames_per_layer: int,
    batch_size: int,
    num_workers: int,
    max_samples: int | None,
    output_csv: str,
    roi_size: int = 224,
) -> dict:
    """
    在线推理：在线裁剪 ROI + 在线用 DINOv3 提特征。
    复刻 extract.py + evaluate_stage2() 的流程。
    """
    print(f"[Online Mode] Building MaskedDrillingDataset...")
    dataset = MaskedDrillingDataset(
        samples_info_path=samples_info,
        roi_size=roi_size,
        max_layers=max_layers,
        max_frames_per_layer=max_frames_per_layer,
        preload=False,
        max_samples=max_samples,
    )
    print(f"[Online Mode] Dataset size: {len(dataset)}")

    # Build feature extractor from encoder checkpoint (stage 1)
    enc_ckpt = encoder_checkpoint or model_checkpoint
    print(f"[Online Mode] Loading feature extractor from {enc_ckpt}...")
    extractor, is_model = build_feature_extractor(
        dinov3_model=dinov3_model,
        dinov3_roi_size=dinov3_roi_size,
        checkpoint_path=enc_ckpt,
        dinov3_feat_dim=dinov3_feat_dim,
    )
    extractor = extractor.to(device)
    extractor.eval()

    # Load model: encoder from stage1, classifier from stage2
    enc_ckpt = encoder_checkpoint or model_checkpoint
    model = load_masked_model(
        model_checkpoint, stage=2,
        encoder_checkpoint=enc_ckpt,
    ).to(device)
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    all_pen_layers: List[int] = []
    all_layer_lists: List[List[int]] = []
    all_sample_paths: List[str] = []
    all_frame_probs: List[tuple] = []
    use_learned = (decision_method == "learned")

    with torch.inference_mode():
        for idx in tqdm(range(len(dataset)), desc="[Online] Extracting features"):
            sample = dataset[idx]

            frame_data = sample["frame_data"]          # (T, F, 3, H, W)
            frame_mask = sample["frame_mask"]           # (T, F) bool
            label = int(sample["label"])
            pen_layer = int(sample["penetration_layer"])
            layer_list: List[int] = sample["layer_list"]
            sample_path = str(sample.get("sample_path", ""))

            T, F, C, H, W = frame_data.shape
            frame_data_batched = frame_data.unsqueeze(0)  # (1, T, F, 3, H, W)
            frame_mask_batched = frame_mask.unsqueeze(0)   # (1, T, F)

            # ---- DINOv3 feature extraction — 复刻 extract.py _extract_single() ----
            feats = _extract_single(
                extractor, is_model,
                frame_data_batched,
                dinov3_roi_size,
                dinov3_chunk_size,
            )  # (1, T, F, feat_dim)
            feats = feats.squeeze(0)  # (T, F, feat_dim)

            # ---- Model forward — 复刻 train.py evaluate_stage2() ----
            feats_b = feats.unsqueeze(0).to(device)  # (1, T, F, C)
            mask_b = frame_mask_batched.to(device)

            result = model.forward(feats_b, frame_mask=mask_b, return_decision_idx=use_learned)
            logits = result["logits"]  # (1, 2, T)

            if lock_layers > 0:
                logits[:, 1, :lock_layers] = float("-inf")

            if use_learned:
                decision_idx = result.get("decision_idx")
                if decision_idx is not None and label == 1:
                    raw = int(round(decision_idx[0].item()))
                    idx_i = max(0, min(raw, len(layer_list) - 1))
                    pred_layer = layer_list[idx_i]
                else:
                    pred_layer = -1
            else:
                probs = F.softmax(logits, dim=1)[0, 1]  # (T,)
                mask_t = frame_mask.any(dim=1)
                all_frame_probs.append((probs.cpu(), mask_t.cpu()))
                if mask_t.sum() == 0:
                    pred_layer = -1
                else:
                    valid = probs[mask_t]
                    valid_idx = torch.where(mask_t)[0]
                    argmax_ti = valid_idx[valid.argmax()].item()
                    pred_layer = layer_list[argmax_ti] if argmax_ti < len(layer_list) else layer_list[-1]

            all_preds.append(pred_layer)
            all_labels.append(label)
            all_pen_layers.append(pen_layer)
            all_layer_lists.append(layer_list)
            all_sample_paths.append(sample_path)

    # Re-compute metrics on S3WD-modified preds — matches train.py's final compute_metrics call
    if use_learned:
        metrics["learned_decision"] = True
        metrics["best_s3wd_params"] = {
            "mode": "learned_decision",
            "pct_within_3": metrics["pct_within_3"],
            "pct_within_5": metrics["pct_within_5"],
            "pct_over_10": metrics["pct_over_10"],
            "total": metrics["total"],
        }
    else:
        best_metric, best_params, s3wd_preds, sources = _s3wd_fixed_eval(
            all_frame_probs, all_labels, all_pen_layers, all_layer_lists, lock_layers,
            wait=s3wd_wait, threshold=s3wd_threshold, accept=s3wd_accept,
        )
        # Re-compute metrics on S3WD preds — matches train.py's final compute_metrics call
        metrics = compute_metrics(
            s3wd_preds, all_labels, all_pen_layers, all_layer_lists,
            lock_layers=lock_layers,
        )
        metrics["best_s3wd_metric"] = best_metric
        metrics["best_s3wd_params"] = best_params
        metrics["learned_decision"] = False
        csv_preds = s3wd_preds
        csv_rows = []
        for sp, true_layer, pred_layer, layers, src in zip(
                all_sample_paths, all_pen_layers, csv_preds, all_layer_lists, sources):
            if true_layer >= 0 and pred_layer >= 0 and pred_layer in layers:
                error = abs(layers.index(pred_layer) - layers.index(true_layer))
            else:
                error = -1
            csv_rows.append({
                "hole_path": sp,
                "true_layer": true_layer,
                "pred_layer": pred_layer,
                "error": error,
                "decision_source": src,
            })
        with open(output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["hole_path", "true_layer", "pred_layer", "error", "decision_source"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[Online Mode] CSV saved to {output_csv}")
        print_metrics(metrics, prefix="[Online Inference]")
        return metrics

    # ---- Build CSV rows (learned mode) ----
    csv_rows = []
    for sp, true_layer, pred_layer, layers in zip(
            all_sample_paths, all_pen_layers, all_preds, all_layer_lists):
        if true_layer >= 0 and pred_layer >= 0 and pred_layer in layers:
            error = abs(layers.index(pred_layer) - layers.index(true_layer))
        else:
            error = -1
        csv_rows.append({
            "hole_path": sp,
            "true_layer": true_layer,
            "pred_layer": pred_layer,
            "error": error,
            "decision_source": "learned",
        })

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hole_path", "true_layer", "pred_layer", "error", "decision_source"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[Online Mode] CSV saved to {output_csv}")
    print_metrics(metrics, prefix="[Online Inference]")
    return metrics


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Stage 2 推理 — 复刻 train.py evaluate_stage2()",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ---- 模式选择 ----
    parser.add_argument(
        "--mode", type=str, default="cached", choices=["cached", "online"],
        help="cached: 预裁剪ROI + 预计算特征（最快）; "
             "online: 在线ROI裁剪 + 在线DINOv3特征提取（最灵活）"
    )

    # ---- 数据 ----
    parser.add_argument("--samples_info", type=str, required=True,
                        help="样本信息 JSON 文件路径")
    parser.add_argument("--max_layers", type=int, default=None,
                        help="最多处理多少层（None=全部）")
    parser.add_argument("--max_frames_per_layer", type=int, default=8,
                        help="每层最多帧数")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最多处理多少样本（None=全部）")
    parser.add_argument("--roi_size", type=int, default=224,
                        help="ROI 图像尺寸")

    # ---- 预计算特征（cached 模式） ----
    parser.add_argument("--crop_cache_dir", type=str, default=None,
                        help="CropCacheDataset 的缓存目录")
    parser.add_argument("--precomputed_dir", type=str, default=None,
                        help="预计算的 DINOv3 特征目录（.pt 文件）")

    # ---- DINOv3 ----
    parser.add_argument("--dinov3_model", type=str, default="vit_small")
    parser.add_argument("--dinov3_feat_dim", type=int, default=384)
    parser.add_argument("--dinov3_roi_size", type=int, default=224)
    parser.add_argument("--dinov3_chunk_size", type=int, default=256,
                        help="DINOv3 每次处理多少帧（显存不够调小）")

    # ---- 模型 ----
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Stage 2 模型 checkpoint 路径（classifier 权重）")
    parser.add_argument("--encoder_checkpoint", type=str, default=None,
                        help="Stage 1 checkpoint 路径（encoder 权重，用于 online 特征提取）")

    # ---- 推理参数（复刻 train.py evaluate_stage2） ----
    parser.add_argument("--decision_method", type=str, default="s3wd",
                        choices=["s3wd", "learned"],
                        help="s3wd: argmax+后处理; learned: TemporalDecisionHead")
    parser.add_argument("--lock_layers", type=int, default=30,
                        help="前 N 层强制为 0（安全锁，和 train.py 一致）")
    parser.add_argument("--s3wd_wait", type=int, default=3,
                        help="S3WD: 连续多少帧超过阈值才确认")
    parser.add_argument("--s3wd_threshold", type=float, default=0.6,
                        help="S3WD: 概率阈值")
    parser.add_argument("--s3wd_accept", type=float, default=1.0,
                        help="S3WD: 概率 >= 此值立即决策（1.0=禁用）")

    # ---- DataLoader ----
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=4)

    # ---- 输出 ----
    parser.add_argument("--output_csv", type=str, default="inference_results.csv")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Main] Device: {device}")
    print(f"[Main] Mode: {args.mode}")
    print(f"[Main] Decision: {args.decision_method}")
    print(f"[Main] Lock layers: {args.lock_layers}")
    if args.decision_method == "s3wd":
        print(f"[Main] S3WD: wait={args.s3wd_wait}, thresh={args.s3wd_threshold}, accept={args.s3wd_accept}")

    if args.mode == "online":
        # ---- 在线模式：在线裁剪 + 在线特征提取 ----
        metrics = run_inference_online(
            samples_info=args.samples_info,
            device=device,
            model_checkpoint=args.checkpoint,
            encoder_checkpoint=args.encoder_checkpoint,
            dinov3_model=args.dinov3_model,
            dinov3_feat_dim=args.dinov3_feat_dim,
            dinov3_roi_size=args.dinov3_roi_size,
            dinov3_chunk_size=args.dinov3_chunk_size,
            decision_method=args.decision_method,
            s3wd_wait=args.s3wd_wait,
            s3wd_threshold=args.s3wd_threshold,
            s3wd_accept=args.s3wd_accept,
            lock_layers=args.lock_layers,
            max_layers=args.max_layers,
            max_frames_per_layer=args.max_frames_per_layer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            output_csv=args.output_csv,
            roi_size=args.roi_size,
        )
        print("\n[DONE]")
        return

    # ---- Cached 模式 ----
    if args.precomputed_dir and os.path.exists(args.precomputed_dir):
        # 模式 A：预裁剪ROI + 预计算DINOv3特征（最快）
        print(f"[Cached Mode] Using precomputed features from: {args.precomputed_dir}")
        from grid_diff_tcn.masked_v2.dataset import CropCacheDataset
        dataset = CropCacheDataset(
            samples_info_path=args.samples_info,
            cache_dir=args.crop_cache_dir or "",
            roi_size=args.roi_size,
            max_layers=args.max_layers,
            max_frames_per_layer=args.max_frames_per_layer,
            preload=False,
            max_samples=args.max_samples,
            precomputed_dir=args.precomputed_dir,
        )
        use_online_feat = False

    elif args.crop_cache_dir and os.path.exists(args.crop_cache_dir):
        # 模式 B：预裁剪ROI + 在线DINOv3特征提取
        print(f"[Cached Mode] Using ROI cache: {args.crop_cache_dir}")
        print(f"[Cached Mode] Feature extraction: ONLINE (per-sample)")
        from grid_diff_tcn.masked_v2.dataset import CropCacheDataset
        dataset = CropCacheDataset(
            samples_info_path=args.samples_info,
            cache_dir=args.crop_cache_dir,
            roi_size=args.roi_size,
            max_layers=args.max_layers,
            max_frames_per_layer=args.max_frames_per_layer,
            preload=False,
            max_samples=args.max_samples,
            precomputed_dir=None,
        )
        use_online_feat = True

    else:
        # 模式 C：纯在线（MaskedDrillingDataset + 在线DINOv3）
        print(f"[Cached Mode] No ROI cache or precomputed features. Using online crop.")
        metrics = run_inference_online(
            samples_info=args.samples_info,
            device=device,
            model_checkpoint=args.checkpoint,
            encoder_checkpoint=args.encoder_checkpoint,
            dinov3_model=args.dinov3_model,
            dinov3_feat_dim=args.dinov3_feat_dim,
            dinov3_roi_size=args.dinov3_roi_size,
            dinov3_chunk_size=args.dinov3_chunk_size,
            decision_method=args.decision_method,
            s3wd_wait=args.s3wd_wait,
            s3wd_threshold=args.s3wd_threshold,
            s3wd_accept=args.s3wd_accept,
            lock_layers=args.lock_layers,
            max_layers=args.max_layers,
            max_frames_per_layer=args.max_frames_per_layer,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_samples=args.max_samples,
            output_csv=args.output_csv,
            roi_size=args.roi_size,
        )
        print("\n[DONE]")
        return

    # Load model
    # - use_cached_features=False: always build encoder so we can load stage1.pt weights
    #   (the forward pass will still skip encoder when input is 4D features from CropCacheDataset)
    # - use_cached_features=True: skip encoder construction (no weights to load anyway)
    # For cached mode we still need encoder weights from stage1.pt for online_feat sub-mode.
    # So always use use_cached_features=False and let forward() detect 4D vs 6D input.
    model = load_masked_model(
        args.checkpoint, stage=2,
        encoder_checkpoint=args.encoder_checkpoint,
        use_cached_features=False,
    ).to(device)
    model.eval()

    if use_online_feat:
        # ---- 模式 B：逐样本在线特征提取 ----
        # 在线特征提取需要 encoder 权重（来自 stage1.pt），
        # classifier 权重来自 stage2.pt
        enc_ckpt = args.encoder_checkpoint or args.checkpoint
        extractor, is_model = build_feature_extractor(
            dinov3_model=args.dinov3_model,
            dinov3_roi_size=args.dinov3_roi_size,
            checkpoint_path=enc_ckpt,
            dinov3_feat_dim=args.dinov3_feat_dim,
        )
        extractor = extractor.to(device)
        extractor.eval()
        use_learned = (args.decision_method == "learned")

        all_preds: List[int] = []
        all_labels: List[int] = []
        all_pen_layers: List[int] = []
        all_layer_lists: List[List[int]] = []
        all_sample_paths: List[str] = []
        all_frame_probs: List[tuple] = []

        with torch.inference_mode():
            for idx in tqdm(range(len(dataset)), desc="[Cached+Online Feat]"):
                sample = dataset[idx]
                frame_data = sample["frame_data"]     # (T, F, 3, H, W)
                frame_mask = sample["frame_mask"]     # (T, F)
                label = int(sample["label"])
                pen_layer = int(sample["penetration_layer"])
                layer_list: List[int] = sample["layer_list"]
                sample_path = str(sample.get("sample_path", ""))

                fb = frame_data.unsqueeze(0).to(device)   # (1, T, F, 3, H, W)
                mb = frame_mask.unsqueeze(0).to(device)   # (1, T, F)

                # Online DINOv3 feature extraction
                feats = _extract_single(extractor, is_model, fb, args.dinov3_roi_size, args.dinov3_chunk_size)
                feats = feats.squeeze(0)  # (T, F, feat_dim)

                # Model forward
                result = model.forward(feats.unsqueeze(0), frame_mask=mb, return_decision_idx=use_learned)
                logits = result["logits"]

                if args.lock_layers > 0:
                    logits[:, 1, :args.lock_layers] = float("-inf")

                if use_learned:
                    decision_idx = result.get("decision_idx")
                    if decision_idx is not None and label == 1:
                        raw = int(round(decision_idx[0].item()))
                        idx_i = max(0, min(raw, len(layer_list) - 1))
                        pred_layer = layer_list[idx_i]
                    else:
                        pred_layer = -1
                else:
                    probs = F.softmax(logits, dim=1)[0, 1]
                    mask_t = frame_mask.any(dim=1)
                    all_frame_probs.append((probs.cpu(), mask_t.cpu()))
                    if mask_t.sum() == 0:
                        pred_layer = -1
                    else:
                        valid = probs[mask_t]
                        valid_idx = torch.where(mask_t)[0]
                        argmax_ti = valid_idx[valid.argmax()].item()
                        pred_layer = layer_list[argmax_ti] if argmax_ti < len(layer_list) else layer_list[-1]

                all_preds.append(pred_layer)
                all_labels.append(label)
                all_pen_layers.append(pen_layer)
                all_layer_lists.append(layer_list)
                all_sample_paths.append(sample_path)

        # Metrics — argmax baseline (overwritten by S3WD post-processing below)
        metrics = compute_metrics(
            all_preds, all_labels, all_pen_layers, all_layer_lists,
            lock_layers=args.lock_layers,
        )

        # S3WD metrics and CSV preds
        if use_learned:
            metrics["learned_decision"] = True
            metrics["best_s3wd_params"] = {
                "mode": "learned_decision",
                "pct_within_3": metrics["pct_within_3"],
                "pct_within_5": metrics["pct_within_5"],
                "pct_over_10": metrics["pct_over_10"],
                "total": metrics["total"],
            }
            csv_preds = all_preds  # learned mode uses all_preds directly
            sources = ["learned"] * len(all_preds)
        else:
            best_metric, best_params, s3wd_preds, sources = _s3wd_fixed_eval(
                all_frame_probs, all_labels, all_pen_layers, all_layer_lists, args.lock_layers,
                wait=args.s3wd_wait, threshold=args.s3wd_threshold, accept=args.s3wd_accept,
            )
            # Re-compute metrics on S3WD preds — matches train.py's final compute_metrics call
            metrics = compute_metrics(
                s3wd_preds, all_labels, all_pen_layers, all_layer_lists,
                lock_layers=args.lock_layers,
            )
            metrics["best_s3wd_metric"] = best_metric
            metrics["best_s3wd_params"] = best_params
            metrics["learned_decision"] = False
            csv_preds = s3wd_preds
        csv_rows = []
        for sp, true_layer, pred_layer, layers, src in zip(
                all_sample_paths, all_pen_layers, csv_preds, all_layer_lists, sources):
            if true_layer >= 0 and pred_layer >= 0 and pred_layer in layers:
                error = abs(layers.index(pred_layer) - layers.index(true_layer))
            else:
                error = -1
            csv_rows.append({
                "hole_path": sp,
                "true_layer": true_layer,
                "pred_layer": pred_layer,
                "error": error,
                "decision_source": src,
            })

        with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["hole_path", "true_layer", "pred_layer", "error", "decision_source"])
            writer.writeheader()
            writer.writerows(csv_rows)
        print(f"\n[Inference] CSV saved to {args.output_csv}")
        print_metrics(metrics, prefix="[Inference]")
        print("\n[DONE]")
        return

    # ---- 模式 A：预计算特征（batch DataLoader，最快）----
    print(f"[Cached Mode] Dataset size: {len(dataset)}, Building DataLoader...")
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_masked_batch,
        pin_memory=(device.type == "cuda"),
        prefetch_factor=2 if args.num_workers > 0 else None,
    )

    csv_rows, metrics = run_inference(
        model, loader, device,
        decision_method=args.decision_method,
        s3wd_wait=args.s3wd_wait,
        s3wd_threshold=args.s3wd_threshold,
        s3wd_accept=args.s3wd_accept,
        lock_layers=args.lock_layers,
        use_learned=(args.decision_method == "learned"),
    )

    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["hole_path", "true_layer", "pred_layer", "error", "decision_source"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\n[Inference] CSV saved to {args.output_csv}")
    print_metrics(metrics, prefix="[Inference]")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
