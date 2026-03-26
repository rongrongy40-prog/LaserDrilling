# -*- coding: utf-8 -*-
"""
多种穿透层决策方法对比 + 每种方法的阈值网格搜索。

功能：对验证/测试集做一次模型前向得到每孔概率曲线；对 Argmax、SmoothFirst、Centroid、
      TopKMedian、TwoStage、FirstThresh、S3WD 七类决策分别做参数网格搜索，按“误差≤5 层占比”取最优参数；
      输出各方法最优参数与对应指标（n_penetrated, ≤3/≤5/>10 层占比），并标出全局最优方法。
数据来源二选一：① 预计算 cache（--cache_dir 指定目录）；② 直接推理（--no_cache 时从 samples_info 用 Dataset 读图）。
依赖：search_s3wd_thresholds（load_val_from_cache, run_forward_batch, _true_idx_from_layer_list）、inference、tcn_model、dataset。
输出：终端表格；可选 --output 写入 JSON。
示例：
  python compare_decision_methods.py --cache_dir ./cache_features_train --device cuda --output results.json
  python compare_decision_methods.py --no_cache --samples_info ../data_drilling/samples_info_test.json --device cuda --output test_results.json
"""

import os
import sys
import json
import argparse
import numpy as np
from itertools import product

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import torch
from search_s3wd_thresholds import (
    load_val_from_cache,
    run_forward_batch,
    _true_idx_from_layer_list,
    SAFETY_LOCK_LAYERS,
)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable


# ---------------------------------------------------------------------------
# 决策方法：输入 probs（已安全锁），输出 (is_penetrated, pred_idx)
# ---------------------------------------------------------------------------

def decide_argmax(probs, min_thresh=0.5):
    """取概率最大的层为穿透层；若最大概率 < min_thresh 则判未穿透。"""
    probs = np.asarray(probs).ravel()
    if len(probs) == 0:
        return False, None
    pred_idx = int(np.argmax(probs))
    is_pen = probs[pred_idx] >= min_thresh
    return is_pen, pred_idx if is_pen else None


def decide_smooth_first(probs, window=5, thresh=0.9):
    """先滑动平均再取首次 >= thresh 的层。"""
    probs = np.asarray(probs).ravel()
    if len(probs) == 0:
        return False, None
    kernel = np.ones(window) / window
    smoothed = np.convolve(probs, kernel, mode="same")
    for t in range(len(smoothed)):
        if smoothed[t] >= thresh:
            return True, t
    return False, None


def decide_centroid(probs, thresh=0.5):
    """在 prob > thresh 的层上做概率加权平均索引，四舍五入。"""
    probs = np.asarray(probs).ravel()
    if len(probs) == 0 or np.max(probs) < thresh:
        return False, None
    mask = probs > thresh
    weight_sum = np.sum(probs[mask])
    if weight_sum <= 0:
        return False, None
    indices = np.where(mask)[0]
    idx_float = np.sum(indices * probs[mask]) / weight_sum
    pred_idx = int(np.round(idx_float))
    pred_idx = max(0, min(pred_idx, len(probs) - 1))
    return True, pred_idx


def decide_topk_median(probs, k=5, min_thresh=0.5):
    """取概率最大的 K 个层的索引中位数作为穿透层。"""
    probs = np.asarray(probs).ravel()
    if len(probs) == 0 or np.max(probs) < min_thresh:
        return False, None
    k = min(k, len(probs))
    top_indices = np.argsort(probs)[-k:]
    pred_idx = int(np.median(top_indices))
    return True, pred_idx


def decide_twostage(probs, region_thresh=0.5, peak_thresh=0.9):
    """先找连续 prob > region_thresh 的区间，再在区间内取首次 >= peak_thresh 或 argmax。"""
    probs = np.asarray(probs).ravel()
    if len(probs) == 0:
        return False, None
    # 找连续高概率区间
    above = probs > region_thresh
    in_run = False
    start = 0
    for t in range(len(above)):
        if above[t] and not in_run:
            start = t
            in_run = True
        elif not above[t] and in_run:
            # 区间 [start, t)
            segment = probs[start:t]
            # 区间内首次 >= peak_thresh
            for i, p in enumerate(segment):
                if p >= peak_thresh:
                    return True, start + i
            # 否则区间内 argmax
            return True, start + int(np.argmax(segment))
        if t == len(above) - 1 and in_run:
            segment = probs[start : t + 1]
            for i, p in enumerate(segment):
                if p >= peak_thresh:
                    return True, start + i
            return True, start + int(np.argmax(segment))
    return False, None


def decide_s3wd(probs, accept=0.94, reject=0.45, wait=6):
    """序贯三支（用于对比）。"""
    from inference import s3wd_decide
    is_pen, idx = s3wd_decide(probs, accept_thresh=accept, reject_thresh=reject, wait_consecutive=wait)
    return is_pen, idx


def decide_first_thresh(probs, thresh=0.9):
    """首次达到 thresh 的层（用于对比）。"""
    probs = np.asarray(probs).ravel()
    for t in range(len(probs)):
        if probs[t] >= thresh:
            return True, t
    return False, None


# ---------------------------------------------------------------------------
# 统一指标计算
# ---------------------------------------------------------------------------

def compute_metrics_with_decide(val_list, probs_list, decide_fn):
    """用给定的 decide_fn(probs) -> (is_penetrated, pred_idx) 统计指标。"""
    n_penetrated = 0
    n_within_3, n_within_5, n_over_10 = 0, 0, 0
    for v, probs in zip(val_list, probs_list):
        if v["is_penetrated"] != 1:
            continue
        true_idx = v["true_idx"]
        if true_idx is None:
            n_penetrated += 1
            n_over_10 += 1
            continue
        n_penetrated += 1
        _, pred_idx = decide_fn(probs)
        if pred_idx is None:
            error = 999
        else:
            error = abs(pred_idx - true_idx)
        if error <= 3:
            n_within_3 += 1
        if error <= 5:
            n_within_5 += 1
        if error > 10:
            n_over_10 += 1
    pct_3 = (n_within_3 / n_penetrated * 100) if n_penetrated else 0.0
    pct_5 = (n_within_5 / n_penetrated * 100) if n_penetrated else 0.0
    pct_over10 = (n_over_10 / n_penetrated * 100) if n_penetrated else 0.0
    return {
        "n_penetrated": n_penetrated,
        "pct_within_3": pct_3,
        "pct_within_5": pct_5,
        "pct_over_10": pct_over10,
    }


def load_from_dataset(samples_info_path, base_dir=None, target_size=(128, 128), roi_size=96, grid=(8, 8),
                     lock_layers=SAFETY_LOCK_LAYERS):
    """
    从 samples_info 用 GridDiffDrillingDataset 逐样本读图并构建与 load_val_from_cache 相同结构的列表，
    用于无预计算 cache 时直接推理（如测试集）。
    返回: (val_list, stats_dict)，val_list 每项含 data, layer_list, true_layer, true_idx, is_penetrated。
    """
    from dataset import GridDiffDrillingDataset
    dataset = GridDiffDrillingDataset(
        samples_info_path,
        target_size=target_size,
        roi_size=roi_size,
        grid=grid,
        base_dir=base_dir,
    )
    val_list = []
    n_penetrated = 0
    n_skipped_exception = 0
    n_skipped_empty = 0
    for i in tqdm(range(len(dataset)), desc="加载样本(读图)", unit="孔"):
        try:
            raw = dataset[i]
        except Exception:
            n_skipped_exception += 1
            continue
        data = raw.get("data")
        layer_list = raw.get("layer_list", [])
        if data is None or (hasattr(data, "numel") and data.numel() == 0) or len(layer_list) == 0:
            n_skipped_empty += 1
            continue
        sample = dataset.samples[i]
        is_penetrated = int(sample.get("is_penetrated", 0))
        true_layer = int(sample.get("penetration_layer", -1))
        true_idx = _true_idx_from_layer_list(true_layer, layer_list) if is_penetrated else None
        if is_penetrated:
            n_penetrated += 1
        val_list.append({
            "data": data,
            "layer_list": layer_list,
            "true_layer": true_layer,
            "true_idx": true_idx,
            "is_penetrated": is_penetrated,
        })
    stats = {
        "n_val_expected": len(dataset),
        "n_val_penetrated_expected": n_penetrated,
        "n_loaded": len(val_list),
        "n_loaded_penetrated": n_penetrated,
        "n_skipped_exception": n_skipped_exception,
        "n_skipped_empty": n_skipped_empty,
    }
    return val_list, stats


def _grid_search_method(val_list, probs_list, method_name, param_grid, build_decide_fn, filter_params=None):
    """对一种方法做网格搜索，返回 (best_params, best_metrics)。filter_params(params)->False 时跳过该组合。"""
    best_pct_5 = -1.0
    best_params = None
    best_metrics = None
    keys = list(param_grid.keys())
    values = list(param_grid.values())
    for combo in product(*values):
        params = dict(zip(keys, combo))
        if filter_params is not None and not filter_params(params):
            continue
        decide_fn = build_decide_fn(**params)
        m = compute_metrics_with_decide(val_list, probs_list, decide_fn)
        if m["pct_within_5"] > best_pct_5:
            best_pct_5 = m["pct_within_5"]
            best_params = params
            best_metrics = m
    return best_params, best_metrics


def main():
    parser = argparse.ArgumentParser(description="多种决策方法 + 阈值网格搜索（可来自 cache 或直接读图推理）")
    parser.add_argument("--samples_info", type=str, default="/home/student2025/wudf2025/dinov3-main/data_drilling/samples_info_train.json", help="samples_info 路径；不传时：若 --output 含 test 则用 samples_info_test.json(205孔)，否则用 samples_info_train.json(820孔)")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--cache_dir", type=str, default="/home/student2025/wudf2025/dinov3-main/grid_diff_tcn/cache_features_train", help="预计算特征目录；不指定时按 output 推断：含 test→cache_features_test，否则→cache_features_train；--no_cache 时忽略")
    parser.add_argument("--no_cache", action="store_true", help="不使用 cache，用 Dataset 读图并前向（如对测试集 samples_info_test.json 直接推理）")
    parser.add_argument("--ckpt", type=str, default="grid_diff_tcn.pt")
    parser.add_argument("--val_ratio", type=float, default=1.0, help="使用 cache 时从 samples_info 取做验证的比例；默认 1.0=全量；no_cache 时忽略")
    parser.add_argument("--val_seed", type=int, default=42)
    parser.add_argument("--lock_layers", type=int, default=30)
    parser.add_argument("--img_size", type=int, default=128, help="直接推理时的读图尺寸（与训练一致）")
    parser.add_argument("--roi_size", type=int, default=96, help="直接推理时的 ROI 边长")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=str, default=None, help="结果 JSON 路径")
    args = parser.parse_args()

    # 按 output 推断 samples_info 与 cache_dir（未显式指定时）
    data_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "data_drilling")
    is_test = args.output and "test" in args.output.lower()
    if args.samples_info is None:
        args.samples_info = os.path.join(data_dir, "samples_info_test.json" if is_test else "samples_info_train.json")
    if args.cache_dir is None and not args.no_cache:
        args.cache_dir = os.path.join(SCRIPT_DIR, "cache_features_test" if is_test else "cache_features_train")
    args.samples_info = os.path.normpath(args.samples_info.replace("\\", os.sep))
    if not os.path.isfile(args.samples_info):
        print("未找到 samples_info:", args.samples_info)
        return
    # 打印当前使用的集合与规模，便于核对
    try:
        with open(args.samples_info, "r", encoding="utf-8") as f:
            raw = json.load(f)
        raw = raw.get("Categories", raw) if isinstance(raw, dict) else raw
        n_total = len(raw) if isinstance(raw, list) else 0
        print(f"使用: {os.path.basename(args.samples_info)}（共 {n_total} 孔）")
    except Exception:
        pass
    if not os.path.isfile(args.ckpt):
        print("未找到模型:", args.ckpt)
        return

    use_cache = not args.no_cache and args.cache_dir and os.path.isdir(args.cache_dir)
    device = torch.device(args.device)
    if args.cache_dir and not use_cache and not args.no_cache:
        print(f"未找到 cache 目录，将直接读图: {args.cache_dir}")

    if use_cache:
        print(f"加载数据（从预计算 cache）: {args.cache_dir}")
        val_list, load_stats = load_val_from_cache(
            args.cache_dir,
            args.samples_info,
            val_ratio=args.val_ratio,
            val_seed=args.val_seed,
            base_dir=args.base_dir,
        )
        print(f"验证集: 加载 {load_stats['n_loaded']} 孔，其中穿透孔 {load_stats['n_loaded_penetrated']}")
        if load_stats.get("n_skipped_no_cache", 0) or load_stats.get("n_skipped_empty", 0):
            print(f"  未读全: 缺缓存 {load_stats.get('n_skipped_no_cache', 0)} 个, 空数据 {load_stats.get('n_skipped_empty', 0)} 个（建议 --val_ratio 1.0 用全量时确保 cache 与 samples_info 一致）")
    else:
        print("加载数据（直接读图，无 cache）...")
        val_list, load_stats = load_from_dataset(
            args.samples_info,
            base_dir=args.base_dir,
            target_size=(args.img_size, args.img_size),
            roi_size=args.roi_size,
        )
        print(f"样本: 加载 {load_stats['n_loaded']} 孔，其中穿透孔 {load_stats['n_loaded_penetrated']}")
        if load_stats.get("n_skipped_exception", 0) or load_stats.get("n_skipped_empty", 0):
            print(f"  未读全: 异常跳过 {load_stats.get('n_skipped_exception', 0)} 个, 空数据 {load_stats.get('n_skipped_empty', 0)} 个")

    print("加载模型并前向（仅一次）...")
    ckpt = torch.load(args.ckpt, map_location=device)
    from tcn_model import GridDiffTCN
    model = GridDiffTCN(in_channels=64, out_channels=2, num_channels=(64, 64, 64, 64), kernel_size=3)
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)
    probs_list = run_forward_batch(model, val_list, device, lock_layers=args.lock_layers)

    # 每种方法的网格、决策函数构造、可选参数过滤（如 S3WD 要求 reject < accept）
    method_configs = [
        ("Argmax", {"min_thresh": [0.3, 0.4, 0.5, 0.6, 0.7]}, lambda min_thresh: (lambda p: decide_argmax(p, min_thresh=min_thresh)), None),
        ("SmoothFirst", {"window": [3, 5, 7], "thresh": [0.85, 0.9, 0.95]}, lambda window, thresh: (lambda p: decide_smooth_first(p, window=window, thresh=thresh)), None),
        ("Centroid", {"thresh": [0.4, 0.5, 0.6, 0.7]}, lambda thresh: (lambda p: decide_centroid(p, thresh=thresh)), None),
        ("TopKMedian", {"k": [3, 5, 7, 9], "min_thresh": [0.4, 0.5, 0.6]}, lambda k, min_thresh: (lambda p: decide_topk_median(p, k=k, min_thresh=min_thresh)), None),
        ("TwoStage", {"region_thresh": [0.4, 0.5, 0.6], "peak_thresh": [0.85, 0.9, 0.95]}, lambda region_thresh, peak_thresh: (lambda p: decide_twostage(p, region_thresh=region_thresh, peak_thresh=peak_thresh)), None),
        ("FirstThresh", {"thresh": [0.75, 0.8, 0.85, 0.9, 0.95]}, lambda thresh: (lambda p: decide_first_thresh(p, thresh=thresh)), None),
        ("S3WD", {"accept": [0.88, 0.9, 0.92, 0.94, 0.98], "reject": [0.4, 0.5, 0.6], "wait": [3, 4, 5, 6]}, lambda accept, reject, wait: (lambda p: decide_s3wd(p, accept=accept, reject=reject, wait=wait)), lambda p: p["reject"] < p["accept"]),
    ]

    results = []
    for method_name, param_grid, build_fn, filter_params in method_configs:
        n_combos = 1
        for v in param_grid.values():
            n_combos *= len(v)
        print(f"网格搜索: {method_name}（{n_combos} 组）...", end=" ", flush=True)
        best_params, best_metrics = _grid_search_method(val_list, probs_list, method_name, param_grid, build_fn, filter_params)
        name_str = f"{method_name}({best_params})"
        results.append({"method": method_name, "best_params": best_params, "metrics": best_metrics})
        print(f"最优 ≤5层: {best_metrics['pct_within_5']:.1f}%")

    print("\n" + "=" * 90)
    print("【各方法网格搜索最优结果】")
    print("=" * 90)
    print(f"{'方法':<20} {'n_pen':>8} {'≤3层%':>10} {'≤5层%':>10} {'>10层%':>10}  最优参数")
    print("-" * 90)
    for r in results:
        m = r["metrics"]
        param_str = json.dumps(r["best_params"], ensure_ascii=False)
        if len(param_str) > 42:
            param_str = param_str[:39] + "..."
        print(f"{r['method']:<20} {m['n_penetrated']:>8} {m['pct_within_3']:>9.1f}% {m['pct_within_5']:>9.1f}% {m['pct_over_10']:>9.1f}%  {param_str}")
    print("=" * 90)
    best_row = max(results, key=lambda x: x["metrics"]["pct_within_5"])
    print(f"全局最优(按≤5层): {best_row['method']}  {best_row['best_params']}  →  ≤5层: {best_row['metrics']['pct_within_5']:.1f}%")

    if args.output:
        out_data = [{"method": r["method"], "best_params": r["best_params"], "metrics": r["metrics"]} for r in results]
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(out_data, f, ensure_ascii=False, indent=2)
        print(f"\n结果已保存: {args.output}")


if __name__ == "__main__":
    main()
