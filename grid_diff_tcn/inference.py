# -*- coding: utf-8 -*-
"""
模块四：结合物理先验的 S3WD 推理 (Inference 模块)。

功能：加载 GridDiffTCN 权重，对 samples_info 中每孔做整序列前向，得到每层穿透概率；
      应用物理安全锁（前 30 层概率置 0）与序贯三支决策 (S3WD)，输出是否穿透及穿透层索引；
      汇总按孔指标（仅穿透孔：误差≤3/≤5/>10 层占比）并写入 JSON。
依赖：dataset.py, tcn_model.py；samples_info.json；已训练权重（如 grid_diff_tcn.pt）。
输出：终端打印按孔指标；--output 指定路径写入逐孔推理结果 JSON。
主要参数：--samples_info, --ckpt, --output, --lock_layers, --device。
示例：python inference.py --samples_info ../data_drilling/samples_info_test.json --ckpt grid_diff_tcn.pt --output inference_results.json --device cuda
"""

import os
import sys
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from dataset import GridDiffDrillingDataset, collate_fn
from tcn_model import GridDiffTCN
from transformer_tcn_model import build_tcn_or_transformer, GridDiffTCNWithTransformer


# ---------------------------------------------------------------------------
# 物理安全锁 + S3WD 状态机
# ---------------------------------------------------------------------------

# 前 SAFETY_LOCK_LAYERS 层强制认为未穿透（物理上不可能已打穿）
SAFETY_LOCK_LAYERS = 30

# S3WD 阈值
PROB_ACCEPT = 0.9   # >= 此值立即判定穿透
PROB_REJECT = 0.75    # <= 此值判定未穿透，并清零 Wait 计数
WAIT_CONSECUTIVE = 3 # Wait 区内连续达到此层数才确认穿透


def apply_safety_lock(probs, lock_layers=SAFETY_LOCK_LAYERS):
    """
    将前 lock_layers 层的穿透概率强制置 0。
    probs: 一维数组或张量，形状 (T,)，表示每层“穿透”类的概率
    """
    if torch.is_tensor(probs):
        probs = probs.clone()
    else:
        probs = np.asarray(probs, dtype=np.float64).copy()
    n = min(lock_layers, len(probs))
    if n > 0:
        probs[:n] = 0.0
    return probs


def s3wd_decide(probs, accept_thresh=PROB_ACCEPT, reject_thresh=PROB_REJECT, wait_consecutive=WAIT_CONSECUTIVE):
    """
    序贯三支决策 (S3WD)：
    - P_t >= accept_thresh：立即 Accept，返回当前层索引（0-based）及“已穿透”
    - reject_thresh < P_t < accept_thresh：进入 Wait，需连续 wait_consecutive 层都在此区间及以上才 Accept
    - P_t <= reject_thresh：Reject，清零 Wait 计数，继续往后

    probs: 一维数组，长度 T，已做过安全锁；每层对应一个穿透概率
    返回: (is_penetrated: bool, penetration_layer_index: int 或 None)
           penetration_layer_index 为 0-based 序列下标；若未穿透则为 None（或 -1，由调用方约定）
    """
    probs = np.asarray(probs).ravel()
    T = len(probs)
    wait_count = 0

    for t in range(T):
        p = float(probs[t])
        if p >= accept_thresh:
            return True, t
        if p > reject_thresh:
            wait_count += 1
            if wait_count >= wait_consecutive:
                return True, t - wait_consecutive + 1
        else:
            wait_count = 0
    return False, None


# ---------------------------------------------------------------------------
# TopKMedian 决策（替代 S3WD）
# ---------------------------------------------------------------------------

def topkmedian_decide(probs, k: int = 9, min_thresh: float = 0.4):
    """
    TopKMedian 决策：
      - 取穿透概率最大的 k 个层索引与概率值；
      - 若这 k 个概率的中位数 < min_thresh，则判定未穿透；
      - 否则判定穿透，并将预测层索引设为 top-k 索引的中位数（稳健，抗单点尖峰）。

    probs: 一维数组/张量，形状 (T,)
    返回: (is_penetrated: bool, penetration_layer_index: int 或 None)
    """
    probs = np.asarray(probs).ravel()
    T = len(probs)
    if T == 0:
        return False, None
    kk = int(max(1, min(k, T)))
    # 取 top-k 下标（按概率降序）
    topk_idx = np.argsort(-probs)[:kk]
    topk_vals = probs[topk_idx]
    med_val = float(np.median(topk_vals))
    if med_val < float(min_thresh):
        return False, None
    # 预测层：top-k 下标的中位数
    topk_idx_sorted = np.sort(topk_idx)
    pred_idx = int(topk_idx_sorted[len(topk_idx_sorted) // 2])
    return True, pred_idx


def _uncertainty_topk_median(mean_probs: np.ndarray, var_probs: np.ndarray, k: int) -> float:
    """与 TopK 相同的 k 个时间步上，逐层方差的中位数。"""
    mean_probs = np.asarray(mean_probs).ravel()
    var_probs = np.asarray(var_probs).ravel()
    T = len(mean_probs)
    if T == 0 or len(var_probs) != T:
        return 0.0
    kk = int(max(1, min(k, T)))
    topk_idx = np.argsort(-mean_probs)[:kk]
    return float(np.median(var_probs[topk_idx]))


def topkmedian_with_uncertainty_gate(
    mean_probs,
    var_probs,
    k: int = 9,
    min_thresh: float = 0.4,
    unc_var_median_thresh: float = 0.05,
):
    """
    先对 mean_probs 做 TopKMedian；若判为穿透，再看不确定性：
      若 top-k（与 TopKMedian 相同）位置上 var 的中位数 > unc_var_median_thresh，
      则改判为未穿透（认为模型在该区域不稳定）。
    var_probs: 每层多次采样概率的方差，长度 T。
    返回: (is_penetrated, pred_idx, meta dict)
    """
    mean_probs = np.asarray(mean_probs).ravel()
    var_probs = np.asarray(var_probs).ravel() if var_probs is not None else None
    is_pen, pred_idx = topkmedian_decide(mean_probs, k=k, min_thresh=min_thresh)
    meta = {"unc_gated": False, "unc_topk_median": None}
    if not is_pen or var_probs is None or len(var_probs) != len(mean_probs):
        return is_pen, pred_idx, meta
    med_var = _uncertainty_topk_median(mean_probs, var_probs, k)
    meta["unc_topk_median"] = med_var
    if med_var > float(unc_var_median_thresh):
        meta["unc_gated"] = True
        return False, None, meta
    return is_pen, pred_idx, meta


def _forward_probs(model, data_tensor, device, force_sample_attention: bool = False) -> torch.Tensor:
    """
    单次前向，返回每层穿透概率（未做安全锁），形状 (T,)。
    force_sample_attention: 仅 GridDiffTCNWithTransformer 有效，True 时推理仍对 K/V 采样。
    """
    if data_tensor.dim() == 2:
        data_tensor = data_tensor.unsqueeze(0).transpose(1, 2)
    if data_tensor.size(1) != 64:
        data_tensor = data_tensor.transpose(0, 1).unsqueeze(0)
    x = data_tensor.to(device)
    if isinstance(model, GridDiffTCNWithTransformer):
        out = model(x, force_sample_attention=force_sample_attention)
    else:
        out = model(x)
    if isinstance(out, tuple):
        logits, _ = out
    else:
        logits = out
    logits = logits.cpu()
    probs = F.softmax(logits, dim=1)[0, 1, :]  # (T,)
    return probs


def run_inference(model, data_tensor, layer_list, device, lock_layers=SAFETY_LOCK_LAYERS):
    """
    对单孔完整序列做推理，得到每层穿透概率并做安全锁与 S3WD 决策。

    model: GridDiffTCN
    data_tensor: (1, 64, T) 或 (T, 64) 若为 (T,64) 会转成 (1,64,T)
    layer_list: 长度 T 的列表，每元素为该时间步对应的层号（用于返回“层号”而非下标）
    返回: dict {
        "probs": (T,) 每层穿透概率（已安全锁）,
        "layer_list": layer_list,
        "is_penetrated": bool,
        "penetration_layer_index": int 或 None,  # 0-based 下标
        "penetration_layer_number": int 或 None # 真实层号，对应 layer_list[penetration_layer_index]
    }
    """
    model.eval()
    with torch.no_grad():
        probs_t = _forward_probs(model, data_tensor, device)
    probs = probs_t.numpy()
    probs = apply_safety_lock(probs, lock_layers)
    is_penetrated, pen_idx = s3wd_decide(probs)
    pen_layer_num = None
    if pen_idx is not None and layer_list and 0 <= pen_idx < len(layer_list):
        pen_layer_num = layer_list[pen_idx]
    return {
        "probs": probs,
        "layer_list": layer_list,
        "is_penetrated": is_penetrated,
        "penetration_layer_index": pen_idx,
        "penetration_layer_number": pen_layer_num,
    }


def run_inference_topkmedian(
    model,
    data_tensor,
    layer_list,
    device,
    lock_layers=SAFETY_LOCK_LAYERS,
    k: int = 9,
    min_thresh: float = 0.4,
    unc_samples: int = 1,
    use_uncertainty_gate: bool = False,
    unc_var_median_thresh: float = 0.05,
):
    """
    对单孔完整序列做推理，得到每层穿透概率并做安全锁与 TopKMedian 决策。

    - unc_samples==1（默认）：与原先完全一致，单次前向。
    - unc_samples>1：多次前向（Transformer 下对 K/V 采样），对每层概率取均值/方差；
      决策在 **均值概率** 上做 TopKMedian；若 use_uncertainty_gate=True，
      再按 top-k 位置上 **方差中位数** 是否超过 unc_var_median_thresh 决定是否改判未穿透。
    """
    model.eval()
    unc_samples = max(1, int(unc_samples))
    force_mc = isinstance(model, GridDiffTCNWithTransformer) and unc_samples > 1

    if unc_samples <= 1:
        with torch.no_grad():
            probs_t = _forward_probs(model, data_tensor, device, force_sample_attention=False)
        probs = probs_t.numpy()
        probs = apply_safety_lock(probs, lock_layers)
        is_penetrated, pen_idx = topkmedian_decide(probs, k=k, min_thresh=min_thresh)
        pen_layer_num = None
        if pen_idx is not None and layer_list and 0 <= pen_idx < len(layer_list):
            pen_layer_num = layer_list[pen_idx]
        return {
            "probs": probs,
            "probs_mean": probs,
            "probs_var": None,
            "layer_list": layer_list,
            "is_penetrated": is_penetrated,
            "penetration_layer_index": pen_idx,
            "penetration_layer_number": pen_layer_num,
            "unc_topk_median": None,
            "unc_gated": False,
        }

    stacks = []
    with torch.inference_mode():
        for _ in range(unc_samples):
            p = _forward_probs(model, data_tensor, device, force_sample_attention=force_mc)
            stacks.append(p.detach().cpu().numpy())
    P = np.stack(stacks, axis=0)
    mean_p = P.mean(axis=0)
    var_p = P.var(axis=0)
    mean_locked = apply_safety_lock(np.asarray(mean_p, dtype=np.float64).copy(), lock_layers)

    if use_uncertainty_gate:
        is_penetrated, pen_idx, meta = topkmedian_with_uncertainty_gate(
            mean_locked, var_p, k=k, min_thresh=min_thresh, unc_var_median_thresh=unc_var_median_thresh
        )
        unc_gated = meta["unc_gated"]
        unc_topk_median = meta["unc_topk_median"]
    else:
        is_penetrated, pen_idx = topkmedian_decide(mean_locked, k=k, min_thresh=min_thresh)
        unc_gated = False
        unc_topk_median = _uncertainty_topk_median(mean_locked, var_p, k) if len(mean_locked) else None

    pen_layer_num = None
    if pen_idx is not None and layer_list and 0 <= pen_idx < len(layer_list):
        pen_layer_num = layer_list[pen_idx]
    return {
        "probs": mean_locked,
        "probs_mean": mean_locked,
        "probs_var": var_p,
        "layer_list": layer_list,
        "is_penetrated": is_penetrated,
        "penetration_layer_index": pen_idx,
        "penetration_layer_number": pen_layer_num,
        "unc_topk_median": unc_topk_median,
        "unc_gated": unc_gated,
    }


def predict_with_uncertainty(model, data_tensor, layer_list, device, lock_layers=SAFETY_LOCK_LAYERS, unc_samples: int = 0):
    """
    多次前向估计不确定性：
      - 重复 K 次前向得到 {probs_k}，形状 (K, T)
      - 计算平均概率 mean_probs 与方差 var_probs。
      - 使用 mean_probs 经过安全锁 + S3WD 得到最终决策。
    返回与 run_inference 相同字段，并额外提供:
      - "probs_mean", "probs_var"
    """
    if unc_samples <= 1:
        out = run_inference(model, data_tensor, layer_list, device, lock_layers)
        out["probs_mean"] = out["probs"]
        out["probs_var"] = None
        return out

    model.eval()
    force_mc = isinstance(model, GridDiffTCNWithTransformer)
    all_probs = []
    with torch.no_grad():
        for _ in range(unc_samples):
            probs_t = _forward_probs(
                model, data_tensor, device, force_sample_attention=force_mc
            )
            all_probs.append(probs_t.unsqueeze(0))
    probs_stack = torch.cat(all_probs, dim=0)  # (K, T)
    mean_probs = probs_stack.mean(dim=0)  # (T,)
    var_probs = probs_stack.var(dim=0, unbiased=False)  # (T,)

    probs_locked = apply_safety_lock(mean_probs.cpu().numpy(), lock_layers)
    is_penetrated, pen_idx = s3wd_decide(probs_locked)
    pen_layer_num = None
    if pen_idx is not None and layer_list and 0 <= pen_idx < len(layer_list):
        pen_layer_num = layer_list[pen_idx]

    return {
        "probs": probs_locked,
        "probs_mean": mean_probs.cpu().numpy(),
        "probs_var": var_probs.cpu().numpy(),
        "layer_list": layer_list,
        "is_penetrated": is_penetrated,
        "penetration_layer_index": pen_idx,
        "penetration_layer_number": pen_layer_num,
    }


def _load_one_hole_like_train(precomputed_dir, precomputed_name_for_idx, base_dataset, base_idx):
    """与 train._load_one_hole 完全一致：按孔从预计算目录加载 (data, layer_list)。"""
    pt_path = os.path.join(precomputed_dir, precomputed_name_for_idx[base_idx])
    if not os.path.isfile(pt_path):
        pt_path = os.path.join(precomputed_dir, f"{base_idx}.pt")
    if os.path.isfile(pt_path):
        raw = torch.load(pt_path, map_location="cpu")
        data = raw.get("data")
        layer_list = raw.get("layer_list", [])
        # 与 JSON 的 int 对齐，避免 true_layer in layer_list 因类型失败
        layer_list = [int(x) for x in layer_list if x is not None]
        return data, layer_list
    # 无缓存时回退读图（与 train 一致）
    raw = base_dataset[base_idx]
    ll = raw.get("layer_list", [])
    ll = [int(x) for x in ll if x is not None]
    return raw["data"], ll


def _build_precomputed_name_for_idx(base_dataset):
    """与 train.WindowedDrillingDataset 里预计算文件名表完全一致。"""
    used = set()

    def safe_basename(path):
        name = os.path.basename(path or "").strip().replace(os.sep, "_").replace("/", "_") or "unknown"
        return name

    out = []
    for i in range(len(base_dataset.samples)):
        path = base_dataset.samples[i].get("sample_path", "")
        base = safe_basename(path) or f"sample_{i}"
        name = base
        k = 2
        while name in used:
            name = f"{base}__{k}"
            k += 1
        used.add(name)
        out.append(name + ".pt")
    return out


def evaluate_dataset(
    model,
    dataset,
    device,
    lock_layers=SAFETY_LOCK_LAYERS,
    max_samples=None,
    unc_samples: int = 0,
    use_topkmedian: bool = False,
    topk_k: int = 9,
    topk_min_thresh: float = 0.4,
    use_uncertainty_gate: bool = False,
    unc_var_median_thresh: float = 0.05,
):
    """
    在数据集上逐样本推理并汇总指标。
    use_topkmedian=True 时用 TopKMedian 决策（与训练验证一致）；否则用 S3WD。
    """
    model.eval()
    results = []
    n = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    for idx in tqdm(range(n), desc="推理", unit="孔"):
        sample = dataset[idx]
        data = sample["data"]
        label = sample["label"]
        true_layer = sample["penetration_layer"]
        layer_list = sample["layer_list"]
        if use_topkmedian:
            out = run_inference_topkmedian(
                model,
                data,
                layer_list,
                device,
                lock_layers=lock_layers,
                k=topk_k,
                min_thresh=topk_min_thresh,
                unc_samples=max(1, int(unc_samples)),
                use_uncertainty_gate=use_uncertainty_gate,
                unc_var_median_thresh=unc_var_median_thresh,
            )
        elif unc_samples and unc_samples > 1:
            out = predict_with_uncertainty(model, data, layer_list, device, lock_layers, unc_samples)
        else:
            out = run_inference(model, data, layer_list, device, lock_layers)
        pred_pen = out["is_penetrated"]
        pred_layer = out["penetration_layer_number"]
        pred_idx = out["penetration_layer_index"]
        true_idx = None
        if int(label) == 1 and layer_list is not None and true_layer is not None:
            try:
                if true_layer in layer_list:
                    true_idx = layer_list.index(true_layer)
            except (ValueError, TypeError):
                pass
        probs = out.get("probs")
        if probs is None:
            probs = out.get("probs_mean")

        def _to_json_serializable(x):
            if x is None:
                return None
            if hasattr(x, "tolist"):
                return x.tolist()
            if hasattr(x, "__array__"):
                return np.asarray(x).tolist()
            return x

        results.append({
            "index": idx,
            "sample_path": sample.get("sample_path", ""),
            "true_label": int(label),
            "true_penetration_layer": true_layer,
            "true_penetration_index": true_idx,
            "pred_penetrated": pred_pen,
            "pred_penetration_layer": pred_layer,
            "pred_penetration_index": pred_idx,
            "probs": _to_json_serializable(probs) if probs is not None else [],
            "probs_mean": _to_json_serializable(out.get("probs_mean")),
            "probs_var": _to_json_serializable(out.get("probs_var")),
        })
    return results


def evaluate_dataset_precomputed(
    model,
    base_dataset,
    precomputed_dir,
    device,
    lock_layers=SAFETY_LOCK_LAYERS,
    max_samples=None,
    unc_samples: int = 0,
    use_topkmedian: bool = False,
    topk_k: int = 9,
    topk_min_thresh: float = 0.4,
    use_uncertainty_gate: bool = False,
    unc_var_median_thresh: float = 0.05,
):
    """
    使用与 train 验证完全一致的按孔加载方式（base_dataset + _precomputed_name_for_idx + _load_one_hole），
    在预计算目录上逐孔推理并汇总指标，保证与训练时验证逻辑一致。
    """
    precomputed_dir = os.path.abspath(precomputed_dir)
    precomputed_name_for_idx = _build_precomputed_name_for_idx(base_dataset)
    model.eval()
    n = len(base_dataset) if max_samples is None else min(max_samples, len(base_dataset))
    results = []
    tkm_kw = dict(
        lock_layers=lock_layers,
        k=topk_k,
        min_thresh=topk_min_thresh,
        unc_samples=max(1, int(unc_samples)),
        use_uncertainty_gate=use_uncertainty_gate,
        unc_var_median_thresh=unc_var_median_thresh,
    )

    def _to_json_serializable(x):
        if x is None:
            return None
        if hasattr(x, "tolist"):
            return x.tolist()
        if hasattr(x, "__array__"):
            return np.asarray(x).tolist()
        return x

    for idx in tqdm(range(n), desc="推理", unit="孔"):
        data, layer_list = _load_one_hole_like_train(
            precomputed_dir, precomputed_name_for_idx, base_dataset, idx
        )
        sample = base_dataset.samples[idx]
        label = sample["is_penetrated"]
        true_layer = sample.get("penetration_layer", -1)
        path = sample.get("sample_path", "")

        if data is None or (hasattr(data, "numel") and data.numel() == 0) or not layer_list:
            results.append({
                "index": idx,
                "sample_path": path,
                "true_label": int(label),
                "true_penetration_layer": true_layer,
                "true_penetration_index": None,
                "pred_penetrated": False,
                "pred_penetration_layer": None,
                "pred_penetration_index": None,
                "probs": [],
                "probs_mean": None,
                "probs_var": None,
            })
            continue

        if use_topkmedian:
            out = run_inference_topkmedian(
                model, data, layer_list, device, **tkm_kw
            )
        else:
            out = run_inference(model, data, layer_list, device, lock_layers)

        pred_pen = out["is_penetrated"]
        pred_layer = out["penetration_layer_number"]
        pred_idx = out["penetration_layer_index"]
        true_idx = None
        if int(label) == 1 and true_layer is not None:
            try:
                if true_layer in layer_list:
                    true_idx = layer_list.index(true_layer)
            except (ValueError, TypeError):
                pass

        probs = out.get("probs")
        if probs is None:
            probs = out.get("probs_mean")
        results.append({
            "index": idx,
            "sample_path": path,
            "true_label": int(label),
            "true_penetration_layer": true_layer,
            "true_penetration_index": true_idx,
            "pred_penetrated": pred_pen,
            "pred_penetration_layer": pred_layer,
            "pred_penetration_index": pred_idx,
            "probs": _to_json_serializable(probs) if probs is not None else [],
            "probs_mean": _to_json_serializable(out.get("probs_mean")),
            "probs_var": _to_json_serializable(out.get("probs_var")),
        })
    return results


def hole_level_metrics_from_results(results):
    """
    从逐孔推理结果计算按孔指标（仅统计标注为穿透的孔）。
    返回: dict 含 n_penetrated, pct_within_3, pct_within_5, pct_over_10。
    """
    n_penetrated = 0
    n_within_3, n_within_5, n_over_10 = 0, 0, 0
    for r in results:
        if r.get("true_label") != 1:
            continue
        true_idx = r.get("true_penetration_index")
        if true_idx is None:
            n_over_10 += 1
            n_penetrated += 1
            continue
        n_penetrated += 1
        pred_idx = r.get("pred_penetration_index")
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


def main():
    parser = argparse.ArgumentParser(description="Grid-Diff TCN 推理（S3WD 或 TopKMedian）")
    parser.add_argument("--samples_info", type=str, default=None, help="samples_info.json 路径")
    parser.add_argument("--base_dir", type=str, default=None)
    parser.add_argument("--ckpt", type=str, default="grid_diff_tcn.pt", help="模型权重路径")
    parser.add_argument("--output", type=str, default="inference_results.json", help="结果 JSON 路径")
    parser.add_argument("--lock_layers", type=int, default=30, help="物理安全锁层数")
    parser.add_argument("--max_samples", type=int, default=None, help="最多评估样本数，默认全部")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--roi_cy", type=float, default=0.5, help="ROI 中心垂直比例，需与训练一致")
    parser.add_argument("--roi_cx", type=float, default=0.5, help="ROI 中心水平比例，需与训练一致")
    parser.add_argument("--use_transformer", action="store_true", help="使用带 Transformer 的 GridDiff 模型")
    parser.add_argument("--num_transformer_layers", type=int, default=2)
    parser.add_argument("--attn_dim", type=int, default=64)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--unc_samples", type=int, default=0, help="不确定性采样次数，>1 时启用（TopKMedian 下有效）")
    parser.add_argument("--topkmedian", action="store_true", help="使用 TopKMedian 决策（与训练验证一致），默认 S3WD")
    parser.add_argument("--k", type=int, default=9, help="TopKMedian 的 k")
    parser.add_argument("--min_thresh", type=float, default=0.4, help="TopKMedian 的 min_thresh")
    parser.add_argument("--unc_gate", action="store_true", help="TopKMedian 下：方差中位数超阈值则否决穿透")
    parser.add_argument("--unc_var_median_thresh", type=float, default=0.1, help="与 --unc_gate 配合的方差中位数阈值")
    parser.add_argument("--precomputed_dir", type=str, default=None, help="预计算特征目录（训练集推理时用 cache_features_train，跳过读图）")
    args = parser.parse_args()

    if args.samples_info is None:
        args.samples_info = os.path.join(SCRIPT_DIR, "..", "data_drilling", "samples_info.json")
    if not os.path.isfile(args.samples_info):
        print("未找到 samples_info.json")
        return
    if not os.path.isfile(args.ckpt):
        print("未找到模型权重:", args.ckpt)
        return

    device = torch.device(args.device)
    ckpt = torch.load(args.ckpt, map_location=device)
    use_transformer = getattr(args, "use_transformer", False)
    model = build_tcn_or_transformer(
        use_transformer=use_transformer,
        in_channels=64,
        out_channels=2,
        tcn_channels=(64, 64, 64, 64),
        kernel_size=3,
        d_model=getattr(args, "attn_dim", 64),
        nhead=getattr(args, "num_heads", 4),
        num_layers=getattr(args, "num_transformer_layers", 2),
        dim_feedforward=256,
        dropout=0.1,
        add_kl=True,
        return_kl=False,
    )
    if "model" in ckpt:
        model.load_state_dict(ckpt["model"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    base_dataset = GridDiffDrillingDataset(
        args.samples_info,
        target_size=(128, 128),
        roi_size=96,
        grid=(8, 8),
        base_dir=args.base_dir,
        roi_center_yx=(args.roi_cy, args.roi_cx),
    )
    precomputed_dir = getattr(args, "precomputed_dir", None) and os.path.normpath(args.precomputed_dir.strip())
    if precomputed_dir:
        if not os.path.isdir(precomputed_dir):
            absp = os.path.abspath(precomputed_dir)
            print(f"错误：预计算目录不存在: {absp}")
            if "/path/to" in absp or absp == "/path/to/cache_features_train":
                print("  若用 inference_train.sh 默认路径，请勿设置环境变量 PRECOMPUTED_DIR 为占位路径。")
            return
        n_pt = len([f for f in os.listdir(precomputed_dir) if f.endswith(".pt")])
        print(f"使用预计算特征: {os.path.abspath(precomputed_dir)}（共 {n_pt} 个 .pt）")
        # 与 train 验证完全一致：同一 base_dataset + 同一 _precomputed_name_for_idx + 同一 _load_one_hole 逻辑
        results = evaluate_dataset_precomputed(
            model,
            base_dataset,
            precomputed_dir,
            device,
            lock_layers=args.lock_layers,
            max_samples=args.max_samples,
            unc_samples=getattr(args, "unc_samples", 0),
            use_topkmedian=getattr(args, "topkmedian", False),
            topk_k=getattr(args, "k", 9),
            topk_min_thresh=getattr(args, "min_thresh", 0.4),
            use_uncertainty_gate=getattr(args, "unc_gate", False),
            unc_var_median_thresh=getattr(args, "unc_var_median_thresh", 0.1),
        )
    else:
        results = evaluate_dataset(
            model,
            base_dataset,
            device,
            lock_layers=args.lock_layers,
            max_samples=args.max_samples,
            unc_samples=getattr(args, "unc_samples", 0),
            use_topkmedian=getattr(args, "topkmedian", False),
            topk_k=getattr(args, "k", 9),
            topk_min_thresh=getattr(args, "min_thresh", 0.4),
            use_uncertainty_gate=getattr(args, "unc_gate", False),
            unc_var_median_thresh=getattr(args, "unc_var_median_thresh", 0.1),
        )

    hole_metrics = hole_level_metrics_from_results(results)
    print(f"按孔指标（仅穿透孔）: 穿透孔数={hole_metrics['n_penetrated']}  误差≤3层: {hole_metrics['pct_within_3']:.1f}%  误差≤5层: {hole_metrics['pct_within_5']:.1f}%  误差>10层: {hole_metrics['pct_over_10']:.1f}%")

    def _make_json_serializable(obj):
        if obj is None:
            return None
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if hasattr(obj, "item"):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _make_json_serializable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_make_json_serializable(x) for x in obj]
        if isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        if isinstance(obj, (np.floating, np.float64, np.float32)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(_make_json_serializable(results), f, ensure_ascii=False, indent=2)
    print(f"推理结果已写入: {args.output}，共 {len(results)} 条")


if __name__ == "__main__":
    main()
