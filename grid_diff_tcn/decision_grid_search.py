#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline decision-method grid search using inference_results_{train,test}.json.

Goal:
  - Try multiple decision rules (TopKMedian/Argmax/FirstThresh/SmoothFirst/Centroid/TwoStage),
    plus uncertainty-aware variants using probs_var when available.
  - Grid-search hyperparameters over a *reasonably large* space.
  - Report best params by <=5-layer accuracy on:
      * train
      * test
      * combined (weighted by n_penetrated)

Inputs must be produced by inference.py and contain at least:
  - true_label (0/1)
  - true_penetration_index (int or null)
  - probs (list[float])  # probability curve after safety lock
Optionally:
  - probs_var (list[float])  # per-layer variance from MC sampling

Outputs:
  - decision_grid_results_train.json
  - decision_grid_results_test.json
  - decision_grid_results_combined.json
  - decision_grid_results.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


def _load_list(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, list):
        raise ValueError(f"Expected list JSON: {path}")
    return [x for x in obj if isinstance(x, dict)]


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


def _as_float_array(x: Any) -> Optional[np.ndarray]:
    if x is None:
        return None
    if isinstance(x, list) and len(x) > 0:
        return np.asarray(x, dtype=np.float32)
    return None


@dataclass(frozen=True)
class Hole:
    sample_path: str
    true_label: int
    true_idx: Optional[int]
    probs: np.ndarray
    var: Optional[np.ndarray]


def parse_holes(rows: Sequence[dict]) -> List[Hole]:
    holes: List[Hole] = []
    for r in rows:
        probs = _as_float_array(r.get("probs"))
        if probs is None:
            continue
        holes.append(
            Hole(
                sample_path=str(r.get("sample_path") or ""),
                true_label=int(r.get("true_label") or 0),
                true_idx=_as_int(r.get("true_penetration_index")),
                probs=probs,
                var=_as_float_array(r.get("probs_var")),
            )
        )
    return holes


def metrics_penetrated(holes: Sequence[Hole], pred_idx: Sequence[Optional[int]]) -> Dict[str, Any]:
    assert len(holes) == len(pred_idx)
    n_pen = 0
    n3 = n5 = no10 = 0
    missing = 0
    for h, p in zip(holes, pred_idx):
        if h.true_label != 1:
            continue
        n_pen += 1
        if h.true_idx is None or p is None:
            missing += 1
            no10 += 1
            continue
        e = abs(int(p) - int(h.true_idx))
        if e <= 3:
            n3 += 1
        if e <= 5:
            n5 += 1
        if e > 10:
            no10 += 1
    return {
        "n_penetrated": n_pen,
        "pct_within_3": (n3 / n_pen * 100.0) if n_pen else 0.0,
        "pct_within_5": (n5 / n_pen * 100.0) if n_pen else 0.0,
        "pct_over_10": (no10 / n_pen * 100.0) if n_pen else 0.0,
        "n_missing": missing,
        "n_within_3": n3,
        "n_within_5": n5,
        "n_over_10": no10,
    }


def _topkmedian_idx(p: np.ndarray, k: int) -> int:
    T = p.size
    kk = max(1, min(int(k), int(T)))
    topk_idx = np.argpartition(-p, kk - 1)[:kk]
    topk_idx_sorted = np.sort(topk_idx)
    return int(topk_idx_sorted[len(topk_idx_sorted) // 2])


def decide_argmax(p: np.ndarray, min_thresh: float) -> Optional[int]:
    if p.size == 0:
        return None
    i = int(np.argmax(p))
    return i if float(p[i]) >= float(min_thresh) else None


def decide_first_thresh(p: np.ndarray, thresh: float) -> Optional[int]:
    if p.size == 0:
        return None
    idx = np.where(p >= float(thresh))[0]
    return int(idx[0]) if idx.size else None


def decide_smooth_first(p: np.ndarray, window: int, thresh: float) -> Optional[int]:
    if p.size == 0:
        return None
    w = max(1, int(window))
    if w <= 1:
        return decide_first_thresh(p, thresh)
    kernel = np.ones(w, dtype=np.float32) / float(w)
    sm = np.convolve(p, kernel, mode="same")
    return decide_first_thresh(sm, thresh)


def decide_centroid(p: np.ndarray, thresh: float) -> Optional[int]:
    if p.size == 0:
        return None
    if float(np.max(p)) < float(thresh):
        return None
    mask = p >= float(thresh)
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    w = p[idx].astype(np.float64)
    c = float(np.sum(idx * w) / (np.sum(w) + 1e-12))
    return int(np.clip(int(round(c)), 0, p.size - 1))


def decide_topkmedian(p: np.ndarray, k: int, min_thresh: float) -> Optional[int]:
    if p.size == 0:
        return None
    kk = max(1, min(int(k), int(p.size)))
    # quick: need median(top-k vals) >= min_thresh
    topk_idx = np.argpartition(-p, kk - 1)[:kk]
    med_val = float(np.median(p[topk_idx]))
    if med_val < float(min_thresh):
        return None
    topk_idx_sorted = np.sort(topk_idx)
    return int(topk_idx_sorted[len(topk_idx_sorted) // 2])


def decide_two_stage(p: np.ndarray, region_thresh: float, min_len: int, peak_thresh: float) -> Optional[int]:
    """
    Two-stage heuristic:
      1) find contiguous regions where p >= region_thresh, keep the *best* region by mean(p)
         among those with length >= min_len.
      2) within that region, pick first index where p >= peak_thresh; if none, pick argmax in region.
      If no region qualifies, fall back to argmax with min_thresh=peak_thresh (conservative).
    """
    if p.size == 0:
        return None
    rt = float(region_thresh)
    ml = max(1, int(min_len))
    pt = float(peak_thresh)
    above = p >= rt
    if not np.any(above):
        return decide_argmax(p, min_thresh=pt)
    # find runs
    idx = np.where(above)[0]
    # split into runs by gaps
    runs = []
    start = idx[0]
    prev = idx[0]
    for i in idx[1:]:
        if i == prev + 1:
            prev = i
        else:
            runs.append((start, prev))
            start = prev = i
    runs.append((start, prev))
    # filter by length
    cand = []
    for a, b in runs:
        if (b - a + 1) >= ml:
            cand.append((a, b))
    if not cand:
        return decide_argmax(p, min_thresh=pt)
    # choose best region by mean probability
    best = max(cand, key=lambda ab: float(np.mean(p[ab[0] : ab[1] + 1])))
    a, b = best
    region = p[a : b + 1]
    hit = np.where(region >= pt)[0]
    if hit.size:
        return int(a + hit[0])
    return int(a + int(np.argmax(region)))


def decide_earliest_stable(p: np.ndarray, thresh: float, stable_len: int) -> Optional[int]:
    """
    Earliest-stable rule:
      Find the first index t such that p[t:t+stable_len] are all >= thresh.
      If none, fall back to first_thresh.
    """
    if p.size == 0:
        return None
    t = float(thresh)
    L = max(1, int(stable_len))
    above = p >= t
    if not np.any(above):
        return None
    if L <= 1:
        return decide_first_thresh(p, t)
    # sliding window AND via convolution over boolean
    run = np.convolve(above.astype(np.int32), np.ones(L, dtype=np.int32), mode="valid")
    hit = np.where(run >= L)[0]
    if hit.size:
        return int(hit[0])
    return decide_first_thresh(p, t)


def decide_earliest_topk(p: np.ndarray, k: int, min_thresh: float) -> Optional[int]:
    """
    Earliest-topk rule:
      Use TopKMedian's "is penetrated" check (median(top-k vals) >= min_thresh),
      but output the earliest index among top-k indices.
    Useful when long high plateaus cause late median-of-indices.
    """
    if p.size == 0:
        return None
    kk = max(1, min(int(k), int(p.size)))
    topk_idx = np.argpartition(-p, kk - 1)[:kk]
    if float(np.median(p[topk_idx])) < float(min_thresh):
        return None
    return int(np.min(topk_idx))

def apply_risk_adjust(p: np.ndarray, var: Optional[np.ndarray], beta: float) -> np.ndarray:
    """Risk-averse score: s = p - beta * sqrt(var). If var missing -> unchanged."""
    if var is None or var.size != p.size or beta <= 0:
        return p
    return p - float(beta) * np.sqrt(np.maximum(var, 0.0)).astype(np.float32)


def apply_gate(
    p_mean: np.ndarray,
    var: Optional[np.ndarray],
    pred: Optional[int],
    gate_mode: str,
    var_thresh: float,
    k_for_topk: int = 9,
    win: int = 7,
    action: str = "veto",
    fallback: str = "first",
    fallback_thresh: float = 0.75,
    fallback_k: int = 7,
    fallback_min_thresh: float = 0.35,
) -> Optional[int]:
    """
    If pred is penetrated, optionally veto based on uncertainty around relevant positions.
    Returns possibly-updated pred (None if veto, or a fallback prediction).
    """
    if pred is None:
        return None
    if var is None or var.size != p_mean.size:
        return pred
    vt = float(var_thresh)
    if vt <= 0:
        return pred

    def needs_gate() -> bool:
        if gate_mode == "pred":
            return float(var[pred]) > vt
        if gate_mode == "win":
            w = max(1, int(win))
            a = max(0, pred - w)
            b = min(p_mean.size, pred + w + 1)
            return float(np.median(var[a:b])) > vt
        # default: topk median var on mean probs
        kk = max(1, min(int(k_for_topk), int(p_mean.size)))
        topk_idx = np.argpartition(-p_mean, kk - 1)[:kk]
        return float(np.median(var[topk_idx])) > vt

    if not needs_gate():
        return pred

    act = str(action)
    if act == "veto":
        return None

    # fallback prediction instead of veto
    fb = str(fallback)
    if fb == "argmax":
        return decide_argmax(p_mean, min_thresh=float(fallback_thresh))
    if fb == "first":
        return decide_first_thresh(p_mean, thresh=float(fallback_thresh))
    if fb == "smooth_first":
        # use win as smoothing window for simplicity
        return decide_smooth_first(p_mean, window=int(win), thresh=float(fallback_thresh))
    if fb == "earliest_stable":
        return decide_earliest_stable(p_mean, thresh=float(fallback_thresh), stable_len=int(win))
    if fb == "earliest_topk":
        return decide_earliest_topk(p_mean, k=int(fallback_k), min_thresh=float(fallback_min_thresh))
    if fb == "topkmedian":
        return decide_topkmedian(p_mean, k=int(fallback_k), min_thresh=float(fallback_min_thresh))
    # default
    return pred


DecisionFn = Callable[[Hole, Dict[str, Any]], Optional[int]]
ParamSampler = Callable[[np.random.RandomState, int], List[Dict[str, Any]]]


def build_methods() -> Dict[str, Tuple[ParamSampler, DecisionFn]]:
    """
    Returns:
      name -> (param_sampler(rng, n)->list[params], decision_fn(hole, params)->pred_idx)
    Notes:
      We avoid full cartesian enumeration (too large). Samplers draw random combos
      from broad ranges (still "grid-like" because each dim is from a discrete set).
    """
    grids: Dict[str, Tuple[ParamSampler, DecisionFn]] = {}

    # Shared ranges (large, but we will *sample* from them for speed)
    min_threshs = np.round(np.arange(0.15, 0.86, 0.05), 2).tolist()  # 0.15..0.85
    threshs = np.round(np.arange(0.20, 0.91, 0.05), 2).tolist()  # 0.20..0.90
    ks = list(range(5, 32, 2))  # 5..31 odd
    smooth_ws = list(range(3, 32, 2))  # 3..31 odd
    betas = np.round(np.arange(0.0, 2.26, 0.25), 2).tolist()  # 0..2.25
    var_threshs = [0.0, 1e-5, 2e-5, 5e-5, 1e-4, 2e-4, 5e-4, 1e-3, 2e-3]
    gate_modes = ["topk", "pred", "win"]
    wins = [3, 5, 7, 9, 13]
    gate_actions = ["veto", "fallback"]
    fallbacks = ["first", "argmax", "topkmedian", "smooth_first", "earliest_stable", "earliest_topk"]

    # Argmax
    def fn_argmax(h: Hole, p: Dict[str, Any]) -> Optional[int]:
        pm = apply_risk_adjust(h.probs, h.var, p["beta"])
        pred = decide_argmax(pm, p["min_thresh"])
        return apply_gate(
            h.probs,
            h.var,
            pred,
            p["gate_mode"],
            p["var_thresh"],
            k_for_topk=9,
            win=p["win"],
            action=p["gate_action"],
            fallback=p["fallback"],
            fallback_thresh=p["fallback_thresh"],
            fallback_k=p["fallback_k"],
            fallback_min_thresh=p["fallback_min_thresh"],
        )

    def samp_argmax(rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
        out = []
        seen = set()
        while len(out) < n:
            params = {
                "min_thresh": float(rng.choice(min_threshs)),
                "beta": float(rng.choice(betas)),
                "gate_mode": str(rng.choice(gate_modes)),
                "var_thresh": float(rng.choice(var_threshs)),
                "win": int(rng.choice(wins)),
                "gate_action": str(rng.choice(gate_actions)),
                "fallback": str(rng.choice(fallbacks)),
                "fallback_thresh": float(rng.choice(threshs)),
                "fallback_k": int(rng.choice(ks)),
                "fallback_min_thresh": float(rng.choice(min_threshs)),
            }
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(params)
        return out

    grids["argmax_unc"] = (samp_argmax, fn_argmax)

    # First threshold
    def fn_first(h: Hole, p: Dict[str, Any]) -> Optional[int]:
        pm = apply_risk_adjust(h.probs, h.var, p["beta"])
        pred = decide_first_thresh(pm, p["thresh"])
        return apply_gate(
            h.probs,
            h.var,
            pred,
            p["gate_mode"],
            p["var_thresh"],
            k_for_topk=9,
            win=p["win"],
            action=p["gate_action"],
            fallback=p["fallback"],
            fallback_thresh=p["fallback_thresh"],
            fallback_k=p["fallback_k"],
            fallback_min_thresh=p["fallback_min_thresh"],
        )

    def samp_first(rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
        out, seen = [], set()
        while len(out) < n:
            params = {
                "thresh": float(rng.choice(threshs)),
                "beta": float(rng.choice(betas)),
                "gate_mode": str(rng.choice(gate_modes)),
                "var_thresh": float(rng.choice(var_threshs)),
                "win": int(rng.choice(wins)),
                "gate_action": str(rng.choice(gate_actions)),
                "fallback": str(rng.choice(fallbacks)),
                "fallback_thresh": float(rng.choice(threshs)),
                "fallback_k": int(rng.choice(ks)),
                "fallback_min_thresh": float(rng.choice(min_threshs)),
            }
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(params)
        return out

    grids["first_thresh_unc"] = (samp_first, fn_first)

    # Smooth first
    def fn_smooth_first(h: Hole, p: Dict[str, Any]) -> Optional[int]:
        pm = apply_risk_adjust(h.probs, h.var, p["beta"])
        pred = decide_smooth_first(pm, p["window"], p["thresh"])
        return apply_gate(
            h.probs,
            h.var,
            pred,
            p["gate_mode"],
            p["var_thresh"],
            k_for_topk=9,
            win=p["win"],
            action=p["gate_action"],
            fallback=p["fallback"],
            fallback_thresh=p["fallback_thresh"],
            fallback_k=p["fallback_k"],
            fallback_min_thresh=p["fallback_min_thresh"],
        )

    def samp_smooth_first(rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
        out, seen = [], set()
        while len(out) < n:
            params = {
                "window": int(rng.choice(smooth_ws)),
                "thresh": float(rng.choice(threshs)),
                "beta": float(rng.choice(betas)),
                "gate_mode": str(rng.choice(gate_modes)),
                "var_thresh": float(rng.choice(var_threshs)),
                "win": int(rng.choice(wins)),
                "gate_action": str(rng.choice(gate_actions)),
                "fallback": str(rng.choice(fallbacks)),
                "fallback_thresh": float(rng.choice(threshs)),
                "fallback_k": int(rng.choice(ks)),
                "fallback_min_thresh": float(rng.choice(min_threshs)),
            }
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(params)
        return out

    grids["smooth_first_unc"] = (samp_smooth_first, fn_smooth_first)

    # Centroid
    def fn_centroid(h: Hole, p: Dict[str, Any]) -> Optional[int]:
        pm = apply_risk_adjust(h.probs, h.var, p["beta"])
        pred = decide_centroid(pm, p["thresh"])
        return apply_gate(
            h.probs,
            h.var,
            pred,
            p["gate_mode"],
            p["var_thresh"],
            k_for_topk=9,
            win=p["win"],
            action=p["gate_action"],
            fallback=p["fallback"],
            fallback_thresh=p["fallback_thresh"],
            fallback_k=p["fallback_k"],
            fallback_min_thresh=p["fallback_min_thresh"],
        )

    def samp_centroid(rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
        out, seen = [], set()
        while len(out) < n:
            params = {
                "thresh": float(rng.choice(threshs)),
                "beta": float(rng.choice(betas)),
                "gate_mode": str(rng.choice(gate_modes)),
                "var_thresh": float(rng.choice(var_threshs)),
                "win": int(rng.choice(wins)),
                "gate_action": str(rng.choice(gate_actions)),
                "fallback": str(rng.choice(fallbacks)),
                "fallback_thresh": float(rng.choice(threshs)),
                "fallback_k": int(rng.choice(ks)),
                "fallback_min_thresh": float(rng.choice(min_threshs)),
            }
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(params)
        return out

    grids["centroid_unc"] = (samp_centroid, fn_centroid)

    # TopKMedian
    def fn_topk(h: Hole, p: Dict[str, Any]) -> Optional[int]:
        pm = apply_risk_adjust(h.probs, h.var, p["beta"])
        pred = decide_topkmedian(pm, p["k"], p["min_thresh"])
        return apply_gate(
            h.probs,
            h.var,
            pred,
            p["gate_mode"],
            p["var_thresh"],
            k_for_topk=p["k"],
            win=p["win"],
            action=p["gate_action"],
            fallback=p["fallback"],
            fallback_thresh=p["fallback_thresh"],
            fallback_k=p["fallback_k"],
            fallback_min_thresh=p["fallback_min_thresh"],
        )

    def samp_topk(rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
        out, seen = [], set()
        while len(out) < n:
            params = {
                "k": int(rng.choice(ks)),
                "min_thresh": float(rng.choice(min_threshs)),
                "beta": float(rng.choice(betas)),
                "gate_mode": str(rng.choice(gate_modes)),
                "var_thresh": float(rng.choice(var_threshs)),
                "win": int(rng.choice(wins)),
                "gate_action": str(rng.choice(gate_actions)),
                "fallback": str(rng.choice(fallbacks)),
                "fallback_thresh": float(rng.choice(threshs)),
                "fallback_k": int(rng.choice(ks)),
                "fallback_min_thresh": float(rng.choice(min_threshs)),
            }
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(params)
        return out

    grids["topkmedian_unc"] = (samp_topk, fn_topk)

    # Two-stage
    region_threshs = np.round(np.arange(0.20, 0.86, 0.05), 2).tolist()
    peak_threshs = np.round(np.arange(0.30, 0.96, 0.05), 2).tolist()
    min_lens = [2, 3, 4, 5, 6, 8, 10]

    def fn_two(h: Hole, p: Dict[str, Any]) -> Optional[int]:
        pm = apply_risk_adjust(h.probs, h.var, p["beta"])
        pred = decide_two_stage(pm, p["region_thresh"], p["min_len"], p["peak_thresh"])
        return apply_gate(
            h.probs,
            h.var,
            pred,
            p["gate_mode"],
            p["var_thresh"],
            k_for_topk=9,
            win=p["win"],
            action=p["gate_action"],
            fallback=p["fallback"],
            fallback_thresh=p["fallback_thresh"],
            fallback_k=p["fallback_k"],
            fallback_min_thresh=p["fallback_min_thresh"],
        )

    def samp_two(rng: np.random.RandomState, n: int) -> List[Dict[str, Any]]:
        out, seen = [], set()
        while len(out) < n:
            params = {
                "region_thresh": float(rng.choice(region_threshs)),
                "min_len": int(rng.choice(min_lens)),
                "peak_thresh": float(rng.choice(peak_threshs)),
                "beta": float(rng.choice(betas)),
                "gate_mode": str(rng.choice(gate_modes)),
                "var_thresh": float(rng.choice(var_threshs)),
                "win": int(rng.choice(wins)),
                "gate_action": str(rng.choice(gate_actions)),
                "fallback": str(rng.choice(fallbacks)),
                "fallback_thresh": float(rng.choice(threshs)),
                "fallback_k": int(rng.choice(ks)),
                "fallback_min_thresh": float(rng.choice(min_threshs)),
            }
            key = tuple(sorted(params.items()))
            if key in seen:
                continue
            seen.add(key)
            out.append(params)
        return out

    grids["two_stage_unc"] = (samp_two, fn_two)

    return grids


def eval_method(holes: Sequence[Hole], params: Dict[str, Any], decide: DecisionFn) -> Dict[str, Any]:
    preds: List[Optional[int]] = []
    for h in holes:
        preds.append(decide(h, params))
    m = metrics_penetrated(holes, preds)
    return m


def weighted_merge(train_m: Dict[str, Any], test_m: Dict[str, Any]) -> Dict[str, Any]:
    parts = []
    for m in (train_m, test_m):
        if m and m.get("n_penetrated", 0) > 0:
            parts.append((int(m["n_penetrated"]), m))
    if not parts:
        return {"n_penetrated": 0, "pct_within_3": 0.0, "pct_within_5": 0.0, "pct_over_10": 0.0}
    if len(parts) == 1:
        m = parts[0][1]
        return {k: m[k] for k in ("n_penetrated", "pct_within_3", "pct_within_5", "pct_over_10")}
    n_total = sum(n for n, _ in parts)
    return {
        "n_penetrated": n_total,
        "pct_within_3": sum(m["pct_within_3"] * n for n, m in parts) / n_total,
        "pct_within_5": sum(m["pct_within_5"] * n for n, m in parts) / n_total,
        "pct_over_10": sum(m["pct_over_10"] * n for n, m in parts) / n_total,
    }


def main():
    ap = argparse.ArgumentParser(description="Grid-search decision rules on inference_results JSONs")
    ap.add_argument("--train_json", type=str, default="inference_results_train.json")
    ap.add_argument("--test_json", type=str, default="inference_results_test.json")
    ap.add_argument("--out_dir", type=str, default="decision_grid")
    ap.add_argument("--evals_per_method", type=int, default=6000, help="how many parameter combos to evaluate per method")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--focus_penetrated_only", action="store_true", help="speed: only keep true_label=1 holes (metrics scope anyway)")
    ap.add_argument("--lambda_over10", type=float, default=0.15, help="objective penalty weight for >10%% (combined).")
    ap.add_argument("--lambda_within3", type=float, default=0.05, help="objective bonus weight for <=3%% (combined).")
    args = ap.parse_args()

    train_rows = _load_list(Path(args.train_json))
    test_rows = _load_list(Path(args.test_json))
    train_holes = parse_holes(train_rows)
    test_holes = parse_holes(test_rows)
    if args.focus_penetrated_only:
        train_holes = [h for h in train_holes if h.true_label == 1]
        test_holes = [h for h in test_holes if h.true_label == 1]

    methods = build_methods()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results_train = []
    results_test = []
    results_comb = []

    rng = np.random.RandomState(int(args.seed))

    for name, (sampler, decide) in methods.items():
        best_train = (None, None)  # (score, payload)
        best_test = (None, None)
        best_comb = (None, None)

        params_list = sampler(rng, int(args.evals_per_method))
        n_eval = 0
        for params in params_list:
            n_eval += 1
            m_tr = eval_method(train_holes, params, decide)
            m_te = eval_method(test_holes, params, decide)
            m_cb = weighted_merge(m_tr, m_te)

            # optimize for combined objective that is more "hard" on extreme errors:
            #   score = <=5  + a*<=3  - b*>10
            # tie-breakers: higher <=5, lower >10, higher <=3
            a = float(getattr(args, "lambda_within3", 0.0))
            b = float(getattr(args, "lambda_over10", 0.0))

            def score(m):
                s = float(m["pct_within_5"]) + a * float(m["pct_within_3"]) - b * float(m["pct_over_10"])
                return (s, float(m["pct_within_5"]), -float(m["pct_over_10"]), float(m["pct_within_3"]))

            sc_tr = score(m_tr)
            sc_te = score(m_te)
            sc_cb = score(m_cb)

            if best_train[0] is None or sc_tr > best_train[0]:
                best_train = (sc_tr, {"method": name, "best_params": params, "metrics": m_tr, "evals": n_eval})
            if best_test[0] is None or sc_te > best_test[0]:
                best_test = (sc_te, {"method": name, "best_params": params, "metrics": m_te, "evals": n_eval})
            if best_comb[0] is None or sc_cb > best_comb[0]:
                best_comb = (sc_cb, {"method": name, "best_params": params, "metrics": m_cb, "evals": n_eval})

        results_train.append(best_train[1])
        results_test.append(best_test[1])
        results_comb.append(best_comb[1])
        print(f"[{name}] evals={n_eval}  best_comb<=5={best_comb[1]['metrics']['pct_within_5']:.2f}%  params={best_comb[1]['best_params']}")

    # sort by combined <=5
    results_train = sorted(results_train, key=lambda r: r["metrics"]["pct_within_5"], reverse=True)
    results_test = sorted(results_test, key=lambda r: r["metrics"]["pct_within_5"], reverse=True)
    results_comb = sorted(results_comb, key=lambda r: r["metrics"]["pct_within_5"], reverse=True)

    (out_dir / "decision_grid_results_train.json").write_text(json.dumps(results_train, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "decision_grid_results_test.json").write_text(json.dumps(results_test, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "decision_grid_results_combined.json").write_text(json.dumps(results_comb, ensure_ascii=False, indent=2), encoding="utf-8")

    # markdown summary
    def md_table(lst: List[dict], title: str) -> str:
        lines = [f"## {title}", "", "| method | n_pen | ≤3% | ≤5% | >10% | params | evals |", "|---|---:|---:|---:|---:|---|---:|"]
        for r in lst:
            m = r["metrics"]
            params = json.dumps(r["best_params"], ensure_ascii=False)
            if len(params) > 72:
                params = params[:69] + "..."
            lines.append(f"| {r['method']} | {m['n_penetrated']} | {m['pct_within_3']:.1f} | {m['pct_within_5']:.1f} | {m['pct_over_10']:.1f} | `{params}` | {r.get('evals',0)} |")
        return "\n".join(lines)

    md = []
    md.append("# Decision grid-search (offline)")
    md.append("")
    md.append("Using `inference_results_train.json` and `inference_results_test.json` probability curves.")
    md.append("All metrics are computed on penetrated holes only (true_label=1).")
    md.append("")
    md.append(md_table(results_train, "Best on train"))
    md.append("")
    md.append(md_table(results_test, "Best on test"))
    md.append("")
    md.append(md_table(results_comb, "Best on combined (weighted by n_penetrated)"))
    md.append("")
    best = results_comb[0] if results_comb else None
    if best:
        md.append(f"**Best combined (≤5%)**: `{best['method']}`  ≤5={best['metrics']['pct_within_5']:.2f}%  params={best['best_params']}")
        md.append("")
    (out_dir / "decision_grid_results.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote results to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

