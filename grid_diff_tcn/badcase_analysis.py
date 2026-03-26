#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Badcase analysis & visualization for Grid-Diff penetration localization.

Input: inference_results_*.json produced by inference.py
Output:
  - scatter: true_idx vs pred_idx (penetrated holes only)
  - error histogram / CDF
  - badcase table (csv + json)
  - probability curves for selected badcases (with true/pred markers)
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_name(s: str, max_len: int = 120) -> str:
    s = (s or "").strip().replace(os.sep, "_").replace("/", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)
    return s[:max_len] if len(s) > max_len else s


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _as_int(x: Any) -> Optional[int]:
    if x is None:
        return None
    try:
        # handles numpy scalars too
        return int(x)
    except Exception:
        try:
            return int(float(x))
        except Exception:
            return None


@dataclass
class HoleRecord:
    split: str
    index: int
    sample_path: str
    true_label: int
    true_layer: Optional[int]
    true_idx: Optional[int]
    pred_penetrated: bool
    pred_layer: Optional[int]
    pred_idx: Optional[int]
    probs: Optional[np.ndarray]
    probs_mean: Optional[np.ndarray]
    probs_var: Optional[np.ndarray]
    # Optional override prediction (e.g., from offline decision rule)
    pred_idx_override: Optional[int] = None
    pred_penetrated_override: Optional[bool] = None

    @property
    def basename(self) -> str:
        return os.path.basename(self.sample_path.rstrip("/")) if self.sample_path else f"idx_{self.index}"

    def error(self) -> int:
        if self.true_label != 1:
            return 0
        pred = self.pred_idx_override if self.pred_idx_override is not None else self.pred_idx
        if self.true_idx is None or pred is None:
            return 999
        return abs(int(pred) - int(self.true_idx))

    def pred_idx_effective(self) -> Optional[int]:
        return self.pred_idx_override if self.pred_idx_override is not None else self.pred_idx

    def pred_penetrated_effective(self) -> bool:
        if self.pred_penetrated_override is not None:
            return bool(self.pred_penetrated_override)
        return bool(self.pred_penetrated)


def parse_results(path: Path, split: str) -> List[HoleRecord]:
    raw = _load_json(path)
    if not isinstance(raw, list):
        raise ValueError(f"{path} is not a list json")
    out: List[HoleRecord] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        probs = r.get("probs")
        probs_arr = None
        if isinstance(probs, list) and len(probs) > 0:
            probs_arr = np.asarray(probs, dtype=np.float32)
        probs_mean = r.get("probs_mean")
        probs_mean_arr = None
        if isinstance(probs_mean, list) and len(probs_mean) > 0:
            probs_mean_arr = np.asarray(probs_mean, dtype=np.float32)
        probs_var = r.get("probs_var")
        probs_var_arr = None
        if isinstance(probs_var, list) and len(probs_var) > 0:
            probs_var_arr = np.asarray(probs_var, dtype=np.float32)
        out.append(
            HoleRecord(
                split=split,
                index=_as_int(r.get("index")) or 0,
                sample_path=str(r.get("sample_path") or ""),
                true_label=int(r.get("true_label") or 0),
                true_layer=_as_int(r.get("true_penetration_layer")),
                true_idx=_as_int(r.get("true_penetration_index")),
                pred_penetrated=bool(r.get("pred_penetrated")),
                pred_layer=_as_int(r.get("pred_penetration_layer")),
                pred_idx=_as_int(r.get("pred_penetration_index")),
                probs=probs_arr,
                probs_mean=probs_mean_arr,
                probs_var=probs_var_arr,
            )
        )
    return out


def _decide_topkmedian_unc(
    probs: np.ndarray,
    var: Optional[np.ndarray],
    k: int,
    min_thresh: float,
    beta: float,
    gate_mode: str,
    var_thresh: float,
    win: int,
    gate_action: str,
    fallback: str,
    fallback_thresh: float,
    fallback_k: int,
    fallback_min_thresh: float,
) -> Optional[int]:
    """
    Offline decision rule consistent with decision_grid_search.py for method 'topkmedian_unc'.
    Uses:
      - risk-adjust score: s = p - beta*sqrt(var) (if var available)
      - TopKMedian decision on s
      - optional gate based on var (pred/win/topk); if triggered:
          * action='veto' -> None
          * action='fallback' -> fallback rule on mean probs p
    """
    p = np.asarray(probs, dtype=np.float32).ravel()
    if p.size == 0:
        return None
    v = None
    if var is not None and isinstance(var, np.ndarray) and var.size == p.size:
        v = np.asarray(var, dtype=np.float32).ravel()
    # risk adjust
    if v is not None and beta and float(beta) > 0:
        s = p - float(beta) * np.sqrt(np.maximum(v, 0.0))
    else:
        s = p
    T = int(s.size)
    kk = max(1, min(int(k), T))
    topk_idx = np.argpartition(-s, kk - 1)[:kk]
    if float(np.median(s[topk_idx])) < float(min_thresh):
        pred = None
    else:
        pred = int(np.sort(topk_idx)[len(topk_idx) // 2])

    # gate
    vt = float(var_thresh)
    if pred is None or v is None or vt <= 0:
        return pred

    def gate_triggered() -> bool:
        gm = str(gate_mode)
        if gm == "pred":
            return float(v[pred]) > vt
        if gm == "win":
            w = max(1, int(win))
            a = max(0, pred - w)
            b = min(T, pred + w + 1)
            return float(np.median(v[a:b])) > vt
        # topk by mean probs
        topk_m = np.argpartition(-p, kk - 1)[:kk]
        return float(np.median(v[topk_m])) > vt

    if not gate_triggered():
        return pred

    act = str(gate_action)
    if act == "veto":
        return None

    # fallback on mean probs p
    fb = str(fallback)
    th = float(fallback_thresh)
    if fb == "argmax":
        i = int(np.argmax(p))
        return i if float(p[i]) >= th else None
    if fb == "first":
        idx = np.where(p >= th)[0]
        return int(idx[0]) if idx.size else None
    if fb == "smooth_first":
        w = max(1, int(win))
        if w <= 1:
            idx = np.where(p >= th)[0]
            return int(idx[0]) if idx.size else None
        kernel = np.ones(w, dtype=np.float32) / float(w)
        sm = np.convolve(p, kernel, mode="same")
        idx = np.where(sm >= th)[0]
        return int(idx[0]) if idx.size else None
    if fb == "earliest_topk":
        kk2 = max(1, min(int(fallback_k), T))
        topk2 = np.argpartition(-p, kk2 - 1)[:kk2]
        if float(np.median(p[topk2])) < float(fallback_min_thresh):
            return None
        return int(np.min(topk2))
    # topkmedian fallback
    kk2 = max(1, min(int(fallback_k), T))
    topk2 = np.argpartition(-p, kk2 - 1)[:kk2]
    if float(np.median(p[topk2])) < float(fallback_min_thresh):
        return None
    return int(np.sort(topk2)[len(topk2) // 2])


def apply_decision_override(recs: List[HoleRecord], decision_json: Path) -> Dict[str, Any]:
    """
    Read decision_grid_search output and override pred_idx for analysis/plots.
    Currently supports method: topkmedian_unc (best_params).
    Returns the loaded (method, best_params, metrics).
    """
    obj = _load_json(decision_json)
    # allow either a list (take first) or a dict with best_params
    if isinstance(obj, list) and obj:
        best = obj[0]
    elif isinstance(obj, dict):
        best = obj
    else:
        raise ValueError(f"Unsupported decision json: {decision_json}")

    method = best.get("method")
    params = best.get("best_params") or {}
    if method != "topkmedian_unc":
        raise ValueError(f"Only supports topkmedian_unc for override, got: {method}")

    for r in recs:
        if r.probs is None or len(r.probs) == 0:
            r.pred_idx_override = None
            r.pred_penetrated_override = False
            continue
        pred = _decide_topkmedian_unc(
            r.probs,
            r.probs_var,
            k=int(params.get("k", 7)),
            min_thresh=float(params.get("min_thresh", 0.3)),
            beta=float(params.get("beta", 0.0)),
            gate_mode=str(params.get("gate_mode", "topk")),
            var_thresh=float(params.get("var_thresh", 0.0)),
            win=int(params.get("win", 7)),
            gate_action=str(params.get("gate_action", "veto")),
            fallback=str(params.get("fallback", "first")),
            fallback_thresh=float(params.get("fallback_thresh", 0.75)),
            fallback_k=int(params.get("fallback_k", 7)),
            fallback_min_thresh=float(params.get("fallback_min_thresh", 0.35)),
        )
        r.pred_idx_override = pred
        r.pred_penetrated_override = (pred is not None)

    return {"method": method, "best_params": params, "metrics": best.get("metrics"), "source": str(decision_json)}


def compute_metrics_penetrated(recs: Sequence[HoleRecord]) -> Dict[str, Any]:
    pen = [r for r in recs if r.true_label == 1]
    n = len(pen)
    within_3 = 0
    within_5 = 0
    over_10 = 0
    missing = 0
    for r in pen:
        e = r.error()
        if e >= 999:
            missing += 1
            over_10 += 1
            continue
        if e <= 3:
            within_3 += 1
        if e <= 5:
            within_5 += 1
        if e > 10:
            over_10 += 1
    return {
        "n_penetrated": n,
        "pct_within_3": (within_3 / n * 100.0) if n else 0.0,
        "pct_within_5": (within_5 / n * 100.0) if n else 0.0,
        "pct_over_10": (over_10 / n * 100.0) if n else 0.0,
        "n_missing_pred_or_true": missing,
        "n_within_3": within_3,
        "n_within_5": within_5,
        "n_over_10": over_10,
    }


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: E402

    return plt


def plot_scatter(recs: Sequence[HoleRecord], out_png: Path, title: str) -> None:
    plt = _matplotlib()
    pen = [r for r in recs if r.true_label == 1 and r.true_idx is not None and r.pred_idx_effective() is not None]
    if not pen:
        return
    xs = np.array([r.true_idx for r in pen], dtype=np.int32)
    ys = np.array([r.pred_idx_effective() for r in pen], dtype=np.int32)
    err = np.abs(xs - ys)

    fig = plt.figure(figsize=(7.2, 6.6), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    sc = ax.scatter(xs, ys, c=err, s=10, cmap="viridis", alpha=0.85, linewidths=0)
    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("|pred-true| (layers)")
    mn = int(min(xs.min(), ys.min()))
    mx = int(max(xs.max(), ys.max()))
    ax.plot([mn, mx], [mn, mx], linestyle="--", color="black", linewidth=1, alpha=0.6, label="y=x")
    ax.set_xlabel("True penetration index")
    ax.set_ylabel("Predicted penetration index")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_error_hist(recs: Sequence[HoleRecord], out_png: Path, title: str) -> None:
    plt = _matplotlib()
    pen = [r for r in recs if r.true_label == 1]
    if not pen:
        return
    errs = np.array([min(r.error(), 200) for r in pen], dtype=np.int32)  # cap for visualization
    # mark 999 as 200+
    errs_vis = np.where(errs >= 999, 200, errs)

    fig = plt.figure(figsize=(7.6, 4.6), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    bins = list(range(0, 51)) + [60, 80, 100, 150, 200]
    ax.hist(errs_vis, bins=bins, color="#2563eb", alpha=0.85)
    ax.axvline(3, color="#16a34a", linestyle="--", linewidth=1.2, label="<=3")
    ax.axvline(5, color="#f59e0b", linestyle="--", linewidth=1.2, label="<=5")
    ax.axvline(10, color="#ef4444", linestyle="--", linewidth=1.2, label=">10 threshold")
    ax.set_xlabel("Absolute error in layers (capped; missing shown as 200)")
    ax.set_ylabel("Count (penetrated holes only)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_error_cdf(recs: Sequence[HoleRecord], out_png: Path, title: str) -> None:
    plt = _matplotlib()
    pen = [r for r in recs if r.true_label == 1]
    if not pen:
        return
    errs = np.array([r.error() for r in pen], dtype=np.int32)
    # treat missing as very large
    errs = np.where(errs >= 999, 10_000, errs)
    errs_sorted = np.sort(errs)
    ys = (np.arange(len(errs_sorted)) + 1) / len(errs_sorted)

    fig = plt.figure(figsize=(7.6, 4.6), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(errs_sorted, ys, color="#0f172a", linewidth=1.6)
    for thr, col in [(3, "#16a34a"), (5, "#f59e0b"), (10, "#ef4444")]:
        ax.axvline(thr, color=col, linestyle="--", linewidth=1.2)
    ax.set_xlim(left=0, right=min(60, max(12, int(np.percentile(errs_sorted, 95)))))
    ax.set_xlabel("Absolute error (layers)")
    ax.set_ylabel("CDF (penetrated holes only)")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def plot_prob_curve(r: HoleRecord, out_png: Path, title: str, min_thresh_ref: Optional[float] = None) -> None:
    """
    Plot probability curve(s).

    Display logic (aligned with current decision usage):
      - If BOTH probs_mean + probs_var exist: treat used=probs_mean, and do NOT plot probs.
      - Else: fallback used=probs.
    Also plots probs_var (secondary y-axis) when available.
    """
    if (r.probs is None or len(r.probs) == 0) and (r.probs_mean is None or len(r.probs_mean) == 0):
        return
    plt = _matplotlib()
    p = np.asarray(r.probs, dtype=np.float32) if r.probs is not None else None
    pm = np.asarray(r.probs_mean, dtype=np.float32) if r.probs_mean is not None else None
    pv = np.asarray(r.probs_var, dtype=np.float32) if r.probs_var is not None else None
    use_mean = (pm is not None and len(pm) > 0) and (pv is not None and len(pv) > 0)
    used = pm if use_mean else p
    if used is None or len(used) == 0:
        return
    T = len(used)
    x = np.arange(T)

    fig = plt.figure(figsize=(10.8, 3.6), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    if use_mean:
        ax.plot(x, pm, color="#0f172a", linewidth=1.3, label="prob_mean (used)")
    else:
        ax.plot(x, p, color="#2563eb", linewidth=1.3, label="prob (used)")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("P(penetration)")
    ax.grid(True, alpha=0.25)

    if r.true_idx is not None:
        ax.axvline(r.true_idx, color="#16a34a", linestyle="--", linewidth=1.4, label=f"true_idx={r.true_idx}")
    pred_eff = r.pred_idx_effective()
    if pred_eff is not None:
        ax.axvline(pred_eff, color="#ef4444", linestyle="--", linewidth=1.4, label=f"pred_idx={pred_eff}")
    if min_thresh_ref is not None:
        ax.axhline(
            float(min_thresh_ref),
            color="#f59e0b",
            linestyle=":",
            linewidth=1.2,
            alpha=0.9,
            label=f"min_thresh={float(min_thresh_ref):.2f} (ref)",
        )
    ax.set_title(title)

    if pv is not None and len(pv) == T:
        ax2 = ax.twinx()
        ax2.plot(x, pv, color="#a855f7", linewidth=1.0, alpha=0.75, label="prob_var")
        ax2.set_ylabel("Var")
        # legend merge
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    else:
        ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_png)
    plt.close(fig)


def write_badcase_tables(recs: Sequence[HoleRecord], out_dir: Path, top_n: int = 50) -> List[Dict[str, Any]]:
    pen = [r for r in recs if r.true_label == 1]
    rows = []
    for r in pen:
        rows.append(
            {
                "split": r.split,
                "index": r.index,
                "sample_basename": r.basename,
                "sample_path": r.sample_path,
                "true_layer": r.true_layer,
                "true_idx": r.true_idx,
                "pred_layer": r.pred_layer,
                "pred_idx": r.pred_idx_effective(),
                "pred_penetrated": r.pred_penetrated_effective(),
                "error": r.error(),
                "T": int(len(r.probs)) if r.probs is not None else None,
                "max_prob": float(np.max(r.probs)) if r.probs is not None and len(r.probs) else None,
            }
        )
    rows_sorted = sorted(rows, key=lambda x: (-1 if x["error"] == 999 else -x["error"], str(x["sample_basename"])))
    top = rows_sorted[: max(1, int(top_n))]

    (out_dir / "badcases_top.json").write_text(json.dumps(top, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV
    import csv

    with (out_dir / "badcases_top.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(top[0].keys()))
        w.writeheader()
        for row in top:
            w.writerow(row)

    return top


def main():
    ap = argparse.ArgumentParser(description="Badcase analysis for inference_results_*.json")
    ap.add_argument("--train_json", type=str, default="inference_results_train.json")
    ap.add_argument("--test_json", type=str, default="inference_results_test.json")
    ap.add_argument("--out_dir", type=str, default="badcase_report")
    ap.add_argument("--top_n_bad", type=int, default=50, help="export top-N badcases by error")
    ap.add_argument("--max_curves", type=int, default=30, help="plot at most N probability curves")
    ap.add_argument("--curve_mode", type=str, default="worst", choices=["worst", "random", "missing"], help="which cases to plot")
    ap.add_argument("--decision_best_json", type=str, default=None, help="Optional: decision_grid_results_combined.json (use first/best entry to override pred_idx)")
    ap.add_argument("--max_good_curves", type=int, default=20, help="plot at most N good-case curves")
    ap.add_argument("--good_max_error", type=int, default=3, help="good-case criterion: error <= this value (penetrated holes)")
    args = ap.parse_args()

    train_path = Path(args.train_json)
    test_path = Path(args.test_json)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)
    _ensure_dir(out_dir / "curves")
    _ensure_dir(out_dir / "good_curves")

    recs: List[HoleRecord] = []
    if train_path.is_file():
        recs.extend(parse_results(train_path, split="train"))
    if test_path.is_file():
        recs.extend(parse_results(test_path, split="test"))
    if not recs:
        raise SystemExit("No input json found.")

    decision_info = None
    min_thresh_ref = None
    if args.decision_best_json:
        decision_info = apply_decision_override(recs, Path(args.decision_best_json))
        try:
            min_thresh_ref = float((decision_info.get("best_params") or {}).get("min_thresh"))
        except Exception:
            min_thresh_ref = None

    m_all = compute_metrics_penetrated(recs)
    m_train = compute_metrics_penetrated([r for r in recs if r.split == "train"])
    m_test = compute_metrics_penetrated([r for r in recs if r.split == "test"])

    summary = {
        "train": m_train,
        "test": m_test,
        "combined": m_all,
        "decision_override": decision_info,
        "notes": {
            "metric_scope": "penetrated holes only (true_label=1)",
            "error_definition": "abs(pred_idx-true_idx) if both exist else 999",
        },
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    plot_scatter(
        [r for r in recs if r.split == "train"],
        out_dir / "scatter_train.png",
        "Train: true vs pred (penetrated holes)",
    )
    plot_scatter(
        [r for r in recs if r.split == "test"],
        out_dir / "scatter_test.png",
        "Test: true vs pred (penetrated holes)",
    )
    plot_scatter(recs, out_dir / "scatter_combined.png", "Combined: true vs pred (penetrated holes)")

    plot_error_hist([r for r in recs if r.split == "train"], out_dir / "err_hist_train.png", "Train: error histogram")
    plot_error_hist([r for r in recs if r.split == "test"], out_dir / "err_hist_test.png", "Test: error histogram")
    plot_error_hist(recs, out_dir / "err_hist_combined.png", "Combined: error histogram")

    plot_error_cdf([r for r in recs if r.split == "train"], out_dir / "err_cdf_train.png", "Train: error CDF")
    plot_error_cdf([r for r in recs if r.split == "test"], out_dir / "err_cdf_test.png", "Test: error CDF")
    plot_error_cdf(recs, out_dir / "err_cdf_combined.png", "Combined: error CDF")

    top = write_badcase_tables(recs, out_dir, top_n=args.top_n_bad)

    # select curves
    pen = [r for r in recs if r.true_label == 1]
    if args.curve_mode == "missing":
        cand = [r for r in pen if r.pred_idx is None or r.true_idx is None]
        cand = sorted(cand, key=lambda r: r.error(), reverse=True)
    elif args.curve_mode == "random":
        rng = np.random.RandomState(42)
        cand = [pen[i] for i in rng.permutation(len(pen))[: min(len(pen), args.max_curves * 2)]]
    else:
        cand = sorted(pen, key=lambda r: r.error(), reverse=True)

    n_plot = 0
    for r in cand:
        if n_plot >= int(args.max_curves):
            break
        if (r.probs is None or len(r.probs) == 0) and (r.probs_mean is None or len(r.probs_mean) == 0):
            continue
        title = f"[{r.split}] err={r.error()}  {r.basename}"
        fname = f"{n_plot:02d}_{_safe_name(r.split)}_{_safe_name(r.basename)}_err{r.error()}.png"
        plot_prob_curve(r, out_dir / "curves" / fname, title=title, min_thresh_ref=min_thresh_ref)
        n_plot += 1

    # good cases (combined, not split)
    good = [
        r
        for r in pen
        if r.error() <= int(args.good_max_error) and r.true_idx is not None and r.pred_idx_effective() is not None
    ]
    # prefer diverse basenames
    good = sorted(good, key=lambda r: (r.error(), r.split, r.basename))
    n_good = 0
    used = set()
    for r in good:
        if n_good >= int(args.max_good_curves):
            break
        key = (r.basename, r.true_idx, r.pred_idx_effective())
        if key in used:
            continue
        used.add(key)
        title = f"[good] err={r.error()}  {r.basename}"
        fname = f"{n_good:02d}_{_safe_name(r.basename)}_err{r.error()}.png"
        plot_prob_curve(r, out_dir / "good_curves" / fname, title=title, min_thresh_ref=min_thresh_ref)
        n_good += 1

    # write a short README for the report dir
    md = []
    md.append("# Badcase report")
    md.append("")
    md.append("## Summary (penetrated holes only)")
    md.append("")
    md.append("```json")
    md.append(json.dumps(summary, ensure_ascii=False, indent=2))
    md.append("```")
    md.append("")
    md.append("## Files")
    md.append("- `scatter_*.png`: true vs pred scatter plots")
    md.append("- `err_hist_*.png`: error histograms")
    md.append("- `err_cdf_*.png`: error CDF plots")
    md.append("- `badcases_top.csv` / `badcases_top.json`: worst cases table")
    md.append("- `curves/*.png`: probability curves for selected cases (true/pred markers)")
    md.append(f"- `good_curves/*.png`: good-case curves (error <= {int(args.good_max_error)}) with prob/mean/var if available")
    (out_dir / "README.md").write_text("\n".join(md), encoding="utf-8")

    print(f"Wrote report to: {out_dir.resolve()}")
    print("Combined metrics (penetrated holes):")
    print(json.dumps(m_all, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

