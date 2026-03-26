#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create side-by-side image comparisons around true/pred penetration indices for badcases.

Input:
  - badcases_top.json produced by badcase_analysis.py
Output:
  - montage PNGs under out_dir

This tool is intentionally "filesystem-driven": it reads raw images from sample_path folders.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


_IDX_RE = re.compile(r"_(\d+)_\d+\.(jpg|jpeg|png)$", re.IGNORECASE)


def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _safe_name(s: str, max_len: int = 140) -> str:
    s = (s or "").strip().replace(os.sep, "_").replace("/", "_")
    s = "".join(ch if ch.isalnum() or ch in "._-+" else "_" for ch in s)
    return s[:max_len] if len(s) > max_len else s


def _parse_frame_idx(name: str) -> Optional[int]:
    m = _IDX_RE.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _list_images_by_idx(sample_path: Path) -> List[Tuple[int, Path]]:
    if not sample_path.exists():
        return []
    imgs: List[Tuple[int, Path]] = []
    for p in sample_path.iterdir():
        if not p.is_file():
            continue
        if p.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue
        idx = _parse_frame_idx(p.name)
        if idx is None:
            continue
        imgs.append((idx, p))
    imgs.sort(key=lambda x: x[0])
    return imgs


@dataclass
class Case:
    split: str
    index: int
    sample_basename: str
    sample_path: Path
    true_idx: Optional[int]
    pred_idx: Optional[int]
    error: int


def _read_cases(badcases_json: Path) -> List[Case]:
    raw = _load_json(badcases_json)
    if not isinstance(raw, list):
        raise ValueError(f"{badcases_json} must be a list")
    out: List[Case] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        out.append(
            Case(
                split=str(r.get("split") or ""),
                index=int(r.get("index") or 0),
                sample_basename=str(r.get("sample_basename") or ""),
                sample_path=Path(str(r.get("sample_path") or "")),
                true_idx=(int(r["true_idx"]) if r.get("true_idx") is not None else None),
                pred_idx=(int(r["pred_idx"]) if r.get("pred_idx") is not None else None),
                error=int(r.get("error") or 0),
            )
        )
    return out


def _matplotlib():
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt  # noqa: E402

    return plt


def _imshow(ax, img_path: Optional[Path], title: str, border_color: Optional[str] = None) -> None:
    plt = _matplotlib()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    if img_path is None or not img_path.exists():
        ax.text(0.5, 0.5, "MISSING", ha="center", va="center", fontsize=10)
        ax.set_frame_on(True)
    else:
        img = plt.imread(str(img_path))
        ax.imshow(img)
    if border_color:
        for sp in ax.spines.values():
            sp.set_edgecolor(border_color)
            sp.set_linewidth(3.0)
            sp.set_visible(True)
    else:
        for sp in ax.spines.values():
            sp.set_visible(True)
            sp.set_linewidth(0.8)
            sp.set_edgecolor("#94a3b8")


def _build_idx_to_path(imgs: Sequence[Tuple[int, Path]]) -> Dict[int, Path]:
    return {i: p for i, p in imgs}


def render_case_montage(
    case: Case,
    out_png: Path,
    radius: int = 3,
) -> bool:
    """
    Layout:
      Row 1: true_idx-radius ... true_idx ... true_idx+radius
      Row 2: pred_idx-radius ... pred_idx ... pred_idx+radius
    """
    if case.true_idx is None or case.pred_idx is None:
        return False
    imgs = _list_images_by_idx(case.sample_path)
    if not imgs:
        return False
    idx2p = _build_idx_to_path(imgs)

    r = int(radius)
    cols = 2 * r + 1

    plt = _matplotlib()
    fig = plt.figure(figsize=(cols * 2.0, 5.2), dpi=160)
    fig.suptitle(
        f"[{case.split}] err={case.error}  true_idx={case.true_idx}  pred_idx={case.pred_idx}  {case.sample_basename}",
        fontsize=10,
    )

    # True row
    for j, idx in enumerate(range(case.true_idx - r, case.true_idx + r + 1)):
        ax = fig.add_subplot(2, cols, 1 + j)
        p = idx2p.get(idx)
        border = "#16a34a" if idx == case.true_idx else None
        d = idx - case.true_idx
        _imshow(
            ax,
            p,
            title=(f"TRUE idx={idx} (Δ{d:+d})" if idx == case.true_idx else f"true idx={idx} (Δ{d:+d})"),
            border_color=border,
        )

    # Pred row
    for j, idx in enumerate(range(case.pred_idx - r, case.pred_idx + r + 1)):
        ax = fig.add_subplot(2, cols, cols + 1 + j)
        p = idx2p.get(idx)
        border = "#ef4444" if idx == case.pred_idx else None
        d = idx - case.pred_idx
        _imshow(
            ax,
            p,
            title=(f"PRED idx={idx} (Δ{d:+d})" if idx == case.pred_idx else f"pred idx={idx} (Δ{d:+d})"),
            border_color=border,
        )

    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(out_png)
    plt.close(fig)
    return True


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--badcases_json",
        type=str,
        default="badcase_report_current/badcases_top.json",
        help="badcases_top.json path",
    )
    ap.add_argument("--out_dir", type=str, default="badcase_report_current/img_compare")
    ap.add_argument("--radius", type=int, default=3, help="±radius around true/pred idx")
    ap.add_argument("--max_cases", type=int, default=30, help="render at most N cases")
    args = ap.parse_args()

    badcases_json = Path(args.badcases_json)
    out_dir = Path(args.out_dir)
    _ensure_dir(out_dir)

    cases = _read_cases(badcases_json)
    cases = sorted(cases, key=lambda c: (-c.error, c.split, c.sample_basename))

    n_ok = 0
    n_fail = 0
    for i, c in enumerate(cases[: max(1, int(args.max_cases))]):
        name = f"{i:03d}_{_safe_name(c.split)}_{_safe_name(c.sample_basename)}_err{c.error}_t{c.true_idx}_p{c.pred_idx}.png"
        ok = render_case_montage(c, out_dir / name, radius=int(args.radius))
        if ok:
            n_ok += 1
        else:
            n_fail += 1

    print(f"Wrote {n_ok} montages to: {out_dir.resolve()}")
    if n_fail:
        print(f"Skipped {n_fail} cases (missing idx/images).")


if __name__ == "__main__":
    main()

