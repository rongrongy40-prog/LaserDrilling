#!/usr/bin/env python3
"""Plot probability curves for all (or all badcase) inference samples."""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import os

PROBS_CSV  = "grid_diff_tcn/masked_v2/probs_test.csv"
OUT_DIR    = "grid_diff_tcn/masked_v2/badcase_plots"
os.makedirs(OUT_DIR, exist_ok=True)

ACCENT  = "#4F46E5"
GREEN   = "#059669"
YELLOW  = "#D97706"
RED     = "#DC2626"
GRAY    = "#9CA3AF"
BG      = "#F4F5F8"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.facecolor": BG,
    "axes.facecolor": BG,
})

df = pd.read_csv(PROBS_CSV)
# one row per (sample, layer_idx)
samples = df["sample"].unique()
print(f"Total samples: {len(samples)}")

for sample_path in samples:
    sp_df = df[df["sample"] == sample_path].sort_values("physical_layer")
    if sp_df.empty:
        continue

    x     = sp_df["physical_layer"].values
    y     = sp_df["prob"].values
    valid = sp_df["is_valid"].values > 0
    true_l  = int(sp_df["true_layer"].iloc[0])
    pred_l  = int(sp_df["pred_layer"].iloc[0])
    error   = int(sp_df["error"].iloc[0])
    src     = sp_df["decision_source"].iloc[0]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(x[~valid], y[~valid], color=GRAY, alpha=0.4, width=1.0, zorder=1)
    ax.bar(x[valid],  y[valid],  color=ACCENT, alpha=0.85, width=1.0, zorder=2)

    ax.axvline(true_l, color=GREEN, lw=2, ls="--", label=f"True ({true_l})", zorder=3)
    if pred_l >= 0:
        clr = YELLOW if error > 0 else GREEN
        ax.axvline(pred_l, color=clr, lw=2, ls=":", label=f"Pred ({pred_l})", zorder=3)
    ax.axhline(0.6, color=RED, lw=1.2, ls="--", alpha=0.7, label="S3WD thresh (0.6)")

    ax.set_xlabel("Physical Layer", fontsize=13)
    ax.set_ylabel("Penetration Probability", fontsize=13)
    ax.set_xlim(x.min() - 2, x.max() + 2)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10, framealpha=0.9, loc="upper left")

    fname = sample_path.split("/")[-1][:50].replace(" ", "_")
    err_tag = f"err={error:+d}" if error != 0 else "correct"
    ax.set_title(f"{fname}  |  {src}  |  {err_tag}", fontsize=11, pad=8)

    plt.tight_layout()
    out = os.path.join(OUT_DIR, f"{fname}.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [{error:+3d}] {out}")

print(f"\nSaved {len(samples)} plots → {OUT_DIR}/")
