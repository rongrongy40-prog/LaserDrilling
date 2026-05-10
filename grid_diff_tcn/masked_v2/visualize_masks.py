# -*- coding: utf-8 -*-
"""
Visualize random-center masking on real cache samples.
Saves a grid of (original, masked, overlay) for N samples.
"""

import os
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import torch

# ── config ──────────────────────────────────────────────────────────────────
CACHE_DIR = "data_drilling/roi_cache"
NUM_SAMPLES = 9          # 3×3 grid
SEED = 42
MASK_RATIO = 0.20        # 20% area
IMAGE_SIZE = 128          # cache images are 128×128
OUT_PATH = "mask_visualization.png"
# ─────────────────────────────────────────────────────────────────────────────

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

files = sorted(glob.glob(os.path.join(CACHE_DIR, "*.pt")))
sample_files = random.sample(files, min(NUM_SAMPLES, len(files)))

# Build mask generator
from grid_diff_tcn.masked_v2.masks import CenterMask
masker = CenterMask(mask_ratio=MASK_RATIO, image_size=IMAGE_SIZE)

cols = 3
rows = NUM_SAMPLES
fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.5, rows * 3.2))
fig.suptitle(
    f"Random-Center Mask  |  ratio={MASK_RATIO:.0%}  |  "
    f"patch={int((MASK_RATIO**0.5)*IMAGE_SIZE)}×{int((MASK_RATIO**0.5)*IMAGE_SIZE)} px  |  "
    f"{IMAGE_SIZE}×{IMAGE_SIZE} input  |  {len(sample_files)} samples",
    fontsize=12, y=1.01
)

for i, fpath in enumerate(sample_files):
    data = torch.load(fpath, map_location="cpu", weights_only=False)
    frames = data["frames"]          # (T, F, 3, H, W)

    # Pick a random layer & frame so we see variety
    t_idx = random.randint(0, frames.shape[0] - 1)
    f_idx = random.randint(0, frames.shape[1] - 1)
    img = frames[t_idx, f_idx]       # (3, H, W) uint8 or float

    # CenterCrop to H=W if needed (some cache might be non-square — pad with zeros)
    H, W = img.shape[-2:]
    side = min(H, W)
    img_sq = img[:, :side, :side]    # force square

    # Prepare (B=1, C, H, W) tensor for the masker
    img_t = img_sq.unsqueeze(0).float() / 255.0 if img_sq.max() > 1.0 else img_sq.unsqueeze(0).float()

    masked_t, mask_t = masker(img_t, return_mask=True)
    masked_np = masked_t.squeeze(0).permute(1, 2, 0).cpu().numpy().astype(np.float32)
    masked_np = np.clip(masked_np, 0, 1)
    mask_np   = mask_t.squeeze(0).numpy()   # (H, W) bool — True=masked

    img_np = img_sq.permute(1, 2, 0).numpy().astype(np.float32)
    if img_np.max() > 1.0:
        img_np = img_np / 255.0
    if img_np.shape[-1] != 3:
        img_np = np.repeat(img_np, 3, axis=-1)
    img_np = np.clip(img_np, 0, 1)

    row = axes[i] if cols == 1 else axes[i]

    # ── col 0: original ───────────────────────────────────────────────────
    row[0].imshow(img_np.astype(np.float32))
    row[0].set_title(os.path.basename(fpath)[:40], fontsize=7)
    row[0].axis("off")

    # ── col 1: masked image ─────────────────────────────────────────────────
    row[1].imshow(masked_np.astype(np.float32))
    row[1].set_title(f"layer={t_idx+1} frame={f_idx+1} | masked={mask_np.mean():.1%}", fontsize=7)
    row[1].axis("off")

    # ── col 2: overlay  (green=visible, red=hashed=masked) ──────────────────
    overlay = img_np.copy().astype(np.float32)
    # red tint where masked
    overlay[mask_np] = [1.0, 0.0, 0.0]
    # green tint where visible (slight)
    vis_mask = ~mask_np
    overlay[vis_mask, 1] = np.clip(overlay[vis_mask, 1] * 0.6 + 0.4, 0, 1)
    overlay = np.clip(overlay, 0, 1)

    row[2].imshow(overlay.astype(np.float32))
    row[2].set_title("green=keep | red=mask", fontsize=7)
    row[2].axis("off")

# column headers
for ax, label in zip(axes[0] if cols > 1 else [axes[0]], ["Original", "Masked (input→model)", "Overlay"]):
    ax.text(0.5, -0.08, label, ha="center", va="top",
            transform=ax.transAxes, fontsize=9, fontweight="bold")

plt.tight_layout(pad=0.5)
plt.savefig(OUT_PATH, dpi=150, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT_PATH}")
