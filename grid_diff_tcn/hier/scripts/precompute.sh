#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录 dinov3-main 执行:
#   bash grid_diff_tcn/hier/precompute.sh
# 输出目录需与 hier/train.sh 里 --precomputed_dir 一致。

python3 -m grid_diff_tcn.hier.precompute \
  --samples_info data_drilling/samples_info_train.json \
  --out_dir grid_diff_tcn/cache_hierarchical_features_v2 \
  --num_workers 8 \
  --img_size 128 \
  --roi_size 96 \
  --max_frames_per_layer 8 \
  --cc_min_area 12 \
  --cc_expand_ratio 0.2 \
  --final_roi_scale 0.85 \
  --roi_window_side 32 \
