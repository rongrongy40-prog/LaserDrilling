#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录 dinov3-main 执行:
#   bash grid_diff_tcn/hier/precompute.sh
# 输出目录需与 hier/train.sh 里 --precomputed_dir 一致。

python3 -m grid_diff_tcn.hier.precompute \
  --samples_info data_drilling/samples_info_train.json \
  --out_dir grid_diff_tcn/cache_hierarchical_features_v2 \
  --by_name \
  --num_workers 8 \
  --img_size 128 \
  --roi_size 96 \
  --max_frames_per_layer 8 \
  --penetration_radius 2 \
  --cc_min_area 12 \
  --cc_expand_ratio 0.2 \
  --final_roi_scale 0.85 \
  --exclude_json data_drilling/no_laser_change_equalbox_full_mad00005_center_and_below.json \
  --use_hole_anchor_box \
  --hole_anchor_num_images 10

# 若 train 不使用孔锚框，请删除上面两行参数。
