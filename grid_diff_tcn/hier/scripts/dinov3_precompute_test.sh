#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录 dinov3-main 执行:
#   bash grid_diff_tcn/hier/scripts/dinov3_precompute_test.sh
#
# 为测试集生成 DINOv3 特征缓存，写入与 train 相同的缓存目录。

python3 -m grid_diff_tcn.hier.dinov3_precompute \
  --samples_info data_drilling/samples_info_test.json \
  --out_dir grid_diff_tcn/cache_dinov3_features_vits \
  --dinov3_model vit_small \
  --dinov3_feat_dim 384 \
  --roi_size 224 \
  --num_workers 8 \
  --device cuda \
  --img_size 128 \
  --max_frames_per_layer 8 \
  --cc_min_area 12 \
  --cc_expand_ratio 0.2 \
  --final_roi_scale 0.85 \
  --roi_window_side 32 \
  --exclude_json data_drilling/no_laser_change_equalbox_full_mad00005_center_and_below.json
