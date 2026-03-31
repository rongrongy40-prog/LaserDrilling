#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录 dinov3-main 执行:
#   bash grid_diff_tcn/hier/train.sh

python3 -m grid_diff_tcn.hier.train \
  --samples_info data_drilling/samples_info_train.json \
  --precomputed_dir grid_diff_tcn/cache_hierarchical_features_v2 \
  --save grid_diff_tcn/grid_diff_tcn_hierarchica_var_gate.pt \
  --epochs 50 \
  --batch_size 4 \
  --lr 1e-4 \
  --val_ratio 0.2 \
  --val_seed 42 \
  --num_workers 4 \
  --device cuda \
  --img_size 128 \
  --roi_size 96 \
  --max_frames_per_layer 8 \
  --penetration_radius 2 \
  --cc_min_area 12 \
  --cc_expand_ratio 0.2 \
  --final_roi_scale 0.85 \
  --exclude_json data_drilling/no_laser_change_equalbox_full_mad00005_center_and_below.json \
  --frame_channels 64,64 \
  --layer_tcn_channels 64,64 \
  --kernel_size 3 \
  --d_model 64 \
  --num_heads 4 \
  --num_transformer_layers 2 \
  --dim_feedforward 256 \
  --dropout 0.1 \
  --kl_weight 0.0 \
  --focal_gamma 2.0 \
  --focal_alpha 0.75 \
  --weight_pos 10.0 \
  --loc5_weight 0.3 \
  --within5_weight 0.7 \
  --lock_layers 30 \
  --val_k 9 \
  --val_min_thresh 0.3 \
  --val_unc_samples 20 \
  --val_unc_gate \
  --val_unc_var_median_thresh 0.05
