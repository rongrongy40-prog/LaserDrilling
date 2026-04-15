#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录 dinov3-main 执行:
#   bash grid_diff_tcn/hier/scripts/train.sh
#
# DINOv3 训练模式（推荐）:
#   bash grid_diff_tcn/hier/scripts/train.sh --use_dinov3 --dinov3_model vit_base
#
# 手工特征模式（对比基准）:
#   bash grid_diff_tcn/hier/scripts/train.sh

python3 -m grid_diff_tcn.hier.train \
  --samples_info data_drilling/samples_info_train.json \
  --precomputed_dir grid_diff_tcn/cache_dinov3_features_vits \
  --save grid_diff_tcn/grid_diff_tcn_hierarchical_dinov3_vits.pt \
  --epochs 40 \
  --batch_size 4 \
  --lr 1e-4 \
  --val_ratio 0.2 \
  --val_seed 42 \
  --num_workers 4 \
  --device cuda \
  --img_size 128 \
  --max_frames_per_layer 8 \
  --penetration_radius 2 \
  --cc_min_area 12 \
  --cc_expand_ratio 0.2 \
  --final_roi_scale 0.85 \
  --roi_window_side 32 \
  --frame_channels 128,128 \
  --layer_tcn_channels 128,128,128 \
  --kernel_size 3 \
  --d_model 128 \
  --num_heads 4 \
  --num_transformer_layers 2 \
  --dim_feedforward 512 \
  --dropout 0.1 \
  --kl_weight 0.01 \
  --focal_gamma 2.0 \
  --focal_alpha 0.75 \
  --weight_pos 10.0 \
  --loc5_weight 0.3 \
  --within5_weight 0.7 \
  --lock_layers 30 \
  --val_decision s3wd \
  --val_s3wd_accept 0.9 \
  --val_s3wd_reject 0.75 \
  --val_s3wd_wait 3 \
  --val_k 9 \
  --val_min_thresh 0.3 \
  --gs_s3wd \
  --gs_accept_range 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 \
  --gs_reject_range 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 \
  --gs_wait_range 1 2 3 4 5 \
  --gs_save_grid \
  --patience 20 \
  --lr_scheduler \
  --lr_patience 5 \
  --lr_factor 0.5 \
  --use_frame_gru \
  --use_frame_attn_pool \
  --frame_gru_layers 1 \
  --use_multiscale \
  \
  --use_dinov3 \
  --dinov3_model vit_small \
  --dinov3_roi_size 224 \
  --dinov3_feat_dim 384 \


  # --use_grayscale \
  # --val_decision topkmedian \
  # --val_unc_samples 20 \
  # --val_unc_gate \
  # --val_unc_var_median_thresh 0.05 \
