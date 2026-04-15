#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录 dinov3-main 执行:
#   bash grid_diff_tcn/hier/scripts/infer.sh train
#   bash grid_diff_tcn/hier/scripts/infer.sh test

SPLIT="${1:-train}"
if [[ "${SPLIT}" == "train" ]]; then
  SAMPLES_INFO="data_drilling/samples_info_train.json"
  OUTPUT_JSON="grid_diff_tcn/inference_results_hierarchical_train_dinov3_vits.json"
  PRECOMPUTED_DIR="grid_diff_tcn/cache_dinov3_features_vits"
elif [[ "${SPLIT}" == "test" ]]; then
  SAMPLES_INFO="data_drilling/samples_info_test.json"
  OUTPUT_JSON="grid_diff_tcn/inference_results_hierarchical_test_dinov3_vits.json"
  PRECOMPUTED_DIR="grid_diff_tcn/cache_dinov3_features_vits"
fi

python3 -m grid_diff_tcn.hier.infer \
  --samples_info "${SAMPLES_INFO}" \
  --ckpt grid_diff_tcn/grid_diff_tcn_hierarchical_dinov3_vits.pt \
  --output "${OUTPUT_JSON}" \
  --precomputed_dir "${PRECOMPUTED_DIR}" \
  --batch_size 4 \
  --num_workers 4 \
  --device cuda \
  --img_size 128 \
  --roi_size 96 \
  --max_frames_per_layer 8 \
  --penetration_radius 2 \
  --cc_min_area 12 \
  --cc_expand_ratio 0.2 \
  --final_roi_scale 0.85 \
  --roi_window_side 32 \
  --frame_channels 128,128 \
  --layer_tcn_channels 128,128 \
  --kernel_size 3 \
  --d_model 128 \
  --num_heads 4 \
  --num_transformer_layers 2 \
  --dim_feedforward 512 \
  --dropout 0.1 \
  --extra_dim 0 \
  --use_frame_gru \
  --use_frame_attn_pool \
  --frame_gru_layers 1 \
  --use_multiscale \
  --use_dinov3 \
  --dinov3_model vit_small \
  --dinov3_roi_size 224 \
  --dinov3_feat_dim 384 \
  --decision_method s3wd \
  --lock_layers 30 \
  --s3wd_accept 0.6 \
  --s3wd_reject 0.5 \
  --s3wd_wait 4 \
  --k 9 \
  --min_thresh 0.3

  # --unc_samples 10 \
  # --unc_gate \
  # --unc_var_median_thresh 0.05
