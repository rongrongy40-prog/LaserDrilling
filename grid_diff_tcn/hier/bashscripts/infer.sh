#!/usr/bin/env bash
set -euo pipefail

# 在仓库根目录 dinov3-main 执行:
#   bash grid_diff_tcn/hier/infer.sh           # 默认 train（预计算 cache）
#   bash grid_diff_tcn/hier/infer.sh train
#   bash grid_diff_tcn/hier/infer.sh test    # 无 cache，在线提特征

SPLIT="${1:-train}"
if [[ "${SPLIT}" == "train" ]]; then
  SAMPLES_INFO="data_drilling/samples_info_train.json"
  OUTPUT_JSON="grid_diff_tcn/inference_results_hierarchical_train.json"
  PRECOMPUTED_DIR="grid_diff_tcn/cache_hierarchical_features_v2"
  PRECOMPUTED_ARG=(--precomputed_dir "${PRECOMPUTED_DIR}")
elif [[ "${SPLIT}" == "test" ]]; then
  SAMPLES_INFO="data_drilling/samples_info_test.json"
  OUTPUT_JSON="grid_diff_tcn/inference_results_hierarchical_test.json"
  PRECOMPUTED_ARG=()
  echo "[hier/infer] test：无 --precomputed_dir，在线计算特征（较慢）。"
else
  echo "Invalid split: ${SPLIT}. Use 'train' or 'test'."
  exit 1
fi

python3 -m grid_diff_tcn.hier.infer \
  --samples_info "${SAMPLES_INFO}" \
  "${PRECOMPUTED_ARG[@]}" \
  --ckpt grid_diff_tcn/grid_diff_tcn_hierarchical_var_gate.pt \
  --output "${OUTPUT_JSON}" \
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
  --exclude_json data_drilling/no_laser_change_equalbox_full_mad00005_center_and_below.json \
  --frame_channels 64,64 \
  --layer_tcn_channels 64,64 \
  --kernel_size 3 \
  --d_model 64 \
  --num_heads 4 \
  --num_transformer_layers 2 \
  --dim_feedforward 256 \
  --dropout 0.1 \
  --decision_method topkmedian \
  --lock_layers 30 \
  --k 9 \
  --min_thresh 0.3 \
  --unc_samples 10 \
  --unc_gate \
  --unc_var_median_thresh 0.05
