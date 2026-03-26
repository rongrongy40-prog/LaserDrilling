#!/usr/bin/env bash

set -e
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

SAMPLES_INFO="${SAMPLES_INFO:-../data_drilling/samples_info_train.json}"
# 默认用本脚本所在目录下的 cache_features_train；若环境变量是占位路径也改用脚本目录
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-$SCRIPT_DIR/cache_features_train}"
[[ "$PRECOMPUTED_DIR" == *"/path/to"* ]] && PRECOMPUTED_DIR="$SCRIPT_DIR/cache_features_train"
CKPT="${CKPT:-grid_diff_tcn_transformer.pt}"
OUTPUT="${OUTPUT:-inference_results_train.json}"
DEVICE="${DEVICE:-cuda}"

python inference.py \
  --samples_info "$SAMPLES_INFO" \
  --precomputed_dir "$PRECOMPUTED_DIR" \
  --ckpt "$CKPT" \
  --output "$OUTPUT" \
  --use_transformer \
  --topkmedian \
  --k 9 \
  --min_thresh 0.4 \
  --unc_samples 8 \
  --device "$DEVICE"
# 可选：与训练验证完全一致时再加 --unc_gate --unc_var_median_thresh 0.1
