#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 预裁剪脚本：生成 ROI 缓存
#
# 使用方式：
#   bash grid_diff_tcn/masked/scripts/pre_crop.sh \
#     --samples_info data_drilling/samples_info_train_split.json \
#     --cache_dir data_drilling/roi_cache
#
# 断点续传：直接重新运行即可，已存在的 .pt 文件会跳过（加 --overwrite 重新生成）
# ============================================================

SAMPLES_INFO="${SAMPLES_INFO:-data_drilling/samples_info_train_split.json}"
VAL_SAMPLES_INFO="${VAL_SAMPLES_INFO:-data_drilling/samples_info_val.json}"
CACHE_DIR="${CACHE_DIR:-data_drilling/roi_cache}"
ROI_SIZE="${ROI_SIZE:-224}"
MAX_FRAMES="${MAX_FRAMES:-20}"
MAX_LAYERS="${MAX_LAYERS:-}"
MAX_WORKERS="${MAX_WORKERS:-}"
OVERWRITE="${OVERWRITE:-}"

ARGS=(
  --samples_info "$SAMPLES_INFO"
  --cache_dir "$CACHE_DIR"
  --roi_size "$ROI_SIZE"
  --max_frames "$MAX_FRAMES"
)

if [[ -n "$MAX_LAYERS" ]]; then
  ARGS+=(--max_layers "$MAX_LAYERS")
fi

if [[ -n "$MAX_WORKERS" ]]; then
  ARGS+=(--max_workers "$MAX_WORKERS")
fi

if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" ]]; then
  ARGS+=(--overwrite)
fi

mkdir -p "$CACHE_DIR"

echo "=============================================="
echo "pre_crop: 预裁剪 ROI 图片"
echo "samples_info : $SAMPLES_INFO"
echo "cache_dir    : $CACHE_DIR"
echo "roi_size     : $ROI_SIZE"
echo "max_frames   : $MAX_FRAMES"
echo "max_layers   : ${MAX_LAYERS:-None}"
echo "val_samples_info: ${VAL_SAMPLES_INFO:-disabled}"
echo "max_workers     : ${MAX_WORKERS:-auto}"
echo "overwrite       : ${OVERWRITE:-0}"
echo "=============================================="

python3 -m grid_diff_tcn.masked.pre_crop "${ARGS[@]}"
