#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 预裁剪脚本：生成 ROI 缓存（固定 box 策略）
#
# 策略：
#   1. 跳过前 30 层
#   2. 从第 31 层起，取连续 20 帧
#   3. 对每帧检测 box，取所有 box 的并集作为最终裁剪框
#   4. resize 成 roi_size x roi_size
#
# 使用方式：
#   bash grid_diff_tcn/masked_v2/scripts/pre_crop.sh
#
# 参数覆盖示例：
#   SKIP_LAYERS=30 NUM_ANCHOR_FRAMES=20 ROI_SIZE=64 \
#     bash grid_diff_tcn/masked_v2/scripts/pre_crop.sh
#
# 断点续传：直接重新运行即可，已存在的 .pt 文件会跳过（OVERWRITE=1 重新生成）
# ============================================================

SAMPLES_INFO="${SAMPLES_INFO:-data_drilling/samples_info_train_split.json}"
VAL_SAMPLES_INFO="${VAL_SAMPLES_INFO:-data_drilling/samples_info_val_split.json}"
CACHE_DIR="${CACHE_DIR:-data_drilling/roi_cache}"
ROI_SIZE="${ROI_SIZE:-224}"
MAX_FRAMES="${MAX_FRAMES:-15}"
MAX_LAYERS="${MAX_LAYERS:-}"
SKIP_LAYERS="${SKIP_LAYERS:-30}"
NUM_ANCHOR_FRAMES="${NUM_ANCHOR_FRAMES:-20}"
MAX_WORKERS="${MAX_WORKERS:-}"
OVERWRITE="${OVERWRITE:-}"

ARGS=(
  --samples_info "$SAMPLES_INFO"
  --cache_dir "$CACHE_DIR"
  --roi_size "$ROI_SIZE"
  --max_frames "$MAX_FRAMES"
  --skip_layers "$SKIP_LAYERS"
  --num_anchor_frames "$NUM_ANCHOR_FRAMES"
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
echo "pre_crop: 预裁剪 ROI 图片（固定 box 策略）"
echo "samples_info   : $SAMPLES_INFO"
echo "cache_dir      : $CACHE_DIR"
echo "roi_size       : $ROI_SIZE"
echo "skip_layers    : $SKIP_LAYERS"
echo "anchor_frames  : $NUM_ANCHOR_FRAMES"
echo "max_frames     : $MAX_FRAMES"
echo "max_layers     : ${MAX_LAYERS:-None}"
echo "val_samples_info: ${VAL_SAMPLES_INFO:-disabled}"
echo "max_workers    : ${MAX_WORKERS:-auto}"
echo "overwrite      : ${OVERWRITE:-0}"
echo "=============================================="

python3 -m grid_diff_tcn.masked_v2.pre_crop "${ARGS[@]}"
