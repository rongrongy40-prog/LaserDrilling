#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 特征提取脚本 (masked_v2)
#
# 推荐用法:
#   1. Stage 1 训练后，用训练好的 encoder 提取特征：
#      bash grid_diff_tcn/masked_v2/scripts/extract_features.sh \
#        --checkpoint grid_diff_tcn/masked_v2/checkpoints/stage1.pt
#
#   2. 然后 Stage 2 训练时使用预计算特征加速：
#      PRECOMPUTED_DIR=grid_diff_tcn/masked_v2/features_cache \
#        bash grid_diff_tcn/masked_v2/scripts/train.sh --stage 2 \
#          --resume_from grid_diff_tcn/masked_v2/checkpoints/stage1.pt \
#          --precomputed_dir "$PRECOMPUTED_DIR" --use_cached_features
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"

# ---- 数据 ----
SAMPLES_INFO="${SAMPLES_INFO:-data_drilling/samples_info_train_split.json}"
VAL_SAMPLES_INFO="${VAL_SAMPLES_INFO:-data_drilling/samples_info_val.json}"
OUTPUT_DIR="${OUTPUT_DIR:-grid_diff_tcn/masked_v2/features_cache}"

# ---- DINOv3 ----
DINOV3_MODEL="${DINOV3_MODEL:-vit_small}"
DINOV3_FEAT_DIM="${DINOV3_FEAT_DIM:-384}"
DINOV3_ROI_SIZE="${DINOV3_ROI_SIZE:-224}"
DINOV3_CHUNK_SIZE="${DINOV3_CHUNK_SIZE:-256}"

# ---- 提取 ----
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
CHECKPOINT="${CHECKPOINT:-}"
CROP_CACHE_DIR="${CROP_CACHE_DIR:-data_drilling/roi_cache}"

echo "=============================================="
echo "Feature Extraction (masked_v2)"
echo "samples_info : $SAMPLES_INFO"
echo "output_dir   : $OUTPUT_DIR"
echo "dinov3_model : $DINOV3_MODEL"
echo "chunk_size   : $DINOV3_CHUNK_SIZE"
echo "checkpoint   : ${CHECKPOINT:-(pretrained DINOv3)}"
echo "crop_cache   : ${CROP_CACHE_DIR:-(none, online cropping)}"
echo "=============================================="

ARGS=(
  --samples_info "$SAMPLES_INFO"
  --output_dir "$OUTPUT_DIR"
  --dinov3_model "$DINOV3_MODEL"
  --dinov3_feat_dim "$DINOV3_FEAT_DIM"
  --dinov3_roi_size "$DINOV3_ROI_SIZE"
  --dinov3_chunk_size "$DINOV3_CHUNK_SIZE"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
)

if [[ -n "$CHECKPOINT" && -f "$CHECKPOINT" ]]; then
  ARGS+=(--checkpoint "$CHECKPOINT")
fi

if [[ -n "$CROP_CACHE_DIR" && -d "$CROP_CACHE_DIR" ]]; then
  ARGS+=(--crop_cache_dir "$CROP_CACHE_DIR")
fi

cd "$REPO_ROOT"
python3 -m grid_diff_tcn.masked_v2.extract "${ARGS[@]}"
