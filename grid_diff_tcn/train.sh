#!/usr/bin/env bash
# Grid-Diff TCN 训练脚本（在 grid_diff_tcn 目录下执行）

set -e
cd "$(dirname "$0")"

SAMPLES_INFO="${SAMPLES_INFO:-../data_drilling/samples_info_train.json}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-./cache_features_train}"
SAVE="${SAVE:-grid_diff_tcn.pt}"
EPOCHS="${EPOCHS:-50}"
BATCH_SIZE="${BATCH_SIZE:-16}"
DEVICE="${DEVICE:-cuda}"

# 选择一种方式运行（取消注释要用的那一组）

# ---------------------------------------------------------------------------
# 1. 基础 TCN
# ---------------------------------------------------------------------------
# python train.py \
#   --samples_info "$SAMPLES_INFO" \
#   --precomputed_dir "$PRECOMPUTED_DIR" \
#   --save "$SAVE" \
#   --epochs "$EPOCHS" \
#   --batch_size "$BATCH_SIZE" \
#   --device "$DEVICE"

# ---------------------------------------------------------------------------
# 2. Transformer-TCN（概率注意力）
# ---------------------------------------------------------------------------
# python train.py \
#   --samples_info "$SAMPLES_INFO" \
#   --precomputed_dir "$PRECOMPUTED_DIR" \
#   --use_transformer \
#   --num_transformer_layers 2 \
#   --attn_dim 64 \
#   --num_heads 4 \
#   --kl_weight 1e-4 \
#   --save "${SAVE%.pt}_transformer.pt" \
#   --epochs "$EPOCHS" \
#   --batch_size "$BATCH_SIZE" \
#   --num_workers 4 \
#   --val_holes_per_epoch 50 \
#   --val_prefetch_workers 8 \
#   --device "$DEVICE"

# ---------------------------------------------------------------------------
# 3. Transformer + 验证时不确定性（多采样 + 门控）【默认执行】
# ---------------------------------------------------------------------------
python train.py \
  --samples_info "$SAMPLES_INFO" \
  --precomputed_dir "$PRECOMPUTED_DIR" \
  --use_transformer \
  --save "${SAVE%.pt}_transformer.pt" \
  --val_unc_samples 8 \
  --val_unc_gate \
  --val_unc_var_median_thresh 0.1 \
  --epochs "$EPOCHS" \
  --batch_size "$BATCH_SIZE" \
  --device "$DEVICE"
