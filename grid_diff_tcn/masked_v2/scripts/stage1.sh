#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Stage 1: MIM 预训练 (masked_v2)
#   encoder + decoder 联合训练，classifier 冻结
# ============================================================

# 获取脚本所在目录（向上4级得到 repo 根目录）
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- 数据 ----
SAMPLES_INFO="${SAMPLES_INFO:-data_drilling/samples_info_train_split.json}"
VAL_SAMPLES_INFO="${VAL_SAMPLES_INFO:-data_drilling/samples_info_val.json}"

# 加速: 预裁剪 ROI 缓存（必须，已由 pre_crop.py 生成）
CROP_CACHE_DIR="${CROP_CACHE_DIR:-data_drilling/roi_cache}"

# ---- DINOv3 ----
DINOV3_MODEL="${DINOV3_MODEL:-vit_small}"
DINOV3_FEAT_DIM="${DINOV3_FEAT_DIM:-384}"
DINOV3_ROI_SIZE="${DINOV3_ROI_SIZE:-224}"
DINOV3_CHUNK_SIZE="${DINOV3_CHUNK_SIZE:-32}"

# ---- 模型 ----
D_MODEL="${D_MODEL:-128}"
NHEAD="${NHEAD:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
# v2: encoder 默认可训练
FREEZE_ENCODER="${FREEZE_ENCODER:-False}"
MASK_RATIO="${MASK_RATIO:-0.75}"
MASK_SHAPE="${MASK_SHAPE:-circle}"

# ---- 训练 ----
BATCH_SIZE="${BATCH_SIZE:-1}"   # 必须用 1，避免 collate padding 导致的 reshape 问题
EPOCHS="${EPOCHS:-30}"
LR="${LR:-1e-4}"                # decoder lr
ENCODER_LR="${ENCODER_LR:-1e-6}"  # encoder lr（低 100 倍，防止遗忘）
PATIENCE="${PATIENCE:-10}"
NUM_WORKERS="${NUM_WORKERS:-4}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
PRELOAD="${PRELOAD:-1}"
RESUME_FROM="${RESUME_FROM:-}"

# ---- 调试 ----
# 快速验证流程：设为正整数（如 3），限制训练样本数
MAX_SAMPLES="${MAX_SAMPLES:-}"

# ---- 保存 ----
SAVE_DIR="${SAVE_DIR:-grid_diff_tcn/masked_v2/checkpoints}"
mkdir -p "$SAVE_DIR"
SAVE="${SAVE:-${SAVE_DIR}/stage1.pt}"

# ---- 其他 ----
MAX_FRAMES_PER_LAYER="${MAX_FRAMES_PER_LAYER:-10}"
LOCK_LAYERS="${LOCK_LAYERS:-30}"

ARGS=(
  --samples_info "$SAMPLES_INFO"
  --val_samples_info "$VAL_SAMPLES_INFO"
  --stage 1
  --dinov3_model "$DINOV3_MODEL"
  --dinov3_feat_dim "$DINOV3_FEAT_DIM"
  --dinov3_roi_size "$DINOV3_ROI_SIZE"
  --d_model "$D_MODEL"
  --nhead "$NHEAD"
  --num_layers "$NUM_LAYERS"
  --freeze_encoder "$FREEZE_ENCODER"
  --mask_ratio "$MASK_RATIO"
  --mask_shape "$MASK_SHAPE"
  --dinov3_chunk_size "$DINOV3_CHUNK_SIZE"
  --accum_steps "$ACCUM_STEPS"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --encoder_lr "$ENCODER_LR"
  --patience "$PATIENCE"
  --num_workers "$NUM_WORKERS"
  --max_frames_per_layer "$MAX_FRAMES_PER_LAYER"
  --lock_layers "$LOCK_LAYERS"
  --seed 42
  --save "$SAVE"
)

if [[ -n "$CROP_CACHE_DIR" && -d "$CROP_CACHE_DIR" ]]; then
  ARGS+=(--crop_cache_dir "$CROP_CACHE_DIR")
fi

if [[ "$PRELOAD" == "1" || "$PRELOAD" == "true" ]]; then
  ARGS+=(--preload)
fi

if [[ -n "$RESUME_FROM" && -f "$RESUME_FROM" ]]; then
  ARGS+=(--resume_from "$RESUME_FROM")
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi

echo "=============================================="
echo "Stage 1: MIM Pre-training (encoder + decoder)"
echo "freeze_encoder  : $FREEZE_ENCODER"
echo "encoder_lr      : $ENCODER_LR"
echo "decoder_lr      : $LR"
echo "batch_size      : $BATCH_SIZE"
echo "max_samples     : ${MAX_SAMPLES:-(all)}"
echo "crop_cache_dir  : $CROP_CACHE_DIR"
echo "save            : $SAVE"
echo "=============================================="
cd "$REPO_ROOT"
python3 -m grid_diff_tcn.masked_v2.train "${ARGS[@]}"
