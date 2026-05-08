#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Stage 2: 分类微调 (masked_v2)
#   encoder 冻结（使用 Standard MAE 学到的特征），classifier 微调
#
# 进阶选项:
#   UNFREEZE_ENCODER_STAGE2=True   继续微调 encoder（谨慎使用）
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- 数据 ----
SAMPLES_INFO="${SAMPLES_INFO:-grid_diff_tcn/samples_info_train_split.json}"
VAL_SAMPLES_INFO="${VAL_SAMPLES_INFO:-grid_diff_tcn/samples_info_val.json}"

# 加速: 预裁剪 ROI 缓存 / 预计算特征
CROP_CACHE_DIR="${CROP_CACHE_DIR:-data_drilling/roi_cache}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-}"    # Stage1 encoder 提取的特征（可选，加速训练）

# ---- DINOv3 ----
DINOV3_MODEL="${DINOV3_MODEL:-vit_small}"
DINOV3_FEAT_DIM="${DINOV3_FEAT_DIM:-384}"
DINOV3_ROI_SIZE="${DINOV3_ROI_SIZE:-224}"
DINOV3_CHUNK_SIZE="${DINOV3_CHUNK_SIZE:-256}"

# ---- 模型 ----
D_MODEL="${D_MODEL:-128}"
NHEAD="${NHEAD:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
# Stage 2 默认冻结 encoder（用 Stage 1 学到的特征）
FREEZE_ENCODER="${FREEZE_ENCODER:-True}"
MASK_RATIO="${MASK_RATIO:-0.75}"
MASK_SHAPE="${MASK_SHAPE:-circle}"

# ---- 训练 ----
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-50}"
LR="${LR:-1e-4}"                         # classifier lr (reduced from 1e-4)
ENCODER_LR_STAGE2="${ENCODER_LR_STAGE2:-1e-4}"  # encoder lr（仅 unfreeze 时用）
UNFREEZE_ENCODER_STAGE2="${UNFREEZE_ENCODER_STAGE2:-False}"  # 默认冻结 encoder
PATIENCE="${PATIENCE:-10}"
EARLY_STOPPING_PATIENCE="${EARLY_STOPPING_PATIENCE:-10}"
NUM_WORKERS="${NUM_WORKERS:-8}"
PRELOAD="${PRELOAD:-1}"
RESUME_FROM="${RESUME_FROM:-grid_diff_tcn/masked_v2/checkpoints/stage1.pt}"           # 必须指定 Stage1 checkpoint
FINETUNE_CLASSIFIER="${FINETUNE_CLASSIFIER:-True}"  # 是否微调分类头权重
ACCUM_STEPS_STAGE2="${ACCUM_STEPS_STAGE2:-4}"  # 梯度累积步数

# ---- 调试 ----
# 快速验证流程：设为正整数（如 3），限制训练样本数
MAX_SAMPLES="${MAX_SAMPLES:-}"

# ---- 保存 ----
SAVE_DIR="${SAVE_DIR:-grid_diff_tcn/masked_v2/checkpoints}"
mkdir -p "$SAVE_DIR"
SAVE="${SAVE:-${SAVE_DIR}/stage2.pt}"

# ---- 其他 ----
MAX_FRAMES_PER_LAYER="${MAX_FRAMES_PER_LAYER:-12}"
WEIGHT_POS="${WEIGHT_POS:-3.0}"
PEAK_LOSS_WEIGHT="${PEAK_LOSS_WEIGHT:-0.15}"
SMOOTHNESS_WEIGHT="${SMOOTHNESS_WEIGHT:-0.05}"
BOUNDARY_WEIGHT="${BOUNDARY_WEIGHT:-0.15}"
LOCK_LAYERS="${LOCK_LAYERS:-30}"
USE_LEARNED_DECISION="${USE_LEARNED_DECISION:-False}"
INDEX_LOSS_WEIGHT="${INDEX_LOSS_WEIGHT:-1.0}"
WITHIN5_TOLERANCE="${WITHIN5_TOLERANCE:-3}"
WITHIN5_WEIGHT="${WITHIN5_WEIGHT:-0.2}"
S3WD_WAIT="${S3WD_WAIT:-3}"
S3WD_THRESHOLD="${S3WD_THRESHOLD:-0.6}"
S3WD_ACCEPT="${S3WD_ACCEPT:-0.7}"

if [[ -z "$RESUME_FROM" ]]; then
  echo "ERROR: 必须指定 Stage1 checkpoint 路径"
  echo "  RESUME_FROM=grid_diff_tcn/masked_v2/checkpoints/stage1.pt \\"
  echo "    bash \$(basename \"\$0\")"
  exit 1
fi

ARGS=(
  --samples_info "$SAMPLES_INFO"
  --val_samples_info "$VAL_SAMPLES_INFO"
  --stage 2
  --dinov3_model "$DINOV3_MODEL"
  --dinov3_feat_dim "$DINOV3_FEAT_DIM"
  --dinov3_roi_size "$DINOV3_ROI_SIZE"
  --d_model "$D_MODEL"
  --nhead "$NHEAD"
  --num_layers "$NUM_LAYERS"
  --freeze_encoder "$FREEZE_ENCODER"
  --mask_ratio "$MASK_RATIO"
  --dinov3_chunk_size "$DINOV3_CHUNK_SIZE"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --encoder_lr_stage2 "$ENCODER_LR_STAGE2"
  --patience "$PATIENCE"
  --early_stopping_patience "$EARLY_STOPPING_PATIENCE"
  --num_workers "$NUM_WORKERS"
  --weight_pos "$WEIGHT_POS"
  --peak_loss_weight "$PEAK_LOSS_WEIGHT"
  --smoothness_weight "$SMOOTHNESS_WEIGHT"
  --boundary_weight "$BOUNDARY_WEIGHT"
  --stage2_scheduler plateau
  --plateau_factor 0.5
  --plateau_patience 3
  --plateau_min_lr 1e-6
  # --use_learned_decision
  --max_frames_per_layer "$MAX_FRAMES_PER_LAYER"
  --lock_layers "$LOCK_LAYERS"
  --seed 42
  --save "$SAVE"
  --resume_from "$RESUME_FROM"
  --finetune_classifier "$FINETUNE_CLASSIFIER"
  --accum_steps_stage2 "$ACCUM_STEPS_STAGE2"
  --within5_tolerance "$WITHIN5_TOLERANCE"
  --within5_weight "$WITHIN5_WEIGHT"
  --s3wd_wait "$S3WD_WAIT"
  --s3wd_threshold "$S3WD_THRESHOLD"
  --s3wd_accept "$S3WD_ACCEPT"
)

if [[ "$USE_LEARNED_DECISION" == "True" ]]; then
  ARGS+=(--use_learned_decision)
  ARGS+=(--index_loss_weight "$INDEX_LOSS_WEIGHT")
fi

if [[ -n "$CROP_CACHE_DIR" && -d "$CROP_CACHE_DIR" ]]; then
  ARGS+=(--crop_cache_dir "$CROP_CACHE_DIR")
fi

if [[ -n "$PRECOMPUTED_DIR" && -d "$PRECOMPUTED_DIR" ]]; then
  ARGS+=(--precomputed_dir "$PRECOMPUTED_DIR")
  ARGS+=(--use_cached_features)
fi

if [[ "$PRELOAD" == "1" || "$PRELOAD" == "true" ]]; then
  ARGS+=(--preload)
fi

if [[ "$UNFREEZE_ENCODER_STAGE2" == "True" ]]; then
  ARGS+=(--unfreeze_encoder_stage2 True)
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi

echo "=============================================="
echo "Stage 2: Classification Fine-tuning"
echo "freeze_encoder : $FREEZE_ENCODER"
echo "unfreeze_enc  : $UNFREEZE_ENCODER_STAGE2"
echo "finetune_clsf : $FINETUNE_CLASSIFIER"
echo "learned_decision: $USE_LEARNED_DECISION"
echo "encoder_lr(S2) : $ENCODER_LR_STAGE2"
echo "batch_size    : $BATCH_SIZE"
echo "accum_steps   : $ACCUM_STEPS_STAGE2"
echo "max_samples   : ${MAX_SAMPLES:-(all)}"
echo "weight_pos    : $WEIGHT_POS"
echo "peak_loss_w   : $PEAK_LOSS_WEIGHT"
echo "smoothness_w  : $SMOOTHNESS_WEIGHT"
echo "boundary_w    : $BOUNDARY_WEIGHT"
echo "label_smooth : 0.05"
echo "scheduler     : plateau"
echo "early_stop_patience: $EARLY_STOPPING_PATIENCE"
echo "lock_layers   : $LOCK_LAYERS"
echo "index_loss_w  : $INDEX_LOSS_WEIGHT"
echo "within5_tol   : $WITHIN5_TOLERANCE"
echo "precomputed   : ${PRECOMPUTED_DIR:-(none)}"
echo "save          : $SAVE"
echo "=============================================="
cd "$REPO_ROOT"
python3 -m grid_diff_tcn.masked_v2.train "${ARGS[@]}"
