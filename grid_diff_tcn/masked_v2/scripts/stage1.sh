# ============================================================
# Stage 1: MAE 预训练 (masked_v2)
#   encoder UNFROZEN — encoder + MAE decoder 联合训练，encoder 学习领域适配特征
#   相比旧版 CenterMask + CLS-only decoder，这是真正的标准 MAE
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- 数据 ----
SAMPLES_INFO="${SAMPLES_INFO:-data_drilling/samples_info_train_split.json}"
VAL_SAMPLES_INFO="${VAL_SAMPLES_INFO:-data_drilling/samples_info_val_split.json}"

# 加速: 预裁剪 ROI 缓存
CROP_CACHE_DIR="${CROP_CACHE_DIR:-data_drilling/roi_cache}"

# ---- DINOv3 ----
DINOV3_MODEL="${DINOV3_MODEL:-vit_small}"
DINOV3_FEAT_DIM="${DINOV3_FEAT_DIM:-384}"
DINOV3_ROI_SIZE="${DINOV3_ROI_SIZE:-128}"
DINOV3_CHUNK_SIZE="${DINOV3_CHUNK_SIZE:-256}"

# ---- 模型 ----
D_MODEL="${D_MODEL:-128}"
NHEAD="${NHEAD:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
# encoder 默认 unfreeze，encoder + decoder 联合训练
FREEZE_ENCODER="${FREEZE_ENCODER:-False}"
ENCODER_LR="${ENCODER_LR:-1e-6}"  # encoder lr (smaller, since pretrained)
MASK_RATIO="${MASK_RATIO:-0.75}"

# ---- 训练 ----
BATCH_SIZE="${BATCH_SIZE:-2}"
EPOCHS="${EPOCHS:-100}"
LR="${LR:-1e-4}"                # MAE decoder lr
PATIENCE="${PATIENCE:-8}"
NUM_WORKERS="${NUM_WORKERS:-8}"
ACCUM_STEPS="${ACCUM_STEPS:-4}"
PRELOAD="${PRELOAD:-0}"
RESUME_FROM="${RESUME_FROM:-}"

# ---- 调试 ----
MAX_SAMPLES="${MAX_SAMPLES:-}"

# ---- 保存 ----
SAVE_DIR="${SAVE_DIR:-grid_diff_tcn/masked_v2/checkpoints}"
mkdir -p "$SAVE_DIR"
SAVE="${SAVE:-${SAVE_DIR}/stage1.pt}"

# ---- 其他 ----
MAX_FRAMES_PER_LAYER="${MAX_FRAMES_PER_LAYER:-15}"
LOCK_LAYERS="${LOCK_LAYERS:-30}"
STAGE1_SCHEDULER="${STAGE1_SCHEDULER:-cosine}"

# ---- MAE decoder 超参数 ----
MAE_DECODER_DIM="${MAE_DECODER_DIM:-256}"
MAE_DECODER_DEPTH="${MAE_DECODER_DEPTH:-4}"
MAE_DECODER_HEADS="${MAE_DECODER_HEADS:-8}"

# ---- 多卡 ----
DEVICE_IDS="${DEVICE_IDS:-0}"

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
  --dinov3_chunk_size "$DINOV3_CHUNK_SIZE"
  --batch_size "$BATCH_SIZE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --encoder_lr "$ENCODER_LR"
  --patience "$PATIENCE"
  --num_workers "$NUM_WORKERS"
  --max_frames_per_layer "$MAX_FRAMES_PER_LAYER"
  --lock_layers "$LOCK_LAYERS"
  --stage1_scheduler "$STAGE1_SCHEDULER"
  --seed 42
  --save "$SAVE"
  --device_ids $DEVICE_IDS
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
echo "Stage 1: MAE Pre-training (joint encoder + decoder)"
echo "freeze_encoder : $FREEZE_ENCODER"
echo "encoder_lr     : $ENCODER_LR"
echo "decoder_lr     : $LR"
echo "mask_ratio     : $MASK_RATIO"
echo "decoder_dim    : $MAE_DECODER_DIM"
echo "decoder_depth  : $MAE_DECODER_DEPTH"
echo "decoder_heads  : $MAE_DECODER_HEADS"
echo "batch_size     : $BATCH_SIZE"
echo "max_samples    : ${MAX_SAMPLES:-(all)}"
echo "crop_cache_dir : $CROP_CACHE_DIR"
echo "scheduler      : $STAGE1_SCHEDULER"
echo "device_ids     : $DEVICE_IDS"
echo "save           : $SAVE"
echo "=============================================="
cd "$REPO_ROOT"
python3 -m grid_diff_tcn.masked_v2.train "${ARGS[@]}"
