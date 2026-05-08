#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# 推理脚本 (masked_v2) — 配套新的 infer.py
#
# 用法:
#   bash grid_diff_tcn/masked_v2/scripts/infer.sh test
#   bash grid_diff_tcn/masked_v2/scripts/infer.sh train
#
# 在线 ROI 裁剪（不使用缓存）:
#   ONLINE_CROP=1 bash grid_diff_tcn/masked_v2/scripts/infer.sh test
#
# 调试用少量样本:
#   MAX_SAMPLES=5 bash grid_diff_tcn/masked_v2/scripts/infer.sh test
#
# 进阶选项:
#   CHECKPOINT=...              指定模型路径
#   ROI_CACHE_DIR=...           ROI 裁剪缓存目录
#   PRECOMPUTED_DIR=...         预计算 DINOv3 特征目录（加速推理）
#   DECISION_METHOD=s3wd       s3wd | topkmedian | threshold
#   S3WD_WAIT=5                 S3WD 连续帧数要求
#   S3WD_THRESH=0.6            S3WD 概率阈值（当 RUN_VAL=1 时会被自动覆盖）
#   S3WD_ACCEPT=0.3            S3WD accept 阈值（当 RUN_VAL=1 时会被自动覆盖）
#   LOCK_LAYERS=30             安全锁层数
#   GRID_SEARCH_OUTPUT=...      网格搜索结果 JSON 输出路径
#
# ============================================================

# ---- CPU 模式（默认自动选择）----
# 设为 1 强制用 CPU，设为 0 强制用 CUDA
FORCE_CPU="${FORCE_CPU:-0}"
if [[ "$FORCE_CPU" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- Split ----
SPLIT="${1:-test}"
if [[ "${SPLIT}" == "train" ]]; then
  SAMPLES_INFO="data_drilling/samples_info_train_split.json"
  OUTPUT_CSV="grid_diff_tcn/masked_v2/inference_results_train.csv"
elif [[ "${SPLIT}" == "test" ]]; then
  SAMPLES_INFO="data_drilling/samples_info_test_split.json"
  OUTPUT_CSV="grid_diff_tcn/masked_v2/inference_results_test.csv"
else
  echo "ERROR: Unknown split '${SPLIT}'. Use 'train' or 'test'."
  exit 1
fi

# ---- 路径 ----
CHECKPOINT="${CHECKPOINT:-grid_diff_tcn/masked_v2/checkpoints/stage2.pt}"
ROI_CACHE_DIR="${ROI_CACHE_DIR:-data_drilling/roi_cache}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-}"

# ---- DINOv3 ----
DINOV3_MODEL="${DINOV3_MODEL:-vit_small}"
DINOV3_FEAT_DIM="${DINOV3_FEAT_DIM:-384}"
DINOV3_ROI_SIZE="${DINOV3_ROI_SIZE:-224}"
DINOV3_CHUNK_SIZE="${DINOV3_CHUNK_SIZE:-256}"

# ---- 模型 ----
D_MODEL="${D_MODEL:-128}"
NHEAD="${NHEAD:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
FREEZE_ENCODER="${FREEZE_ENCODER:-True}"
MASK_RATIO="${MASK_RATIO:-0.75}"
MASK_SHAPE="${MASK_SHAPE:-circle}"

# ---- 推理 ----
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
DECISION_METHOD="${DECISION_METHOD:-s3wd}"
LOCK_LAYERS="${LOCK_LAYERS:-30}"
PRELOAD="${PRELOAD:-0}"

# ---- 调试 ----
MAX_SAMPLES="${MAX_SAMPLES:-20}"

# ---- S3WD / 决策参数 ----
S3WD_WAIT="${S3WD_WAIT:-3}"
S3WD_THRESH="${S3WD_THRESH:-0.6}"
S3WD_ACCEPT="${S3WD_ACCEPT:-0.7}"

# ---- 流式推理（early stopping）----
STREAMING="${STREAMING:-0}"
STOP_THRESH="${STOP_THRESH:-0.6}"
STOP_WAIT="${STOP_WAIT:-3}"
MAX_INFERENCE_LAYERS="${MAX_INFERENCE_LAYERS:-12}"

# ---- 验证集调参 ----
RUN_VAL="${RUN_VAL:-0}"
VAL_SAMPLES_INFO="${VAL_SAMPLES_INFO:-data_drilling/samples_info_val.json}"
GRID_SEARCH_OUTPUT="${GRID_SEARCH_OUTPUT:-grid_diff_tcn/masked_v2/grid_search_results.json}"
SKIP_GRID_SEARCH="${SKIP_GRID_SEARCH:-0}"


# ---- 其他推理参数 ----
MAX_FRAMES_PER_LAYER="${MAX_FRAMES_PER_LAYER:-12}"

mkdir -p "$(dirname "${OUTPUT_CSV}")"

ARGS=(
  --checkpoint "$CHECKPOINT"
  --samples_info "$SAMPLES_INFO"
  --output_csv "$OUTPUT_CSV"
  --dinov3_model "$DINOV3_MODEL"
  --dinov3_feat_dim "$DINOV3_FEAT_DIM"
  --dinov3_roi_size "$DINOV3_ROI_SIZE"
  --dinov3_chunk_size "$DINOV3_CHUNK_SIZE"
  --d_model "$D_MODEL"
  --nhead "$NHEAD"
  --num_layers "$NUM_LAYERS"
  --freeze_encoder "$FREEZE_ENCODER"
  --mask_ratio "$MASK_RATIO"
  --mask_shape "$MASK_SHAPE"
  --batch_size "$BATCH_SIZE"
  --num_workers "$NUM_WORKERS"
  --decision_method "$DECISION_METHOD"
  --lock_layers "$LOCK_LAYERS"
  --max_frames_per_layer "$MAX_FRAMES_PER_LAYER"
  --s3wd_wait "$S3WD_WAIT"
  --s3wd_threshold "$S3WD_THRESH"
  --s3wd_accept "$S3WD_ACCEPT"
)

if [[ "$RUN_VAL" == "1" ]]; then
  ARGS+=(--run_val)
  ARGS+=(--val_samples_info "$VAL_SAMPLES_INFO")
  ARGS+=(--grid_search_output "$GRID_SEARCH_OUTPUT")
  if [[ "$SKIP_GRID_SEARCH" == "1" ]]; then
    ARGS+=(--skip_grid_search)
  fi
fi

if [[ "$ONLINE_CROP" == "1" ]]; then
  ARGS+=(--online_crop)
else
  if [[ -n "$ROI_CACHE_DIR" && -d "$ROI_CACHE_DIR" ]]; then
    ARGS+=(--roi_cache_dir "$ROI_CACHE_DIR")
  fi
fi

if [[ -n "$PRECOMPUTED_DIR" && -d "$PRECOMPUTED_DIR" ]]; then
  ARGS+=(--precomputed_dir "$PRECOMPUTED_DIR")
  ARGS+=(--use_cached_features)
fi

if [[ "$STREAMING" == "1" ]]; then
  ARGS+=(--streaming)
  ARGS+=(--stop_thresh "$STOP_THRESH")
  ARGS+=(--stop_wait "$STOP_WAIT")
  if [[ -n "$MAX_INFERENCE_LAYERS" ]]; then
    ARGS+=(--max_inference_layers "$MAX_INFERENCE_LAYERS")
  fi
fi

if [[ "$PRELOAD" == "1" || "$PRELOAD" == "true" ]]; then
  ARGS+=(--preload)
fi

if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi

echo "=============================================="
echo "Masked V2 Inference"
echo "split            : $SPLIT"
echo "checkpoint       : $CHECKPOINT"
echo "samples_info     : $SAMPLES_INFO"
echo "output_csv       : $OUTPUT_CSV"
echo "decision_method  : $DECISION_METHOD"
echo "streaming        : $STREAMING"
if [[ "$STREAMING" == "1" ]]; then
echo "stop_thresh     : $STOP_THRESH"
echo "stop_wait       : $STOP_WAIT"
echo "max_inference   : ${MAX_INFERENCE_LAYERS:-all}"
fi
echo "s3wd_wait        : $S3WD_WAIT"
echo "s3wd_threshold   : $S3WD_THRESH"
echo "s3wd_accept      : $S3WD_ACCEPT"
echo "lock_layers      : $LOCK_LAYERS"
echo "batch_size       : $BATCH_SIZE"
echo "max_samples      : ${MAX_SAMPLES:-(all)}"
echo "roi_cache_dir    : ${ROI_CACHE_DIR:-(none)}"
echo "precomputed_dir  : ${PRECOMPUTED_DIR:-(none)}"
echo "preload         : $PRELOAD"
if [[ "$RUN_VAL" == "1" ]]; then
echo "--- Validation + Grid Search ---"
echo "run_val          : ON"
echo "val_samples_info : $VAL_SAMPLES_INFO"
echo "grid_search_out  : $GRID_SEARCH_OUTPUT"
echo "skip_grid_search : $SKIP_GRID_SEARCH"
fi
if [[ "$ONLINE_CROP" == "1" ]]; then
echo "online_crop      : ${ONLINE_CROP} (MaskedDrillingDataset)"
fi
echo "=============================================="
cd "$REPO_ROOT"
python3 -m grid_diff_tcn.masked_v2.infer "${ARGS[@]}"
