#!/usr/bin/env bash
# ==============================================================================
# Stage 2 推理脚本 — 配套 infer_simple.py
#
# 用法:
#   bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh train       # 训练集
#   bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh test        # 测试集
#   bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh val        # 验证集
#
# 模式选择（默认 cached）:
#   --mode online    在线裁剪 ROI + 在线 DINOv3 特征提取（最灵活）
#   --mode cached    预裁剪 ROI + 预计算特征（最快，默认）
#
# 调试用少量样本:
#   MAX_SAMPLES=5 bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh train
#
# 决策方法:
#   DECISION=s3wd     argmax + S3WD 后处理（默认）
#   DECISION=learned  TemporalDecisionHead
#
# 手动指定:
#   CHECKPOINT=...  SAMPLES_INFO=...  MODE=online  bash ...
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# ---- Split ----
SPLIT="${1:-test}"
case "$SPLIT" in
  train)  SAMPLES_INFO="${SAMPLES_INFO:-data_drilling/samples_info_train_split_v2.json}"; OUTPUT_CSV="grid_diff_tcn/masked_v2/inference_results_train.csv"; OUTPUT_PROBS_CSV="grid_diff_tcn/masked_v2/probs_train.csv" ;;
  test)   SAMPLES_INFO="${SAMPLES_INFO:-data_drilling/samples_info_test_split_v2.json}";  OUTPUT_CSV="grid_diff_tcn/masked_v2/inference_results_test.csv";   OUTPUT_PROBS_CSV="grid_diff_tcn/masked_v2/probs_test.csv" ;;
  val)    SAMPLES_INFO="${SAMPLES_INFO:-data_drilling/samples_info_val_split_v2.json}";        OUTPUT_CSV="grid_diff_tcn/masked_v2/inference_results_val.csv";       OUTPUT_PROBS_CSV="grid_diff_tcn/masked_v2/probs_val.csv" ;;
  *)      echo "ERROR: split must be train|test|val, got '$SPLIT'"; exit 1 ;;
esac

# ---- 路径 ----
CHECKPOINT="${CHECKPOINT:-grid_diff_tcn/masked_v2/checkpoints/stage2.pt}"
ROI_CACHE_DIR="${ROI_CACHE_DIR:-data_drilling/roi_cache}"
PRECOMPUTED_DIR="${PRECOMPUTED_DIR:-grid_diff_tcn/masked_v2/features_cache2}"

# ---- DINOv3 ----
DINOV3_MODEL="${DINOV3_MODEL:-vit_small}"
DINOV3_FEAT_DIM="${DINOV3_FEAT_DIM:-384}"
DINOV3_ROI_SIZE="${DINOV3_ROI_SIZE:-224}"
DINOV3_CHUNK_SIZE="${DINOV3_CHUNK_SIZE:-256}"

# ---- 模型参数 ----
D_MODEL="${D_MODEL:-128}"
NHEAD="${NHEAD:-4}"
NUM_LAYERS="${NUM_LAYERS:-2}"
FREEZE_ENCODER="${FREEZE_ENCODER:-True}"

# ---- 推理 ----
MODE="${MODE:-cached}"          # cached | online
DECISION="${DECISION:-s3wd}"   # s3wd | learned
LOCK_LAYERS="${LOCK_LAYERS:-30}"
S3WD_WAIT="${S3WD_WAIT:-3}"
S3WD_THRESH="${S3WD_THRESH:-0.6}"
S3WD_ACCEPT="${S3WD_ACCEPT:-0.7}"
BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MAX_FRAMES="${MAX_FRAMES:-12}"
MAX_SAMPLES="${MAX_SAMPLES:-}"  # 空=全部

# ---- CPU 强制 ----
if [[ "${FORCE_CPU:-0}" == "1" ]]; then
  export CUDA_VISIBLE_DEVICES=""
fi

mkdir -p "$(dirname "${OUTPUT_CSV}")"

# ---- 构建参数 ----
ARGS=(
  --mode "$MODE"
  --checkpoint "$CHECKPOINT"
  --samples_info "$SAMPLES_INFO"
  --output_csv "$OUTPUT_CSV"
  --output_probs_csv "$OUTPUT_PROBS_CSV"
  --decision_method "$DECISION"
  --lock_layers "$LOCK_LAYERS"
  --s3wd_wait "$S3WD_WAIT"
  --s3wd_threshold "$S3WD_THRESH"
  --s3wd_accept "$S3WD_ACCEPT"
  --max_frames_per_layer "$MAX_FRAMES"
)

if [[ -n "$ROI_CACHE_DIR" && -d "$ROI_CACHE_DIR" ]]; then
  ARGS+=(--crop_cache_dir "$ROI_CACHE_DIR")
fi
if [[ -n "$PRECOMPUTED_DIR" && -d "$PRECOMPUTED_DIR" ]]; then
  ARGS+=(--precomputed_dir "$PRECOMPUTED_DIR")
fi
if [[ -n "$MAX_SAMPLES" ]]; then
  ARGS+=(--max_samples "$MAX_SAMPLES")
fi

# ---- 打印配置 ----
echo "=============================================="
echo "Stage 2 Inference (infer_simple.py)"
echo "split            : $SPLIT"
echo "mode             : $MODE"
echo "decision         : $DECISION"
echo "checkpoint       : $CHECKPOINT"
echo "samples_info     : $SAMPLES_INFO"
echo "output_csv       : $OUTPUT_CSV"
echo "lock_layers      : $LOCK_LAYERS"
echo "s3wd_wait        : $S3WD_WAIT"
echo "s3wd_threshold   : $S3WD_THRESH"
echo "s3wd_accept      : $S3WD_ACCEPT"
echo "batch_size       : $BATCH_SIZE"
echo "num_workers      : $NUM_WORKERS"
echo "max_samples      : ${MAX_SAMPLES:-(all)}"
echo "roi_cache_dir    : ${ROI_CACHE_DIR:-(none)}"
echo "precomputed_dir  : ${PRECOMPUTED_DIR:-(none)}"
echo "=============================================="

cd "$REPO_ROOT"
python3 -m grid_diff_tcn.masked_v2.infer_simple "${ARGS[@]}"
