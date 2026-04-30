#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Per-Frame Penetration Detection Launcher
# Uses SimPenDec/stage1.pt encoder for per-frame similarity
# search + spike detection to find the penetration frame.
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ---- Paths ----
CHECKPOINT="${CHECKPOINT:-${REPO_ROOT}/SimPenDec/stage1.pt}"
TRAIN_SAMPLES_INFO="${TRAIN_SAMPLES_INFO:-${REPO_ROOT}/data_drilling/samples_info_train_split.json}"
TEST_SAMPLES_INFO="${TEST_SAMPLES_INFO:-${REPO_ROOT}/data_drilling/samples_info_test.json}"
OUTPUT_JSON="${OUTPUT_JSON:-${REPO_ROOT}/grid_diff_tcn/simpen_inference_results.json}"

# ---- Encoder ----
DINOV3_MODEL="${DINOV3_MODEL:-vit_small}"
DINOV3_FEAT_DIM="${DINOV3_FEAT_DIM:-384}"
DINOV3_CHUNK_SIZE="${DINOV3_CHUNK_SIZE:-32}"

# ---- ROI (must match pre_crop.py) ----
ROI_SIZE="${ROI_SIZE:-96}"
FINAL_ROI_SCALE="${FINAL_ROI_SCALE:-0.85}"
CC_MIN_AREA="${CC_MIN_AREA:-12}"
CC_EXPAND_RATIO="${CC_EXPAND_RATIO:-0.2}"
MIN_LASER_PIXELS="${MIN_LASER_PIXELS:-0}"
MIN_LASER_AREA_RATIO="${MIN_LASER_AREA_RATIO:-0.0}"
ROI_WINDOW_SIDE="${ROI_WINDOW_SIDE:-32}"
USE_COLOR_CC_V2_GEOMETRY="${USE_COLOR_CC_V2_GEOMETRY:-true}"

# ---- Spike detection ----
SKIP_FIRST_LAYERS="${SKIP_FIRST_LAYERS:-30}"          # skip layers <= N
BASELINE_WINDOW="${BASELINE_WINDOW:-20}"             # recent frames for baseline
SPIKE_K="${SPIKE_K:-1.0}"                             # spike_threshold = mean + k*std + lower
SPIKE_LOWER="${SPIKE_LOWER:-0.05}"                  # absolute lower bound
WARMUP_FRAMES="${WARMUP_FRAMES:-10}"                # min frames before detection activates

# ---- FAISS ----
USE_FAISS="${USE_FAISS:-true}"
FAISS_NPROBE="${FAISS_NPROBE:-8}"

DEVICE="${DEVICE:-cuda}"

echo "=============================================="
echo "Per-Frame Penetration Detection"
echo "checkpoint             : $CHECKPOINT"
echo "train_samples          : $TRAIN_SAMPLES_INFO"
echo "test_samples           : $TEST_SAMPLES_INFO"
echo "dinov3_model           : $DINOV3_MODEL"
echo "dinov3_feat_dim        : $DINOV3_FEAT_DIM"
echo "--- ROI params (pre_crop.py) ---"
echo "roi_size               : $ROI_SIZE"
echo "final_roi_scale        : $FINAL_ROI_SCALE"
echo "cc_min_area            : $CC_MIN_AREA"
echo "cc_expand_ratio        : $CC_EXPAND_RATIO"
echo "min_laser_pixels       : $MIN_LASER_PIXELS"
echo "min_laser_area_ratio   : $MIN_LASER_AREA_RATIO"
echo "roi_window_side        : $ROI_WINDOW_SIDE"
echo "use_color_cc_v2_geometry: $USE_COLOR_CC_V2_GEOMETRY"
echo "--- Spike detection ---"
echo "skip_first_layers      : $SKIP_FIRST_LAYERS"
echo "baseline_window        : $BASELINE_WINDOW  (recent frames for baseline)"
echo "spike_k                : $SPIKE_K  (threshold = mean + k*std + lower)"
echo "spike_lower            : $SPIKE_LOWER  (absolute lower bound)"
echo "warmup_frames          : $WARMUP_FRAMES  (before detection activates)"
echo "--- FAISS ---"
echo "use_faiss              : $USE_FAISS, nprobe=$FAISS_NPROBE"
echo "device                 : $DEVICE"
echo "output                 : $OUTPUT_JSON"
echo "=============================================="

ARGS=(
    --checkpoint "$CHECKPOINT"
    --train_samples_info "$TRAIN_SAMPLES_INFO"
    --test_samples_info "$TEST_SAMPLES_INFO"
    --output_json "$OUTPUT_JSON"
    --dinov3_model "$DINOV3_MODEL"
    --dinov3_feat_dim "$DINOV3_FEAT_DIM"
    --dinov3_chunk_size "$DINOV3_CHUNK_SIZE"
    --roi_size "$ROI_SIZE"
    --final_roi_scale "$FINAL_ROI_SCALE"
    --cc_min_area "$CC_MIN_AREA"
    --cc_expand_ratio "$CC_EXPAND_RATIO"
    --min_laser_pixels "$MIN_LASER_PIXELS"
    --min_laser_area_ratio "$MIN_LASER_AREA_RATIO"
    --roi_window_side "$ROI_WINDOW_SIDE"
    --use_color_cc_v2_geometry "$USE_COLOR_CC_V2_GEOMETRY"
    --skip_first_layers "$SKIP_FIRST_LAYERS"
    --baseline_window "$BASELINE_WINDOW"
    --spike_k "$SPIKE_K"
    --spike_lower "$SPIKE_LOWER"
    --warmup_frames "$WARMUP_FRAMES"
    --use_faiss "$USE_FAISS"
    --faiss_nprobe "$FAISS_NPROBE"
    --device "$DEVICE"
)

cd "$REPO_ROOT"
python3 -m grid_diff_tcn.simpen_infer "${ARGS[@]}"
