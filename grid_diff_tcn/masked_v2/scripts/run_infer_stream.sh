#!/bin/bash
# ============================================================
# 流式推理脚本
# 用法: bash run_infer_stream.sh <sample_dir> [--extra-args ...]
# ============================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INFER_SCRIPT="${SCRIPT_DIR}/grid_diff_tcn/masked_v2/infer_stream.py"
STAGE2_CKPT="${SCRIPT_DIR}/grid_diff_tcn/masked_v2/checkpoints/stage2.pt"
STAGE1_CKPT="${SCRIPT_DIR}/grid_diff_tcn/masked_v2/checkpoints/stage1.pt"

# 默认参数
ROI_SIZE=224
MAX_FRAMES=8
LOCK_LAYERS=30
S3WD_WAIT=3
S3WD_THRESHOLD=0.6
S3WD_ACCEPT=0.7
DEVICE="cuda"
VERBOSE=""

# 解析参数
if [ $# -lt 1 ]; then
    echo "用法: bash run_infer_stream.sh <sample_dir> [--extra-args ...]"
    echo ""
    echo "示例:"
    echo "  bash run_infer_stream.sh /path/to/sample"
    echo "  bash run_infer_stream.sh /path/to/sample --verbose"
    echo "  bash run_infer_stream.sh /path/to/sample --dinov3_roi_size 224 --lock_layers 30"
    exit 1
fi

SAMPLE_DIR="$1"
shift

# 检查必要文件
if [ ! -f "${INFER_SCRIPT}" ]; then
    echo "[错误] 推理脚本不存在: ${INFER_SCRIPT}"
    exit 1
fi
if [ ! -f "${STAGE2_CKPT}" ]; then
    echo "[错误] Stage2 checkpoint 不存在: ${STAGE2_CKPT}"
    exit 1
fi
if [ ! -f "${STAGE1_CKPT}" ]; then
    echo "[错误] Stage1 checkpoint 不存在: ${STAGE1_CKPT}"
    exit 1
fi
if [ ! -d "${SAMPLE_DIR}" ]; then
    echo "[错误] 样本目录不存在: ${SAMPLE_DIR}"
    exit 1
fi

# 彩色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

echo -e "${BOLD}${CYAN}============================================${NC}"
echo -e "${BOLD}${CYAN}         流式推理 - DINOV3 GridDiffTCN       ${NC}"
echo -e "${BOLD}${CYAN}============================================${NC}"
echo ""
echo -e "${BLUE}样本路径:${NC}   ${SAMPLE_DIR}"
echo -e "${BLUE}Stage2:${NC}     ${STAGE2_CKPT}"
echo -e "${BLUE}Stage1:${NC}     ${STAGE1_CKPT}"
echo -e "${BLUE}ROI Size:${NC}   ${ROI_SIZE}"
echo -e "${BLUE}Device:${NC}     ${DEVICE}"
echo ""

# 执行推理
python "${INFER_SCRIPT}" \
    --stage2_checkpoint "${STAGE2_CKPT}" \
    --stage1_checkpoint "${STAGE1_CKPT}" \
    --sample_dir "${SAMPLE_DIR}" \
    --dinov3_roi_size ${ROI_SIZE} \
    --max_frames_per_layer ${MAX_FRAMES} \
    --lock_layers ${LOCK_LAYERS} \
    --s3wd_wait ${S3WD_WAIT} \
    --s3wd_threshold ${S3WD_THRESHOLD} \
    --s3wd_accept ${S3WD_ACCEPT} \
    --device ${DEVICE} \
    --early_stop \
    "$@"

EXIT_CODE=$?

echo ""
if [ ${EXIT_CODE} -eq 0 ]; then
    echo -e "${GREEN}${BOLD}[完成] 推理成功${NC}"
else
    echo -e "${RED}${BOLD}[失败] 推理失败，退出码: ${EXIT_CODE}${NC}"
fi

exit ${EXIT_CODE}
