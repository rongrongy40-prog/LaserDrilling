#!/bin/bash
# =============================================================================
# 下载 inference_results_test.csv 中误差 >= 10 层的高误差样本（按孔目录去重）
#
# 两种运行模式：
#   模式 A（本地运行 -> 从远程拉取）：
#     填写 REMOTE_USER / REMOTE_HOST，脚本通过 rsync/scp 从远程下载到本地
#   模式 B（直接在远程运行）：
#     设置 RUN_MODE="remote"，脚本在远程本机直接 cp 到本地下载目录
# =============================================================================
#
# 用法:
#   chmod +x download_high_error.sh
#   # 方式 A（从本机拉远程）:
#   ./download_high_error.sh
#   # 方式 B（直接在远程本机执行）:
#   RUN_MODE="remote" ./download_high_error.sh
# =============================================================================

# ---------------------------------------------------------------------------
# 模式选择
# ---------------------------------------------------------------------------
RUN_MODE="${RUN_MODE:-local}"   # "local" 或 "remote"

# ---------------------------------------------------------------------------
# 远程服务器配置（RUN_MODE="local" 时使用）
# ---------------------------------------------------------------------------
REMOTE_USER="wudf2025"          # 远程服务器用户名
REMOTE_HOST="210.26.51.126"          # 远程服务器地址
REMOTE_PORT="9172"          # SSH 端口（留空默认为 22）
SSH_OPTS="-o StrictHostKeyChecking=no"

# ---------------------------------------------------------------------------
# 本地下载目录
# ---------------------------------------------------------------------------
LOCAL_DOWNLOAD_DIR="D:\project\laser_drilling"

# ---------------------------------------------------------------------------
# 高误差孔目录列表（error >= 10，去重）
# ---------------------------------------------------------------------------
declare -A HOLES
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/1/24060808-b/11-3_2024_12_01_10_18_55_620"]="10|存疑"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/2/JA1ES25040403(3-8)/6-7_2025_06_26_20_28_35_430"]="10|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/3/HF-35#/12-5_2024_12_13_12_11_23_146"]="10|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/1/3D-8-7/11-6_2024_08_07_23_40_24_824"]="11|标注无误"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/2/30zs/6-8_2024_11_25_11_56_40_508"]="11|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/1/27-0-b/7-9_2024_12_18_23_30_20_020"]="13|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/3/3d-1/12-8_2024_11_11_10_30_48_836"]="13|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/4/3d-1/12-8_2024_11_11_10_30_48_836"]="13|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/1/27-0-b/6-10_2024_12_18_23_10_30_491"]="14|存疑"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/4/11pai/11-5_2025_03_27_16_10_31_336"]="14|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/4/3d-1/12-12_2024_11_11_10_44_55_539"]="14|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/1/3D-8-7/6-9_2024_08_07_17_09_33_776"]="15|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/2/30-20250430/6-9_2025_04_30_04_32_49_874"]="15|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/3/35#chuantou2025.3.11/13-8_2025_03_03_22_43_40_197"]="16|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/2/25060503-3-8/6-2_2025_08_12_16_51_36_653"]="18|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/2/试刀件/7-9_2024_11_25_13_33_00_079"]="19|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/3/35#chuantou2025.3.11/13-5_2025_03_03_22_36_37_694"]="24|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/2/25061201-3-8/6-4_2025_08_14_22_24_53_523"]="26|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/2/25060503-3-8/6-10_2025_08_12_19_11_12_635"]="29|"
HOLES["/home/student2025/wudf2025/dinov3-main/data_drilling/train/1/24081304/9-11_2025_01_27_02_11_08_768"]="49|标注错误"

# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------
mkdir -p "$LOCAL_DOWNLOAD_DIR"

TOTAL=${#HOLES[@]}
echo "========================================"
echo " 高误差样本下载脚本（error >= 10）"
echo " 模式: $([ "$RUN_MODE" = "remote" ] && echo "B - 直接在远程运行" || echo "A - 从本机拉取远程")"
echo " 下载目录: $(pwd)/${LOCAL_DOWNLOAD_DIR}"
echo " 孔数量:   ${TOTAL}"
echo "========================================"
echo

count=0
for remote_path in "${!HOLES[@]}"; do
    count=$((count + 1))
    info="${HOLES[$remote_path]}"
    error_val="${info%%|*}"
    note="${info#*|}"

    hole_name=$(basename "$remote_path")
    local_subdir="${LOCAL_DOWNLOAD_DIR}/${hole_name}"

    echo "[${count}/${TOTAL}] error=${error_val}  note=[${note}]"
    echo "  源路径: ${remote_path}"
    echo "  目标:   ${local_subdir}"

    # 已存在则跳过
    if [[ -d "$local_subdir" ]]; then
        echo "  -> 已存在，跳过"
        echo
        continue
    fi

    if [[ "$RUN_MODE" == "remote" ]]; then
        # 模式 B：直接在远程本机 cp
        cp -r "$remote_path" "$LOCAL_DOWNLOAD_DIR/"
    else
        # 模式 A：从本机通过 scp/rsync 拉取远程
        if [[ -z "$REMOTE_USER" ]] || [[ -z "$REMOTE_HOST" ]]; then
            echo "  -> ERROR: 请填写 REMOTE_USER 和 REMOTE_HOST"
            echo
            continue
        fi
        scp_cmd="scp -r ${SSH_OPTS}"
        if [[ -n "$REMOTE_PORT" ]]; then
            scp_cmd="scp -P ${REMOTE_PORT} ${SSH_OPTS}"
        fi
        ${scp_cmd} "${REMOTE_USER}@${REMOTE_HOST}:${remote_path}/" "${LOCAL_DOWNLOAD_DIR}/"
    fi

    if [[ $? -eq 0 ]]; then
        echo "  -> 完成"
    else
        echo "  -> 失败！"
    fi
    echo
done

echo "========================================"
echo " 完成！样本目录: $(pwd)/${LOCAL_DOWNLOAD_DIR}/"
echo "========================================"
