# Grid-Diff 1D-TCN：激光钻孔穿透检测

基于 **层内融合 + 帧间差分 + 8×8 网格池化** 的 1D 时序卷积网络，用于激光钻孔**穿透层定位**：输入单孔逐层图像序列，输出是否穿透及穿透层号。

---

## 算法流程概览

1. **层内融合**：同层多张图求均值 → 每层一张代表图 \(I_n\)  
2. **帧间绝对差分**：\(D_n = |I_n - I_{n-1}|\)  
3. **8×8 网格池化**：对 \(D_n\) 的 ROI 划分 64 个 patch，每 patch 求均值 → 每层 64 维向量  
4. **因果 TCN 前端**：对序列 `[Seq_Len, 64]` 做因果膨胀卷积，得到逐层时序特征  
5. **（可选）Transformer 编码器**：在 TCN 特征上叠加概率注意力 Transformer，建模长程依赖，并支持 MC 采样得到不确定性  
6. **输出头**：每时间步输出 2 类 logits（未穿透/穿透）  
7. **训练**：按「穿透层」做逐层标签，Focal Loss + 辅助定位损失 +（可选）KL 正则  
8. **推理**：物理安全锁（前 30 层置 0）+ 决策（TopKMedian 或 S3WD）；TopKMedian 可选不确定性门控  

---

## 目录结构

**详细代码与结果索引见 [CODE_INDEX.md](CODE_INDEX.md)。** 每个 .py 文件开头也有功能说明。

| 文件 | 说明 |
|------|------|
| `dataset.py` | 数据集：Grid-Diff 预处理、层内融合、差分、8×8 网格，输出 `[Seq_Len, 64]` |
| `tcn_model.py` | 模型：因果 1D-TCN，输入 `(B, 64, T)`，输出 `(B, 2, T)` |
| `transformer_tcn_model.py` | 模型：TCN + Transformer（概率注意力 K/V 重参数化），可选 KL 正则与 MC 不确定性采样 |
| `train.py` | 训练：窗口采样、Focal/CE、定位损失、平衡 batch、预计算缓存 |
| `inference.py` | 推理：整孔前向 + 安全锁 +（S3WD 或 TopKMedian + 可选不确定性门控），输出穿透层或未穿透 |
| `precompute_features.py` | 预计算每孔 `[Seq_Len, 64]` 到磁盘，训练时用 `--precomputed_dir` 加速 |
| `split_train_test.py` | 将 `samples_info.json` 按比例划分为训练集/测试集 |
| `infer_missing_json_holes.py` | 对「无 JSON 标注」的孔目录做推理（有图无标签） |
| `link_cache_features_by_name.py` | 为预计算文件建立按样本名命名的硬链接/副本 |
| `visualize_roi_simple.py` | ROI 裁剪与网格可视化 |
| `train.sh` | 训练命令脚本（默认：Transformer + 验证不确定性门控；可改） |
| `inference_train.sh` | 训练集推理脚本（强制使用预计算目录） |

数据与配置约定：

- 样本列表：`../data_drilling/samples_info.json`（或 `samples_info_train.json` / `samples_info_test.json`）  
- 每条需包含：`sample_path`（孔目录，内含 `*.jpg`）、`is_penetrated`、`penetration_layer` 等  

---

## 环境与依赖

- Python 3.7+
- PyTorch
- OpenCV 或 PIL
- NumPy
- 可选：`tqdm`

---

## 快速开始

### 1. 划分训练集 / 测试集

```bash
cd grid_diff_tcn
python3 split_train_test.py --samples_info ../data_drilling/samples_info.json --out_dir ../data_drilling --ratio 0.8 --seed 42
```

会生成 `../data_drilling/samples_info_train.json` 与 `samples_info_test.json`。

### 2. 预计算特征（推荐，显著加速训练）

按**训练集**预计算，并与训练参数一致（如 `img_size=128`、`roi_size=96`）：

```bash
python3 precompute_features.py \
  --samples_info ../data_drilling/samples_info_train.json \
  --out_dir ./cache_features_train \
  --img_size 128 --roi_size 96 \
  --load_workers 6 \
  --by_name
```

`--by_name` 会按样本文件夹名保存为 `xxx.pt`（如 `10-10_2024_09_02_16_17_40_139.pt`），训练时会自动按名加载。

---

## TCN 与 Transformer 如何协同（架构说明）

本项目有两种模型形态：

### A. 基础 `GridDiffTCN`（纯 TCN）

- **输入**：整孔特征序列 `X`，形状 `(B, 64, T)`
- **主干**：多层因果膨胀卷积 `TCNBlock`，得到时序特征 `(B, C, T)`
- **头部**：`1×1 Conv` 映射为逐层二分类 logits `(B, 2, T)`

用一句话概括：**TCN 负责“局部/中程”的时序模式提取**，并保持因果性（不看未来）。

### B. `GridDiffTCNWithTransformer`（TCN + Transformer）

在纯 TCN 的基础上增加 Transformer 编码器（见 `transformer_tcn_model.py`）：

1. **TCN 前端**：与基础 TCN 相同，得到 `(B, C_mid, T)`
2. **投影到 Transformer 维度**：`proj_in: Conv1d(C_mid → d_model)`，并转置为 `(B, T, d_model)`
3. **Transformer 编码器堆叠**：若干层 `ProbTransformerEncoderLayer`
4. **输出头**：转回 `(B, d_model, T)`，再用 `Conv1d(d_model → 2)` 输出 logits

其中 Transformer 的自注意力是“概率注意力”：

- 线性层输出 `q/k/v` 的 **均值**与 **log 方差**（`mu_*`, `logvar_*`）
- 仅对 `K/V` 做重参数化采样：`K = mu_K + exp(0.5 logvar_K) * eps`（`V` 同理）
- **训练时**默认采样；**推理时**默认用均值（不采样，保证稳定与兼容）
- 若要估计不确定性，可在推理中做多次前向并强制采样（MC）

用一句话概括：**TCN 先把 64 维“物理特征”变成更可用的时序表征，Transformer 再在其上做“长程依赖建模”与（可选）不确定性估计**。

---

## 训练（推荐用 `train.sh`）

`train.sh` 已把常用训练命令写好，并支持用环境变量覆盖参数：

```bash
cd /home/student2025/wudf2025/dinov3-main/grid_diff_tcn
./train.sh
```

例如：

```bash
EPOCHS=30 SAVE=grid_diff_transformer_v2.pt ./train.sh
```

若你不想用脚本，也可直接运行 `train.py`（见下）。

### 3.1 直接运行 `train.py`（示例）

```bash
python3 train.py \
  --samples_info ../data_drilling/samples_info_train.json \
  --precomputed_dir ./cache_features_train \
  --num_workers 4 \
  --save grid_diff_tcn.pt
```

常用参数：

- `--window_len`：窗口长度（默认 60 层）  
- `--penetration_radius`：穿透层前后多少层也标 1（软标签，默认 2）  
- `--pos_per_batch`：每 batch 最少正样本数（默认 `batch_size//2`），用于平衡采样  
- `--loc_loss_weight`：辅助定位损失权重（默认 0.5）  
- `--no_focal`：关闭 Focal Loss，改用加权交叉熵  

不使用预计算时去掉 `--precomputed_dir` 即可（会从图像实时计算，较慢）。

---

## 推理（TopKMedian / S3WD）

### 4.1 测试集推理（推荐 TopKMedian，与训练验证一致）

> 训练验证默认用 TopKMedian（`k=9, min_thresh=0.4`），因此测试推理建议也显式开启 `--topkmedian`。

```bash
python3 inference.py \
  --ckpt grid_diff_tcn.pt \
  --samples_info ../data_drilling/samples_info_test.json \
  --output inference_results_test.json \
  --topkmedian --k 9 --min_thresh 0.4
```

Transformer 权重推理需要加 `--use_transformer`（并与训练时层数/维度一致）：

```bash
python3 inference.py \
  --ckpt grid_diff_tcn_transformer.pt \
  --samples_info ../data_drilling/samples_info_test.json \
  --output inference_results_test.json \
  --use_transformer --num_transformer_layers 2 --attn_dim 64 --num_heads 4 \
  --topkmedian --k 9 --min_thresh 0.4
```

### 4.2 不确定性（可选，多采样 + 门控）

```bash
python3 inference.py \
  --ckpt grid_diff_tcn_transformer.pt \
  --samples_info ../data_drilling/samples_info_test.json \
  --output inference_results_test.json \
  --use_transformer \
  --topkmedian --k 9 --min_thresh 0.4 \
  --unc_samples 8 --unc_gate --unc_var_median_thresh 0.1
```

含义：
- `--unc_samples`: 多次前向次数（>1 才会计算方差；对纯 TCN 多次前向结果相同，方差≈0）
- `--unc_gate`: 开启门控（TopKMedian 判穿透后，如果 top-k 位置方差中位数过大则否决穿透）
- `--unc_var_median_thresh`: 方差中位数阈值

### 4.3 训练集推理（必须用预计算特征）

训练集通常很大，建议用 `inference_train.sh`，脚本会强制传入 `--precomputed_dir`：

```bash
cd /home/student2025/wudf2025/dinov3-main/grid_diff_tcn
./inference_train.sh
```

可用环境变量改权重或输出：

```bash
CKPT=grid_diff_transformer_v2.pt OUTPUT=inference_results_train_v2.json ./inference_train.sh
```

> 注意：`cache_features_train` 目录必须存在且包含 `.pt`。如果只有 `cache_features_train.zip`，先解压到该目录。

可选：`--base_dir`、`--device`、`--lock_layers`（安全锁层数，默认 30）等。

### 5. 对无标注孔推理

在 `data_drilling/train` 下找有图但无 `Auto.json`/`Project.json` 的孔，用已训练模型推理：

```bash
python3 infer_missing_json_holes.py --ckpt ./grid_diff_tcn.pt --train_root ../data_drilling/train --topk 5
```

---

## 模型与数据约定

- **输入**：单孔目录下按层组织的 `*.jpg`，文件名需能解析出层号（如 `xxx_帧号_层号.jpg`）。  
- **预处理**：缩放 → 中心/ROI 裁剪 → 层均图 → 帧间绝对差 → 8×8 网格均值 → `[Seq_Len, 64]`。  
- **训练**：在 `samples_info` 的每个孔上做窗口采样；正样本围绕真实穿透层截窗，负样本随机截窗；损失为逐层分类（Focal/CE）+ 可选穿透层回归（Smooth L1）。  
- **推理**：整孔前向得到每层穿透概率；前 30 层强制为 0（安全锁）；再用 S3WD 判定穿透层或未穿透。

---

## 训练超参数一览（`train.py`）

> 下表按 `train.py` 的命令行参数整理。默认值以当前代码为准（如果你改过脚本/参数，以实际运行命令为准）。

### 1) 数据与预处理

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--samples_info` | `../data_drilling/samples_info.json`（若不传） | 样本列表 JSON |
| `--base_dir` | `None` | 若 `sample_path` 是相对路径，用它拼接根目录 |
| `--img_size` | `128` | 读图缩放边长（正方形） |
| `--roi_size` | `None`（内部默认 `min(96,img_size)` 并对齐 8） | ROI 裁剪边长 |
| `--roi_cy` / `--roi_cx` | `0.5 / 0.5` | ROI 中心比例 |
| `--crop_mode` | `center` | `center`=严格中心裁剪；`roi`=按比例偏移裁剪 |
| `--load_workers` | `6` | 单孔内部多线程读图数（预热/读图提速） |

### 2) 窗口采样

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--window_len` | `60` | 训练窗口层数 |
| `--penetration_radius` | `2` | 穿透层±r 也标为正（软标签，缓解不平衡） |
| `--max_samples` | `None` | 最多使用多少孔（快速实验用） |
| `--precomputed_dir` | `.../grid_diff_tcn/cache_features_train` | 预计算特征目录（强烈推荐） |
| `--num_workers` | `4` | DataLoader 进程数（用预计算时建议 4） |

### 3) 训练优化

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | `50` | 训练轮数 |
| `--batch_size` | `16` | batch 大小 |
| `--lr` | `1e-4` | 学习率 |
| `--device` | `cuda`（若可用） | 设备 |
| `--no_amp` | `False` | 关闭 AMP（默认 CUDA 下启用 AMP） |

### 4) 不平衡与损失

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no_focal` | `False` | 关闭 Focal（默认启用） |
| `--focal_gamma` | `2.0` | Focal 的 gamma |
| `--focal_alpha` | `0.75` | Focal 正类 alpha |
| `--weight_pos` | `10.0` | 关闭 Focal 时用于加权 CE 的正类权重 |
| `--weight_neg` | `1.0` | （主要用于信息显示/保留接口） |
| `--subsample_neg` | `10` | 每样本参与损失的负时间步数（0=全部） |
| `--loc_loss_weight` | `0.5` | 辅助定位损失权重（0=不加） |

### 5) 采样/组 batch 方式

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--pos_per_batch` | `None`（内部默认 `batch_size//2`） | 平衡采样：每 batch 最少正样本数 |
| `--no_batch_by_hole` | `False` | 关闭按孔组 batch（默认同一 batch 来自同一孔） |
| `--pos_inject_ratio` | `0.2` | 按孔组 batch 时：对“被选中的全负 batch”注入穿透样本的比例 |
| `--neg_batch_inject_ratio` | `0.5` | 全负 batch 中有多少比例会被注入穿透样本 |
| `--simple` | `False` | 简单模式（混合/平衡采样；训练中不做每轮按孔验证） |

### 6) 验证（按孔 + TopKMedian）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--val_ratio` | `0.2` | 按孔划分 train/val 的比例 |
| `--val_seed` | `42` | 划分随机种子 |
| `--val_holes_per_epoch` | `200` | 每轮按孔验证抽样孔数（README 以代码默认为准） |
| `--val_prefetch_workers` | `8` | 按孔验证预取线程数 |
| `--val_unc_samples` | `1` | 验证时 TopKMedian 前向次数（>1 才有方差） |
| `--val_unc_gate` | `False` | 开启不确定性门控 |
| `--val_unc_var_median_thresh` | `0.05` | 门控阈值（方差中位数） |

### 7) Transformer（可选）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--use_transformer` | `False` | 启用 TCN+Transformer 模型 |
| `--num_transformer_layers` | `2` | Transformer 层数 |
| `--attn_dim` | `64` | Transformer \(d\_model\) |
| `--num_heads` | `4` | 多头数 |
| `--kl_weight` | `0.0` | 概率注意力 KL 正则权重（0=不加） |

# LaserDrilling
