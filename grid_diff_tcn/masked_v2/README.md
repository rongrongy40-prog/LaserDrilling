# masked_v2: MAE Pretraining + Classification Fine-tuning

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                          masked_v2 完整训练流程                              │
└─────────────────────────────────────────────────────────────────────────────┘

Stage 1 — MAE 预训练（encoder + MAE decoder 联合训练，encoder 学习钻孔领域特征）
────────────────────────────────────────────────────────────────────────────
原始 ROI 图像 ──→ [DINOv3 Encoder (可训练)] ──→ 全部 patch tokens
                                                    │
                                            随机遮蔽 75%
                                                    ↓
                                      MAE Decoder ──→ 重建像素
                                                    ↑
                                        L1 重建损失
                                                    │
                                          梯度回传 → encoder + decoder

Stage 2 — 分类微调（encoder 冻结，提取 Stage 1 领域适配后的 encoder 特征）
────────────────────────────────────────────────────────────────────────────
领域适配后的 DINOv3 Encoder ──→ 384-dim 特征 ──→ 分类器

推理：argmax(prob) + S3WD 后处理
────────────────────────────────────────────────────────────────────────────
每帧渗漏概率 softmax[:,1] ──→ 找连续 N 帧 >= 阈值的位置 ──→ 映射到物理层号
                              （或 TemporalDecisionHead 直接输出）
```

## 目录

- [概述](#概述)
- [文件结构](#文件结构)
- [两阶段训练](#两阶段训练)
- [损失函数详解](#损失函数详解)
- [决策方法](#决策方法)
- [推理脚本](#推理脚本)
- [快速开始](#快速开始)
- [关键超参数](#关键超参数)
- [实验结果](#实验结果)

---

## 概述

`masked_v2` 是一套两阶段钻孔图像建模方案，核心改进在于：

1. **Stage 1 采用 MAE**（encoder + decoder 联合训练），encoder 通过像素重建任务学习钻孔领域特征，decoder 学习重建被遮蔽的像素块。
2. **Stage 2 在领域适配特征上做分类**，配合多分量损失函数联合优化。
3. **三种数据加速机制**：预裁剪 ROI 缓存（`pre_crop.sh`）、预计算 DINOv3 特征（`extract_features.sh`）、训练集预加载（`--preload`）。

---

## 文件结构

```
grid_diff_tcn/masked_v2/
├── model.py          # MaskedPixelModel — 统一两阶段模型
├── train.py          # 训练入口：Stage 1 MAE + Stage 2 分类微调
│                      # 包含全部损失函数实现
├── extract.py        # 用 Stage 1 MAE decoder 提取领域适配特征（256-dim）
├── infer.py          # 完整推理脚本（含验证集网格搜索）
├── infer_simple.py   # 简化推理脚本（推荐）
├── pre_crop.py       # 预裁剪 ROI（固定 box 策略，生成 .pt 缓存）
├── dataset.py        # MaskedDrillingDataset / CropCacheDataset
├── masks.py          # CenterMask + MaskedImageModelingLoss
├── mae.py            # StandardMAEPreTrainer + MAEDecoder
├── decoder.py        # PixelDecoder（Stage 1 旧版，保留但不使用）
├── scripts/
│   ├── stage1.sh           # Stage 1 MAE 训练
│   ├── stage2.sh           # Stage 2 分类微调
│   ├── extract_features.sh  # 特征提取（配合 Stage 2 加速）
│   ├── pre_crop.sh         # ROI 预裁剪
│   ├── infer.sh            # 完整推理（含网格搜索）
│   └── infer_simple.sh      # 简化推理（推荐）
└── README.md
```

---

## 两阶段训练

### Stage 1：Standard MAE 预训练

```
bash grid_diff_tcn/masked_v2/scripts/stage1.sh
```

| 特点 | 说明 |
|------|------|
| Encoder | DINOv3（**可训练**），Stage 1 联合微调，学习钻孔领域视觉特征 |
| 可训练部分 | encoder (lr=1e-6) + MAE Decoder (lr=1e-4) |
| 训练目标 | 重建被随机遮蔽（75%）的 patch 像素 |
| 优化器 | AdamW，两个参数组分别设置学习率 |
| 调度器 | CosineAnnealingLR 或 ReduceLROnPlateau |
| 输出 | `checkpoints/stage1.pt` |

**为什么 encoder unfreeze（联合训练）？**
- 像素重建任务驱动 encoder 学习钻孔领域的视觉特征（纹理、深度、灰度分布等）
- encoder 和 decoder 协同优化，encoder 能根据重建反馈调整特征表达
- decoder 学会了从领域适配后的特征重建像素

**MAE Decoder 架构**（`mae.py`）：

```
输入 patch tokens (已替换 masked 位置为 [MASK] token)
    │
    ├── Linear: encoder_dim(384) → decoder_dim(256)
    ├── + 可学习 decoder_pos_embed
    ├── Transformer Decoder (4 层, 8 head, d_ff=1024)
    └── Linear → pixel_head: decoder_dim(256) → 16×16×3 (= 768)
                                          ↓ pixel_shuffle
                              (B, 3, H, W) 重建图像
```

每个图像被分成 `(224/16)² = 196` 个 patch，随机保留 25%（49 个），decoder 从 49 个可见 patch 重建全部 196 个 patch。

---

### Stage 2：分类微调

```
bash grid_diff_tcn/masked_v2/scripts/stage2.sh \
  --resume_from grid_diff_tcn/masked_v2/checkpoints/stage1.pt
```

| 特点 | 说明 |
|------|------|
| Encoder | DINOv3（**冻结**），使用 Stage 1 联合微调后的领域适配 encoder 特征（384-dim） |
| 可训练部分 | 默认仅 classifier；可选 `--unfreeze_encoder_stage2=True` 继续微调 encoder |
| 优化器 | classifier: AdamW lr=1e-4；encoder (可选): lr=1e-6 |
| 调度器 | ReduceLROnPlateau（监控 val_loss） |
| Early Stopping | 基于 `pct_within_3 + pct_within_5` 组合指标 |
| 输出 | `checkpoints/stage2.pt` |

**Stage 2 三种加速模式**（可选，层层递进）：

```
模式 A（最快）: 预裁剪 ROI (.pt) + 预计算 DINOv3 特征 (.pt)
模式 B（中等）: 预裁剪 ROI (.pt) + 在线 DINOv3 特征提取
模式 C（最慢）: MaskedDrillingDataset 在线 ROI 裁剪 + 在线特征提取
```

推荐使用模式 A（预裁剪 + 预计算特征），DataLoader batch 加载，无任何图像处理开销。

**进阶：继续微调 encoder**

```bash
UNFREEZE_ENCODER_STAGE2=True \
  bash grid_diff_tcn/masked_v2/scripts/stage2.sh \
    --resume_from grid_diff_tcn/masked_v2/checkpoints/stage1.pt
```

同时训练 encoder (lr=1e-6) 和 classifier (lr=1e-4)，风险是过拟合，仅在默认冻结效果不好时尝试。

---

## 损失函数详解

### Stage 1：L1 重建损失

```
L_mae = mean(|pred_pixels − target_pixels|)   仅在被遮蔽位置计算
```

`mae.py` 中，`forward()` 返回：
- `pred`: 重建图像 `(B, 3, H, W)`
- `target`: 原始图像 `(B, 3, H, W)`
- `mask_img`: `(B, 1, H, W)`，1=遮蔽，0=可见

```python
valid_count = mask_img.sum().clamp(min=1)
loss = (pred - images).abs() * mask_img
loss = loss.sum() / valid_count
```

仅在遮蔽位置计算，Visible 位置不参与梯度。L1 比 L2 对极端像素值更鲁棒。

---

### Stage 2：多分量损失函数

Stage 2 的总损失是多个分量的加权组合：

```
L_total = L_focal                 (主损失, 权重 1.0)
        + L_peak      × λ_peak    (高斯峰损失, 默认 0.15)
        + L_smooth   × λ_smooth  (时间平滑, 默认 0.05)
        + L_boundary × λ_boundary (边界感知, 默认 0.15)
        + L_decision × λ_index   (可选, BCE 决策, 默认 1.0)
```

所有分量均在 `train.py` 中实现，以下逐一说明。

---

#### 1. Focal Cross-Entropy（主损失）

**文件**: `train.py` — `focal_cross_entropy()`

**公式**:
```
L_focal = mean[ α_t × (1 − p_t)^γ × (−log p_t) ]
```

其中：
- \( p_t \): 模型对**真实标签类别**的预测概率（softmax 后取对应类）
- \( \alpha_t \): 类别权重，渗透类正样本提升权重
- \( \gamma = 2.0 \): Focal 系数，难分类样本（低 p_t）的损失被放大

**核心设计**:

```python
# 类别权重：weight_pos=3.0 → alpha_pos = 3.0/4.0 = 0.75
alpha_pos = weight_pos / (weight_pos + 1.0)   # 正类权重
alpha_neg = 1.0 / (weight_pos + 1.0)          # 负类权重
alpha_t = where(label == 1, alpha_pos, alpha_neg)
```

- `weight_pos=3.0`（默认值）：渗透帧的损失权重是未渗透帧的 3 倍
- `label_smoothing=0.05`：将硬标签 0/1 软化为 0.05/0.95，防止过拟合
- `lock_layers=30`：前 30 帧不参与 loss（安全锁，强制预测为未渗透）
- 因果约束：t=0 现在有来自 t+1 的 lookahead 信息，可以预测

**时间步加权**（`window_ce_weights`）:

```python
# 真值层 ±5 范围内的时间步权重翻倍（in_window_weight=2.0）
w[|t − true_layer| ≤ 5] = 2.0
```

鼓励模型在真值层附近表现精确，间接促进最终预测误差 ≤5。

---

#### 2. Gaussian Peak Loss（定位软约束）

**文件**: `train.py` — `gaussian_peak_loss()`

**公式**:
```
target_t = exp(−(t − μ)² / (2σ²)) × causal_mask
L_peak  = MSE(softmax(logits)[:,1], target)
```

- \( \mu \): 真值穿透层层号
- \( \sigma = 3.0 \): 高斯标准差，±3 层内都有较高目标值
- `causal=True`: 仅对 t ≥ μ 建立目标（渗透前无信号），避免矛盾监督

**作用**：
- 软约束真值层附近概率形成高斯峰
- 全程可导，与 Focal CE 梯度方向一致
- 与硬标签 CE 互补：CE 只监督类别，高斯峰鼓励概率分布形状

```python
# 与 focal CE 梯度方向一致：真实层附近概率 ↑，远离层 ↓
# σ=3.0 表示 ±3 层内都有显著目标值，不会过于尖锐
```

---

#### 3. Temporal Smoothness Loss（时间平滑）

**文件**: `train.py` — `temporal_smoothness_loss()`

**公式**:
```
L_smooth = mean[ (prob[t+1] − prob[t])² ]  , t = 1..T−1
```

- 计算相邻时间步渗透概率的一阶差分 L2 损失
- 仅在有渗透的样本上计算
- 鼓励预测概率随时间缓慢上升，避免剧烈跳变

```python
prob = softmax(logits)[:, 1]   # (B, T)
diff = prob[:, 1:] − prob[:, :-1]   # (B, T-1)
loss = (diff ** 2).mean()
```

**效果**：减少假阳性尖峰，预测更稳定。

---

#### 4. Boundary-Aware Loss（边界感知）

**文件**: `train.py` — `boundary_aware_loss()`

**公式**:
```
target_t = 1  if t ≥ μ  else  0
L_boundary = MSE(prob, target)   仅在有渗透样本上
```

- 渗透层之前目标=0（尚未穿透），之后目标=1（已穿透）
- 与 Gaussian Peak Loss 互补：
  - 高斯峰：软化真值层附近概率分布
  - 边界感知：明确区分"渗透前"和"渗透后"两个阶段
- 两者一起使用效果最好

---

#### 5. Decision BCE Loss（Learned Decision Head）

**文件**: `train.py` — `decision_bce_loss()`

**背景**：S3WD 需要人工调阈值/帧数。为消除这个 gap，设计了可学习的 `TemporalDecisionHead`。

**训练标签**：Causal Cumulative Label

```
真值穿透层号 = μ
causal_label[t] = 1  if t ≥ μ
                = 0  if t < μ
```

这与 S3WD 推理逻辑完全一致（S3WD 找第一个 prob>threshold 的位置 = 第一个 causal_label=1 的位置）。

**Head 架构**（`model.py` 中 `TemporalDecisionHead`）:

```
每个时间步 t 的输入：
  ├── prob_cummean[t]   = mean(prob_base[0..t])      ← 因果累积均值
  ├── prob_base[t]      = softmax(logits)[:,1][t]    ← 当前层概率
  ├── prob_future[t]    = prob_base[t+2]              ← lookahead
  ├── prob_cummean × prob_base                         ← 交互项
  ├── history_mean[t]   = mean(z[0..t])               ← Transformer 特征均值
  ├── current[t]        = z[t]                         ← 当前层特征
  └── future[t]         = z[t+2]                       ← lookahead 特征

拼接 → 2层MLP → sigmoid → 穿透概率
```

**推理**：找第一个 decision_prob > 0.5 的位置作为预测层。

**与 S3WD 对比**：

| | S3WD | LearnedDecision |
|---|---|---|
| 阈值 | 人工调 (threshold, wait) | 可学习 |
| 泛化性 | 仅当参数恰好适配数据时最优 | 自动学习最优决策边界 |
| 调参成本 | 需网格搜索 | 无需调参 |

---

#### 6. Within-5 Adaptive Loss（自适应跳过）

**文件**: `train.py` — `_compute_stage2_loss()` 中的 `achieved` 集合

**机制**：

```
每 epoch 维护 achieved 集合（用真实 dataset index 标识）：
  1. 前向传播得到预测层号 pred_idx
  2. 计算 |pred_idx − true_idx|
  3. 如果误差 ≤ within5_tolerance（默认 5）：
       → 该样本损失权重降为 within5_weight（默认 0.1）
       → 加入 achieved 集合（后续 batch 中持续降权）
     否则：
       → 该样本正常参与训练
  4. Epoch 结束时 achieved 集合清空，重新评估
```

**效果**：
- 快速收敛的样本被降低优先级
- 模型将更多梯度资源集中在"还没学会"的困难样本上
- 等效于隐式课程学习（curriculum learning）

---

#### 损失分量汇总

| 损失分量 | 符号 | 默认权重 | 作用 |
|----------|------|----------|------|
| Focal Cross-Entropy | \( L_{focal} \) | 1.0（主损失） | 逐帧二分类，带类权重和 Focal 调制 |
| Gaussian Peak | \( L_{peak} \) | 0.15 | 软约束真值层附近概率峰 |
| Temporal Smoothness | \( L_{smooth} \) | 0.05 | 惩罚相邻帧概率跳变 |
| Boundary-Aware | \( L_{boundary} \) | 0.15 | 明确区分渗透前/后阶段 |
| Decision BCE | \( L_{decision} \) | 1.0（可选） | 训练可学习决策头 |
| Within-5 Adaptive | — | 降权 0.1 | 降低已达标样本的梯度权重 |

**为什么需要这么多分量？**

钻孔穿透决策是一个**多尺度时序问题**：
- **帧级别**：需要精确到具体哪帧有渗漏信号（Focal CE）
- **峰值定位**：需要知道最可能的穿透层在哪个区间（Gaussian Peak）
- **时间连续性**：真实钻孔信号是连续渐进的（Smoothness + Boundary）
- **决策一致性**：训练和推理的决策逻辑需要对齐（Decision BCE 或 S3WD）

---

## 决策方法

### S3WD（Sliding Window with Delay）

**文件**: `train.py` — `s3wd_decision()`

```
算法：
1. 从前往后扫描所有帧
2. 找第一个满足以下条件的位置 ti：
      连续 wait 帧（默认 3）的 prob >= threshold（默认 0.6）
   若某帧 prob >= accept（默认 1.0，即禁用），立即决策
3. 将 ti 映射回物理层号：pred_layer = layer_list[ti]
4. 找不到则 fallback 到 argmax
```

**物理层映射**：每个样本有不同的物理层序列（如 [30, 31, 32, ...]），S3WD 输出的是时间步索引，需要映射回真实层号。

**三个关键参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `s3wd_wait` | 3 | 连续几帧超过阈值才确认决策 |
| `s3wd_threshold` | 0.6 | 帧级别渗透概率阈值 |
| `s3wd_accept` | 1.0 | 概率达到此值立即决策（1.0=禁用） |

**wait 的作用**：防止单帧噪声导致误判。钻孔穿透是持续过程，不会只有一帧高概率。

---

### TemporalDecisionHead（可学习决策）

推理时直接用 Head 输出：

```python
decision_probs: (B, T) — 每帧的穿透概率
pred_idx: (B,)          — 第一个 prob > 0.5 的位置

pred_layer = layer_list[round(pred_idx)]
```

无需调参，自动从数据学习最优决策边界。

---

## 推理脚本

### infer_simple.sh（推荐）

```bash
# 训练集
bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh train

# 测试集
bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh test

# 验证集
bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh val
```

**模式选择**：

| MODE | 说明 | 速度 |
|------|------|------|
| `cached`（默认） | 预裁剪 ROI + 预计算特征 | 最快 |
| `online` | 在线裁剪 + 在线 DINOv3 | 最灵活 |

**决策方法**：

```bash
DECISION=s3wd    bash ...  # argmax + S3WD 后处理（默认）
DECISION=learned bash ...  # TemporalDecisionHead 直接输出
```

**调试少量样本**：

```bash
MAX_SAMPLES=5 bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh test
```

### infer.sh（完整版，含验证集网格搜索）

```bash
# 验证集调参（网格搜索 S3WD 参数）
RUN_VAL=1 bash grid_diff_tcn/masked_v2/scripts/infer.sh test
```

自动在验证集上搜索最优 `wait`、`threshold`、`accept` 组合，输出到 `grid_search_results.json`。

---

## 快速开始

### 完整流程

```bash
# ── Step 1: 预裁剪 ROI ─────────────────────────────────────────
bash grid_diff_tcn/masked_v2/scripts/pre_crop.sh

# ── Step 2: Stage 1 MAE 预训练 ────────────────────────────────
bash grid_diff_tcn/masked_v2/scripts/stage1.sh

# ── Step 3: 特征提取（加速 Stage 2）────────────────────────────
bash grid_diff_tcn/masked_v2/scripts/extract_features.sh \
  --checkpoint grid_diff_tcn/masked_v2/checkpoints/stage1.pt

# ── Step 4: Stage 2 分类微调 ──────────────────────────────────
RESUME_FROM=grid_diff_tcn/masked_v2/checkpoints/stage1.pt \
PRECOMPUTED_DIR=grid_diff_tcn/masked_v2/features_cache \
CROP_CACHE_DIR=data_drilling/roi_cache \
  bash grid_diff_tcn/masked_v2/scripts/stage2.sh

# ── Step 5: 推理 ──────────────────────────────────────────────
bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh test
```

### 参数覆盖示例

```bash
# 快速验证流程（少量样本）
MAX_SAMPLES=10 bash grid_diff_tcn/masked_v2/scripts/stage1.sh
MAX_SAMPLES=10 bash grid_diff_tcn/masked_v2/scripts/stage2.sh \
  --resume_from grid_diff_tcn/masked_v2/checkpoints/stage1.pt

# 使用在线模式（不依赖预裁剪）
MODE=online bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh test

# CPU 推理
FORCE_CPU=1 bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh test

# 可学习决策头（无需调 S3WD 参数）
DECISION=learned bash grid_diff_tcn/masked_v2/scripts/infer_simple.sh test

# 继续微调 encoder（高风险高回报）
UNFREEZE_ENCODER_STAGE2=True bash grid_diff_tcn/masked_v2/scripts/stage2.sh \
  --resume_from grid_diff_tcn/masked_v2/checkpoints/stage1.pt
```

### 断点续传

- `pre_crop.sh`：已存在的 .pt 文件自动跳过，`OVERWRITE=1` 强制重新生成
- `stage1.sh / stage2.sh`：`--resume_from` 参数加载已有 checkpoint 继续训练
- `extract_features.sh`：已提取的特征自动跳过，仅处理缺失文件

---

## 关键超参数

### Stage 1

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mask_ratio` | 0.75 | 随机遮蔽 75% 的 patch |
| `mae_decoder_dim` | 256 | MAE decoder 隐层维度 |
| `mae_decoder_depth` | 4 | MAE decoder Transformer 层数 |
| `mae_decoder_heads` | 8 | MAE decoder 注意力头数 |
| `lr` | 1e-4 | MAE decoder 学习率 |
| `batch_size` | 2 | |
| `accum_steps` | 4 | 梯度累积（有效 batch=8） |
| `DINOV3_ROI_SIZE` | 128 | Stage 1 ROI 尺寸 |
| `MAX_FRAMES_PER_LAYER` | 15 | 每层最多帧数 |
| `stage1_scheduler` | cosine | CosineAnnealingLR |

### Stage 2

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `weight_pos` | 3.0 | 渗透帧 Focal CE 权重 = 3.0/(3.0+1.0) = 0.75 |
| `label_smoothing` | 0.05 | 标签平滑因子 |
| `peak_loss_weight` | 0.15 | 高斯峰损失权重 |
| `smoothness_weight` | 0.05 | 时间平滑损失权重 |
| `boundary_weight` | 0.15 | 边界感知损失权重 |
| `index_loss_weight` | 1.0 | Decision BCE 损失权重（可选） |
| `within5_tolerance` | 3 | 自适应损失容差（0=禁用） |
| `within5_weight` | 0.1 | 已达标样本的损失权重 |
| `lock_layers` | 30 | 前 N 层强制预测=0（安全锁） |
| `lr` | 1e-4 | classifier 学习率 |
| `encoder_lr_stage2` | 1e-4 | encoder 学习率（仅 unfreeze 时） |
| `batch_size` | 16 | |
| `accum_steps_stage2` | 4 | 梯度累积 |
| `DINOV3_ROI_SIZE` | 224 | Stage 2 ROI 尺寸 |
| `MAX_FRAMES_PER_LAYER` | 12 | |
| `stage2_scheduler` | plateau | ReduceLROnPlateau |

### 推理（S3WD 参数）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `s3wd_wait` | 3 | 连续帧数要求 |
| `s3wd_threshold` | 0.6 | 概率阈值 |
| `s3wd_accept` | 0.7 | 立即决策阈值（0.7 时提前决策） |

---

## 实验结果

### 当前最优结果

```
训练集  (n=2094)
  pct_within_3: 79.8%
  pct_within_5: 89.2%
  pct_over_10:  3.9%

验证集  (n=443)
  pct_within_3: 79.5%
  pct_within_5: 88.3%
  pct_over_10:  5.2%

测试集  (n=444)
  pct_within_3: 73.6%
  pct_within_5: 85.6%
  pct_over_10:  6.3%
```

### 指标说明

- **pct_within_3**：预测层号与真实穿透层号误差 ≤3 的样本比例
- **pct_within_5**：误差 ≤5 的样本比例（业务核心指标）
- **pct_over_10**：误差 >10 的样本比例（严重错误）

---

## 常见问题

**Q: Stage 1 显存不够？**
降低 `DINOV3_CHUNK_SIZE`（每次送入 DINOv3 的图像数量，默认 256），或减小 `batch_size`。

**Q: Stage 2 过拟合？**
- 增大 `label_smoothing`（0.05 → 0.1）
- 减小 `weight_pos`（3.0 → 1.0）
- 启用 `UNFREEZE_ENCODER_STAGE2=False`（确认 encoder 已冻结）

**Q: S3WD 预测全是 argmax fallback？**
检查 `lock_layers` 是否覆盖了真实穿透层（真实穿透层需在 lock_layers 之后）；检查 `s3wd_threshold` 是否过高（0.6 可能太保守，尝试 0.4）。

**Q: cache 目录权限问题？**
`features_cache` 目录属于其他用户时无法写入，让目录所有者执行 `chmod -R 777 features_cache/`。
