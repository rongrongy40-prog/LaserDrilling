# grid_diff_tcn 项目详细流程说明

## 一、项目概述

本项目实现了一个**分层网格差异检测（Hierarchical Grid Difference Detection）** 系统，用于从钻孔序列图像中检测孔是否穿透以及预测穿透层号。项目采用两级时序建模架构：

- **帧级（Frame Level）**：对每层的帧序列进行特征提取和融合
- **层间级（Layer Level）**：对不同层的特征序列进行时序建模，预测每层的穿透概率

核心模型：`HierarchicalGridDiffProbTransformer`，结合了：

- TCN（Temporal Convolutional Network）进行时序卷积
- Probabilistic Transformer（概率Transformer）建模不确定性
- GRU + Attention Pooling 进行帧级特征融合
- 多尺度特征融合（可选）

**特征提取支持两种模式**：

1. **手工8x8网格特征**：基于ROI区域的8×8网格划分，提取mean/std/max统计量（192维）
2. **DINOv3 ViT特征**：使用预训练的DINOv3视觉Transformer提取768维CLS token特征（推荐）

---

## 二、完整流程架构图

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          数据准备阶段                                    │
│  samples_info_train.json / samples_info_test.json                       │
│  (包含孔的样本路径、穿透标签、穿透层号)                                    │
└──────────────────────────────┬──────────────────────────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │   ROI 检测与裁剪      │
                    │  color_cc_extract   │
                    │  _color_cc_resolve_box│
                    └──────────┬──────────┘
                               │
          ┌─────────────────────┴─────────────────────┐
          │                                           │
┌─────────▼─────────────┐              ┌──────────────▼──────────────┐
│  手工特征模式          │              │  DINOv3特征模式              │
│  8x8网格划分           │              │  DINOv3 ViT特征提取        │
│  (mean+std+max)       │              │  (CLS token, 768-dim)       │
│  192维                │              │  (推荐)                     │
└─────────┬─────────────┘              └──────────────┬──────────────┘
          │                                        │
┌─────────▼──────────────────┐      ┌──────────────▼──────────────┐
│  precompute.py              │      │  dinov3_precompute.py       │
│  手工特征预计算              │      │  DINOv3特征预计算            │
│  输出.cache_hierarchical_   │      │  输出.cache_dinov3_features │
│  features/*.pt              │      │  _vits/*.pt                  │
└─────────┬──────────────────┘      └──────────────┬──────────────┘
          │                                        │
          └──────────────────┬─────────────────────┘
                             │
                    ┌────────▼────────┐
                    │   训练阶段       │
                    │  train.py       │
                    └────────┬────────┘
                             │
          ┌───────────────────┼───────────────────┐
          │                   │                    │
┌─────────▼────────────┐  ┌───▼────────────┐  ┌──▼──────────────┐
│ 帧级特征编码器        │  │  层间TCN建模   │  │ 概率Transformer  │
│ FrameLevelTCNWithAttn│  │  Layer TCN     │  │ ProbTransformer  │
│ GRU + Attn Pool      │  │                │  │ (建模不确定性)   │
│ 多尺度融合(可选)      │  │                │  │                  │
└─────────┬────────────┘  └───┬────────────┘  └──┬──────────────┘
          │                    │                   │
          └────────────────────┼───────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  输出: 穿透概率       │
                    │  (B, 2, T) logits   │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  决策函数             │
                    │  topkmedian_decide  │
                    │  s3wd_decide        │
                    │  apply_safety_lock  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │  输出结果             │
                    │  是否穿透 + 穿透层号 │
                    │  inference_results  │
                    │  _hierarchical.json │
                    └─────────────────────┘
```

---

## 三、数据准备

### 3.1 样本信息文件

项目使用JSON格式的样本信息文件（`samples_info_train.json` / `samples_info_test.json`），结构如下：

```json
{
  "Categories": [
    {
      "sample_path": "path/to/hole_sample_folder",
      "is_penetrated": 1,
      "penetration_layer": 45
    },
    ...
  ]
}
```

每个样本包含：

- `sample_path`：孔的文件夹路径（内含多张`.jpg`图像）
- `is_penetrated`：是否穿透（0或1）
- `penetration_layer`：穿透发生的层号

### 3.2 排除列表

可通过`--exclude_json`参数排除特定图像：

```bash
--exclude_json data_drilling/no_laser_change_equalbox_full_mad00005_center_and_below.json
```

---

## 四、ROI检测与图像处理

### 4.1 ROI检测流程（image_ops.py）

```python
# 核心流程
图像加载 → 颜色连通域检测 → ROI框计算 → letterbox缩放 → 归一化
```

**关键参数**：

- `roi_size`：ROI目标尺寸（默认224 for DINOv3, 96 for 手工特征）
- `final_roi_scale`：ROI占检测框的比例（默认0.85）
- `cc_min_area`：连通域最小面积（默认12）
- `cc_expand_ratio`：检测框扩展比例（默认0.2）
- `roi_window_side`：固定窗口边长（默认32，0=不固定）

### 4.2 图像处理函数


| 函数                                    | 功能                    |
| ------------------------------------- | --------------------- |
| `load_image_as_float()`               | 加载图像为float32 [0,1]    |
| `to_grayscale()`                      | RGB转灰度图               |
| `color_cc_extract_gray_letterbox()`   | 颜色连通域+灰度letterbox ROI |
| `_color_cc_resolve_box()`             | 计算ROI边界框              |
| `_crop()` / `_resize_rgb_letterbox()` | 裁剪和缩放                 |


---

## 五、特征提取

### 5.1 手工8x8网格特征（默认模式）

**流程**：

1. 对ROI区域划分8×8网格
2. 每个格子提取3个统计量：`mean`, `std`, `max`
3. 最终特征维度：8×8×3 = **192维**

**特征池化**：

```python
_pool_stats = ("mean", "std", "max")  # 每个格子3个统计量
_feat_dim = 8 * 8 * 3 = 192  # 总维度
```

### 5.2 DINOv3 ViT特征（推荐模式）

**支持的模型**：


| 模型名称            | 特征维度 | 参数量  |
| --------------- | ---- | ---- |
| `vit_small`     | 384  | 22M  |
| `vit_base`      | 768  | 86M  |
| `vit_large`     | 1024 | 300M |
| `vit_huge_plus` | 1280 | 600M |


**DINOv3特征提取器**（`dinov3_features.py`）：

- 输入：ROI图像（3×H×W, [0,1]）
- 输出：CLS token特征（768维 for vit_base）
- 预处理：ImageNet归一化 + resize到目标尺寸

**DINOv3数据集类**（`dinov3_dataset.py`）：

- 封装DINOv3FeatureExtractor
- 支持预计算缓存（precomputed_dir）
- 在线提取时使用批处理

---

## 六、预计算阶段

### 6.1 手工特征预计算

```bash
python -m grid_diff_tcn.hier.precompute \
  --samples_info data_drilling/samples_info_train.json \
  --out_dir grid_diff_tcn/cache_hierarchical_features \
  --num_workers 8
```

**输出**：每个样本一个`.pt`文件，包含：

```python
{
    "frame_data": (T, F, 192),    # 特征张量
    "frame_mask": (T, F),         # 有效帧掩码
    "seq_label": (T,),            # 序列标签
    "label": int,                 # 穿透标签
    "penetration_layer": int,     # 穿透层号
    "layer_list": [int],          # 层列表
    "sample_path": str            # 样本路径
}
```

### 6.2 DINOv3特征预计算

```bash
python -m grid_diff_tcn.hier.dinov3_precompute \
  --samples_info data_drilling/samples_info_train.json \
  --out_dir grid_diff_tcn/cache_dinov3_features_vits \
  --dinov3_model vit_small \
  --dinov3_feat_dim 384 \
  --roi_size 224 \
  --num_workers 8 \
  --device cuda
```

**输出**：与手工特征格式相同，但`frame_data`维度为768（vit_base）或384（vit_small）

### 6.3 多进程预计算

支持`--num_workers > 0`使用多进程池加速：

- 每个worker进程独立加载模型
- 使用`spawn`上下文避免共享状态问题
- 支持跳过已计算文件

---

## 七、模型架构详解

### 7.1 整体结构

```
HierarchicalGridDiffProbTransformer
├── 帧级编码器（FrameEncoder）
│   ├── MultiScaleFrameEncoder（多尺度，默认）
│   └── FrameLevelTCNWithAttn（可选）
│       ├── TCN Blocks
│       ├── GRU（可选）
│       └── Attention Pooling（可选）
├── 额外特征投影（ExtraFeatureProj，可选）
├── 层间TCN（LayerTCN）
├── 概率Transformer层（ProbTransformer）
└── 输出头（Conv1d → 2类）
```

### 7.2 帧级编码器

**MultiScaleFrameEncoder**（默认启用）：

```python
# 三路卷积分支
conv1: Conv1d(768→128, kernel=3, padding=1)     # 局部特征
conv2: Conv1d(768→128, kernel=3, padding=2, dilation=2)  # 中程
conv3: Conv1d(768→128, kernel=3, padding=4, dilation=4)  # 长程

# 融合 + Attention Pooling
fusion: Linear(384, 128)
```

**FrameLevelTCNWithAttn**（可选）：

```python
# TCN + GRU + Attention Pooling
TCN Blocks: dilation=1, 2
GRU: bidirectional=False, layers=1
Attention Pooling: Query来自可学习参数
```

### 7.3 层间TCN

```python
LayerTCN Blocks:
├── TCNBlock(ch_in→ch_out, kernel=3, dilation=1)
├── TCNBlock(ch_in→ch_out, kernel=3, dilation=2)
└── TCNBlock(ch_in→ch_out, kernel=3, dilation=4)

残差连接：输入channels不匹配时用1×1卷积对齐
```

### 7.4 概率Transformer

**核心创新**：对K/V使用高斯分布参数化，通过重参数化采样引入不确定性

```python
ProbTransformerEncoderLayer:
├── ProbabilisticSelfAttention
│   ├── Q/K/V投影到6D（q_mu, q_logvar, k_mu, k_logvar, v_mu, v_logvar）
│   ├── 重参数化采样K/V（训练时+force_sample时）
│   ├── KL散度损失（可选）
│   └── Multi-Head Attention
├── FFN: Linear→GELU→Linear
└── LayerNorm
```

**KL损失计算**：

```python
KL = -0.5 * (1 + logvar - mu^2 - logvar.exp())
loss = 0.5 * (KL_k + KL_v)
```

### 7.5 模型输入输出

**输入**：

- `x`: (B, T, F, C) — 批次数, 层数, 每层帧数, 特征维度
- `frame_mask`: (B, T, F) — 有效帧掩码

**输出**：

- `logits`: (B, 2, T) — 每层两分类logits
- `extra.kl_loss`: KL散度损失（可选）

---

## 八、训练阶段

### 8.1 训练脚本

```bash
python -m grid_diff_tcn.hier.train \
  --samples_info data_drilling/samples_info_train.json \
  --precomputed_dir grid_diff_tcn/cache_dinov3_features_vits \
  --save grid_diff_tcn/grid_diff_tcn_hierarchical_dinov3_vits.pt \
  --epochs 40 \
  --batch_size 4 \
  --lr 1e-4 \
  --use_dinov3 \
  --dinov3_model vit_small
```

### 8.2 损失函数

**主损失**：Focal Cross Entropy

```python
def focal_cross_entropy(logits, seq_y, mask, gamma=2.0, alpha_pos=0.75):
    pt = softmax(logits)[:, positive_class]
    focal_w = (1 - pt)^gamma
    loss = focal_w * (-log(pt)) * alpha_t
```

**辅助损失**：

1. **位置损失（loc5）**：预测穿透位置与真实位置的距离
2. **窗口内损失（within5）**：真实层±5范围内的概率质量
3. **KL损失**（可选）：概率Transformer的不确定性正则化

**窗口加权**：

```python
# 穿透孔：真值层±window_radius内timestep加权
window_ce_weights(seq_y, window_radius=5, in_window_weight=2.0)
```

### 8.3 训练策略


| 参数                 | 默认值  | 说明           |
| ------------------ | ---- | ------------ |
| `--use_focal`      | True | 使用Focal Loss |
| `--focal_gamma`    | 2.0  | Focal指数      |
| `--weight_pos`     | 10.0 | 正样本权重        |
| `--loc5_weight`    | 0.3  | 位置损失权重       |
| `--within5_weight` | 0.7  | 窗口内损失权重      |
| `--kl_weight`      | 0.0  | KL损失权重（>0启用） |
| `--use_amp`        | True | 混合精度训练       |
| `--patience`       | 10   | 早停耐心值        |


### 8.4 验证指标与网格搜索

**核心指标**（仅对穿透孔计算）：

- `pct_within_3`：预测层号与真实层号误差≤3的百分比
- `pct_within_5`：误差≤5的百分比
- `pct_over_10`：误差>10的百分比

**验证决策函数**：默认 **S3WD**，可通过 `--val_decision` 切换。

**S3WD 网格搜索**：每个 epoch 验证结束后，对验证集上收集的概率矩阵进行离线网格搜索，枚举所有 `(accept, reject, wait)` 组合，选择 `pct_within_5` 最高的参数组合。

```bash
# 网格搜索参数（默认已启用）
--gs_s3wd                           # 启用网格搜索
--gs_accept_range 0.7 0.75 0.8 0.85 0.9 0.95  # accept 候选值
--gs_reject_range 0.5 0.55 0.6 0.65 0.7 0.75  # reject 候选值
--gs_wait_range 1 2 3 4 5           # wait_consecutive 候选值
--gs_save_grid                       # 将完整网格搜索结果保存为 JSON
```

搜索完成后：
- 最优参数打印到终端，同时写入 checkpoint 的 `config.s3wd_best_params`
- 完整网格结果（所有组合的指标）保存为 `{模型路径}_s3wd_grid.json`
- 训练阶段用于选模型保存 best checkpoint 的指标为**网格搜索后的最优结果**

---

## 九、推理阶段

### 9.1 推理脚本

```bash
python -m grid_diff_tcn.hier.infer \
  --samples_info data_drilling/samples_info_test.json \
  --ckpt grid_diff_tcn/grid_diff_tcn_hierarchical_dinov3_vits.pt \
  --output grid_diff_tcn/inference_results_hierarchical_test.json \
  --use_dinov3 \
  --precomputed_dir grid_diff_tcn/cache_dinov3_features_vits
```

### 9.2 决策函数

**1. Safety Lock**

```python
# 前lock_layers层强制为0（物理上不可能穿透）
apply_safety_lock(probs, lock_layers=30)
```

**2. TopKMedian决策**

```python
topkmedian_decide(probs, k=9, min_thresh=0.3):
    # 1. 取概率最高的k个位置
    topk_idx = argsort(-probs)[:k]
    # 2. 中位数概率过阈值则穿透
    if median(probs[topk_idx]) >= min_thresh:
        return True, median(topk_idx)
    return False, None
```

**3. S3WD决策**（序贯三支决策）

```python
s3wd_decide(probs, accept=0.9, reject=0.75, wait=3):
    # P>=accept → 立即穿透
    # P<=reject → 拒绝
    # else → 进入等待状态，连续wait次则穿透
```

**4. 方差门控**（不确定性感知）

```python
topkmedian_with_uncertainty_gate(mean_probs, var_probs, ...):
    # MC采样多次估计方差
    # top-k位置方差中位数过大则否决穿透
```

### 9.3 输出格式

```json
{
  "decision_method": "topkmedian",
  "decision_params": {...},
  "metrics": {
    "n_penetrated": 100,
    "pct_within_3": 85.0,
    "pct_within_5": 92.0,
    "pct_over_10": 3.0
  },
  "results": [
    {
      "sample_path": "...",
      "true_label": 1,
      "true_penetration_layer": 45,
      "true_penetration_index": 5,
      "pred_penetrated": true,
      "pred_penetration_layer": 47,
      "pred_penetration_index": 7,
      "probs": [0.1, 0.2, 0.8, ...],
      "probs_mean": [...],
      "probs_var": [...]
    },
    ...
  ]
}
```

---

## 十、完整使用示例

### 10.1 方式一：使用脚本（推荐）

```bash
# 1. DINOv3特征预计算
bash grid_diff_tcn/hier/scripts/dinov3_precompute.sh

# 2. 训练
bash grid_diff_tcn/hier/scripts/train.sh

# 3. 推理（train集）
bash grid_diff_tcn/hier/scripts/infer.sh train

# 4. 推理（test集）
bash grid_diff_tcn/hier/scripts/infer.sh test
```

### 10.2 方式二：手工特征模式

```bash
# 1. 预计算手工特征
python -m grid_diff_tcn.hier.precompute \
  --samples_info data_drilling/samples_info_train.json \
  --out_dir grid_diff_tcn/cache_hierarchical_features

# 2. 训练（不使用--use_dinov3）
python -m grid_diff_tcn.hier.train \
  --samples_info data_drilling/samples_info_train.json \
  --precomputed_dir grid_diff_tcn/cache_hierarchical_features \
  --save grid_diff_tcn/grid_diff_tcn_hierarchical.pt
```

### 10.3 方式三：在线特征提取（无预计算）

```bash
# 训练时实时提取特征（慢）
python -m grid_diff_tcn.hier.train \
  --samples_info data_drilling/samples_info_train.json \
  --use_dinov3 \
  --dinov3_model vit_base \
  --dinov3_feat_dim 768
```

---

## 十一、核心文件清单


| 文件路径                                  | 功能说明            |
| ------------------------------------- | --------------- |
| `hier/dinov3_precompute.py`           | DINOv3特征预计算脚本   |
| `hier/precompute.py`                  | 手工特征预计算脚本       |
| `hier/train.py`                       | 训练主脚本           |
| `hier/infer.py`                       | 推理主脚本           |
| `hier/frame_layer/__init__.py`        | 模块导出            |
| `hier/frame_layer/dinov3_features.py` | DINOv3特征提取器     |
| `hier/frame_layer/dinov3_dataset.py`  | DINOv3数据集类      |
| `hier/frame_layer/dataset.py`         | 手工特征数据集类        |
| `hier/frame_layer/model.py`           | 分层模型架构          |
| `common/decision.py`                  | 决策函数集合          |
| `common/image_ops.py`                 | 图像处理工具          |
| `common/roi_crop_defaults.py`         | ROI裁剪默认参数       |
| `modules/tcn.py`                      | TCN块实现          |
| `modules/prob_transformer.py`         | 概率Transformer实现 |
| `hier/scripts/dinov3_precompute.sh`   | DINOv3预计算快捷脚本   |
| `hier/scripts/train.sh`               | 训练快捷脚本          |
| `hier/scripts/infer.sh`               | 推理快捷脚本          |


---

## 十二、关键参数对照表

### 12.1 数据处理参数


| 参数                       | 默认值    | 说明        |
| ------------------------ | ------ | --------- |
| `--img_size`             | 128    | 图像目标尺寸    |
| `--roi_size`             | 96/224 | ROI裁剪尺寸   |
| `--max_frames_per_layer` | 8      | 每层最大帧数    |
| `--max_layers`           | None   | 最大层数      |
| `--cc_min_area`          | 12     | 连通域最小面积   |
| `--cc_expand_ratio`      | 0.2    | 检测框扩展比例   |
| `--final_roi_scale`      | 0.85   | ROI占检测框比例 |
| `--use_grayscale`        | False  | 使用灰度图     |


### 12.2 模型参数


| 参数                         | 默认值     | 说明            |
| -------------------------- | ------- | ------------- |
| `--frame_channels`         | 128,128 | 帧级TCN通道       |
| `--layer_tcn_channels`     | 64,64   | 层间TCN通道       |
| `--kernel_size`            | 3       | 卷积核大小         |
| `--d_model`                | 64/128  | Transformer维度 |
| `--num_heads`              | 4       | 注意力头数         |
| `--num_transformer_layers` | 2       | Transformer层数 |
| `--dim_feedforward`        | 256/512 | FFN维度         |
| `--dropout`                | 0.1     | Dropout比例     |
| `--kl_weight`              | 0.0     | KL损失权重        |
| `--use_frame_gru`          | True    | 帧级GRU         |
| `--use_multiscale`         | True    | 多尺度融合         |


### 12.3 DINOv3参数


| 参数                  | 默认值      | 说明       |
| ------------------- | -------- | -------- |
| `--use_dinov3`      | False    | 启用DINOv3 |
| `--dinov3_model`    | vit_base | 模型规模     |
| `--dinov3_feat_dim` | 768/384  | 特征维度     |
| `--dinov3_roi_size` | 224      | ROI尺寸    |


### 12.4 推理参数


| 参数                  | 默认值        | 说明           |
| ------------------- | ---------- | ------------ |
| `--decision_method` | s3wd | 决策方法（s3wd/topkmedian）  |
| `--lock_layers`     | 30         | 安全锁层数        |
| `--k`               | 9          | TopKMedian的K |
| `--min_thresh`      | 0.3        | TopKMedian 概率阈值 |
| `--s3wd_accept`     | 0.9        | S3WD 接受阈值 |
| `--s3wd_reject`     | 0.75       | S3WD 拒绝阈值 |
| `--s3wd_wait`       | 3          | S3WD 连续等待次数 |
| `--unc_samples`     | 1          | MC采样次数       |
| `--unc_gate`        | False      | 方差门控（仅 topkmedian） |


### 12.5 训练网格搜索参数


| 参数                  | 默认值        | 说明           |
| ------------------- | ---------- | ------------ |
| `--gs_s3wd`         | True       | 启用 S3WD 网格搜索 |
| `--gs_accept_range` | 见下方       | S3WD accept 候选值列表 |
| `--gs_reject_range` | 见下方       | S3WD reject 候选值列表 |
| `--gs_wait_range`   | 见下方       | S3WD wait_consecutive 候选值 |
| `--gs_save_grid`    | True       | 保存完整网格搜索结果 JSON |


---

## 十三、技术亮点

### 13.1 分层建模

- **帧级**：GRU + Attention Pooling捕获单层内的时序变化
- **层间级**：ProbTransformer建模穿透决策的时序依赖

### 13.2 不确定性建模

- 概率注意力机制：通过KL散度正则化学习隐式的不确定性
- 方差门控：推理时可使用MC采样估计预测方差

### 13.3 多尺度特征

- 帧级使用多分支卷积捕获不同感受野的特征
- 融合局部、中程、长程信息

### 13.4 预计算加速

- 特征预计算到.pt文件，训练/推理时直接加载
- 多进程并行预计算

### 13.5 DINOv3自监督特征

- 使用大规模自监督预训练的视觉特征
- 768维CLS token比手工特征更语义丰富

---
损失	权重	说明
Focal CE	-	基础分类损失，α=0.75, γ=2.0
Window CE	in_window=2.0	真值±3层范围内加权
loc3_loss	0.3	软预测位置与真值距离 >3 的均方误差
within3_loss	0.7	概率集中在真值±3层内的交叉熵
KL Loss	0.01	概率分布正则化
