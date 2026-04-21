# masked_v2: Trainable Encoder for Masked Image Modeling

```


v2 Stage 1 (encoder 可训练):
  原始图像 ──→ [可训练 DINOv3] ──→ CLS ──→ [decoder] ──→ 重建像素
                     ↑ ↓ 用 L1 loss 反向传播
               学到钻孔特征
```

## 文件结构

```
masked_v2/
├── model.py          # MaskedPixelModel（freeze_encoder 默认 False）
│                      # 新增 set_encoder_trainable() / freeze_classifier() / unfreeze_classifier()
├── train.py           # Stage 1: encoder+decoder 联合训练（classifier 冻结）
│                      # Stage 2: encoder 冻结，classifier 微调
├── extract.py         # 支持从 Stage1 checkpoint 提取领域适应特征
├── infer.py          # 推理
├── dataset.py        # 数据集
├── masks.py          # CenterMask
├── decoder.py        # PixelDecoder
├── scripts/
│   ├── train.sh           # 训练启动脚本
│   └── extract_features.sh # 特征提取脚本
└── README.md         # 本文件
```

## 训练流程

### Stage 1: MIM 预训练（encoder + decoder）

```bash
bash grid_diff_tcn/masked_v2/scripts/stage1.sh
```

- encoder_lr = 1e-6（远低于 decoder lr=1e-4，防止灾难性遗忘）
- classifier 参数冻结（不参与 Stage 1 训练）
- 输出: `grid_diff_tcn/masked_v2/checkpoints/stage1.pt`

> 注意：`batch_size` 必须为 1（内部 collate 使用 max_f padding，多样本 batch 时 reshape 会出错）

### Stage 2: 分类微调（encoder 冻结，classifier 训练）

```bash
bash grid_diff_tcn/masked_v2/scripts/stage2.sh \
  --resume_from grid_diff_tcn/masked_v2/checkpoints/stage1.pt
```

- encoder 冻结（使用 Stage 1 学到的领域适应特征）
- classifier lr = 1e-4
- 输出: `grid_diff_tcn/masked_v2/checkpoints/stage2.pt`

### Stage 2 进阶：继续微调 encoder

如果 Stage 2 分类效果仍不好，可以尝试继续微调 encoder：

```bash
UNFREEZE_ENCODER_STAGE2=True \
  bash grid_diff_tcn/masked_v2/scripts/stage2.sh \
    --resume_from grid_diff_tcn/masked_v2/checkpoints/stage1.pt
```

- encoder lr = 1e-6，classifier lr = 1e-4
- encoder 和 classifier 一起微调
- 风险：容易过拟合，谨慎使用

### 完整流程示例

```bash
# 1. Stage 1 MIM 预训练
bash grid_diff_tcn/masked_v2/scripts/stage1.sh

# 2. (可选) 用训练好的 encoder 提取特征，加速 Stage 2
bash grid_diff_tcn/masked_v2/scripts/extract_features.sh \
  --checkpoint grid_diff_tcn/masked_v2/checkpoints/stage1.pt \
  --output_dir grid_diff_tcn/masked_v2/features_cache

# 3. Stage 2 分类微调
PRECOMPUTED_DIR=grid_diff_tcn/masked_v2/features_cache \
  RESUME_FROM=grid_diff_tcn/masked_v2/checkpoints/stage1.pt \
  bash grid_diff_tcn/masked_v2/scripts/stage2.sh
```

## 新增参数


| 参数                          | 默认值     | 说明                                                          |
| --------------------------- | ------- | ----------------------------------------------------------- |
| `--freeze_encoder`          | `False` | v2 默认 encoder 可训练（v1 默认 True）                               |
| `--encoder_lr`              | `1e-6`  | Stage 1 encoder 学习率                                         |
| `--encoder_lr_stage2`       | `1e-6`  | Stage 2 encoder 学习率（仅 `--unfreeze_encoder_stage2=True` 时生效） |
| `--unfreeze_encoder_stage2` | `False` | Stage 2 是否继续训练 encoder                                     |



|                 | v2 (masked_v2/)          |
| --------------- | ------------------------ |
| Stage 1 encoder | **可训练**                  |
| Stage 1 训练目标    | **encoder + decoder 联合** |
| Stage 1 学到了什么   | **钻孔领域特征**               |
| Stage 2 特征质量    | **领域适应后的特征**             |
| Stage 2 encoder | 冻结（默认）                   |
| 显存需求            | 较高（encoder 参与反向传播）       |


## 快速开始

```bash
# 1. Stage 1 MIM 预训练
bash grid_diff_tcn/masked_v2/scripts/train.sh --stage 1

# 2. (可选) 用训练好的 encoder 提取特征，加速 Stage 2
bash grid_diff_tcn/masked_v2/scripts/extract_features.sh \
  --checkpoint grid_diff_tcn/masked_v2/checkpoints/stage1.pt

# 3. Stage 2 分类微调
PRECOMPUTED_DIR=grid_diff_tcn/masked_v2/features_cache \
  bash grid_diff_tcn/masked_v2/scripts/train.sh --stage 2 \
    --resume_from grid_diff_tcn/masked_v2/checkpoints/stage1.pt \
    --precomputed_dir "$PRECOMPUTED_DIR" --use_cached_features
```

