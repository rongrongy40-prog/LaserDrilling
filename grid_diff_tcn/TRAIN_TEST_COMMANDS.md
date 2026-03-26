# 训练集 / 测试集划分与命令

## 已生成的划分

- **训练集**: `../data_drilling/samples_info_train.json`（820 条）
- **测试集**: `../data_drilling/samples_info_test.json`（205 条）

由 `split_train_test.py` 按 80% / 20% 随机划分（种子 42）。重新划分可执行：

```bash
cd grid_diff_tcn
python3 split_train_test.py --samples_info ../data_drilling/samples_info.json --out_dir ../data_drilling --ratio 0.8 --seed 42
```

---

## 1. 预计算特征（仅训练集，可选但推荐）

当前 `cache_features` 是按**全量** `samples_info.json` 生成的，划分后训练应用**训练集**单独预计算，再给训练用：

```bash
cd grid_diff_tcn
python3 precompute_features.py \
  --samples_info ../data_drilling/samples_info_train.json \
  --out_dir ./cache_features_train
```

（参数 `--img_size`、`--roi_size` 等需与训练一致。）

---

## 2. 训练（仅用训练集）

```bash
cd grid_diff_tcn
python3 train.py \
  --samples_info ../data_drilling/samples_info_train.json \
  --precomputed_dir ./cache_features_train \
  --num_workers 4 \
  --save grid_diff_tcn.pt
```

不使用预计算时去掉 `--precomputed_dir` 和 `--num_workers` 或设 `--num_workers 0`。

---

## 3. 推理（仅在测试集上评估）

```bash
cd grid_diff_tcn
python3 inference.py \
  --ckpt grid_diff_tcn.pt \
  --samples_info ../data_drilling/samples_info_test.json \
  --output inference_results_test.json
```

如需指定数据根目录或设备：

```bash
python3 inference.py \
  --ckpt grid_diff_tcn.pt \
  --samples_info ../data_drilling/samples_info_test.json \
  --base_dir /home/student2025/wudf2025/dinov3-main \
  --output inference_results_test.json \
  --device cuda
```

这样评估结果对应**未参与训练的测试集**。
