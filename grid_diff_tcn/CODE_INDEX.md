# grid_diff_tcn 代码索引

目录分为两套：**`legacy/`（旧版经典）** 与 **`hier/`（新版分层）**。  
新版 `hier/frame_layer` 依赖 **`legacy/dataset.py`** 中的读图与 `color_cc` 工具函数。

---

## 一、`hier/` 新版（分层，推荐）

| 路径 | 说明 |
|------|------|
| `hier/frame_layer/dataset.py` | 分层数据集 `(T,F,64)`、`frame_mask`、`seq_label`；可读预计算 `.pt`。 |
| `hier/frame_layer/model.py` | 帧内 TCN + 层间 TCN + 概率 Transformer。 |
| `hier/train.py` | 分层训练。 |
| `hier/precompute.py` | 分层特征预计算（多进程 `--num_workers`）。 |
| `hier/infer.py` | 分层推理（`topkmedian` / `s3wd`）；决策函数来自 `legacy/inference.py`。 |
| `hier/visualize_badcase.py` | 分层推理 JSON 的 bad/good case ROI 拼图。 |
| `hier/train.sh` / `hier/precompute.sh` / `hier/infer.sh` | 快捷命令（`infer.sh` 支持 `train` / `test`）。 |

---

## 二、`legacy/` 旧版（经典 GridDiff + TCN）

| 路径 | 说明 |
|------|------|
| `legacy/dataset.py` | 层内融合 ROI、帧间差分、`8×8` 网格 → `[Seq_Len,64]` 与 `aux`。 |
| `legacy/tcn.py` | 1D 因果 TCN。 |
| `legacy/transformer_tcn.py` | TCN + 概率 Transformer。 |
| `legacy/two_tower.py` | 双塔（主特征 + aux）。 |
| `legacy/train.py` | 经典训练。 |
| `legacy/inference.py` | 经典推理与决策工具（`apply_safety_lock`、`topkmedian_*` 等）。 |
| `legacy/precompute.py` | 经典特征预计算。 |
| `legacy/infer_missing_json_holes.py` | 无标注 JSON 的孔推理。 |
| `legacy/visualize_two_holes_roi.py` | 双孔 ROI 可视化。 |
| `legacy/decision_grid_search.py` | 离线决策参数搜索。 |
| `legacy/build_decision_report.py` | 决策对比报告 HTML/JSON。 |
| `legacy/badcase_analysis.py` | Badcase 统计。 |
| `legacy/split_train_test.py` | 划分 `samples_info`。 |
| `legacy/build_samples_info.py` | 从 `data_drilling/train` 生成索引 JSON。 |
| `legacy/train.sh` / `legacy/precompute.sh` / `legacy/inference.sh` | 快捷命令。 |

---

## 三、根目录与其它

| 路径 | 说明 |
|------|------|
| `__init__.py` | 包级导出 `GridDiffDrillingDataset`、`GridDiffTCN`（自 `legacy/` 加载）。 |
| `README.md` | 总体说明。 |
| `outputs/`、`cache_*` | 结果与预计算缓存（路径不变，仍在 `grid_diff_tcn/` 下）。 |

---

## 四、运行方式（均在仓库根 `dinov3-main` 下）

```bash
# 新版分层
bash grid_diff_tcn/hier/precompute.sh
bash grid_diff_tcn/hier/train.sh
bash grid_diff_tcn/hier/infer.sh test

# 旧版经典
bash grid_diff_tcn/legacy/precompute.sh
bash grid_diff_tcn/legacy/train.sh
python3 grid_diff_tcn/legacy/inference.py --help
```
