# grid_diff_tcn

激光钻孔穿透检测代码，按版本分目录：

| 目录 | 内容 |
|------|------|
| **`hier/`** | **新版（推荐）**：帧内 TCN + 层间 TCN + 概率 Transformer |
| **`legacy/`** | **旧版**：GridDiff + (Transformer-)TCN 经典窗口管线 |

ROI 统一为 **`color_cc`**（颜色 + 连通域）+ letterbox，不变形。

---

## 1. 任务与指标

对每个孔的层序列预测穿透层。仅统计真值「已穿透」的孔：

- `<=3` / `<=5`：与真值层号误差不超过 3 / 5 层  
- `>10`：误差超过 10 层  

---

## 2. 数据约定

孔目录下多张 `jpg`，文件名含 `..._<frame>_<layer>.jpg`。  
索引见 `data_drilling/samples_info*.json`（`sample_path`、`is_penetrated`、`penetration_layer`）。

---

## 3. Pipeline 概要

### 3.1 `legacy/` 经典

层内融合 → `color_cc` ROI → letterbox → 层间差分 → `8×8` 网格 → 每层 64D → 1D TCN / Transformer-TCN → S3WD / TopKMedian 等决策。

### 3.2 `hier/` 分层

每层保留多帧 → 每帧 `color_cc` + 网格 64D → 张量 `(B,T,F,64)` → 帧内 TCN → 层间 TCN + 概率 Transformer → `(B,2,T)` → 决策。

---

## 4. 入口脚本（简化命名）

### 新版 `hier/`

- `precompute.py` — 预计算分层 `.pt`  
- `train.py` — 训练  
- `infer.py` — 推理  
- `train.sh` / `precompute.sh` / `infer.sh` — 快捷命令（`infer.sh` 支持 `train` 与 `test`，test 无预计算缓存时为在线提特征）

### 旧版 `legacy/`

- `precompute.py` — 预计算 `[Seq,64]`  
- `train.py` / `inference.py`  
- `train.sh` / `precompute.sh` / `inference.sh`

---

## 5. 推荐流程（分层）

在仓库根目录执行：

```bash
bash grid_diff_tcn/hier/precompute.sh
bash grid_diff_tcn/hier/train.sh
bash grid_diff_tcn/hier/infer.sh          # train 集 + 预计算 cache
bash grid_diff_tcn/hier/infer.sh test    # 测试集，无 cache 则在线计算
```

预计算与训练的 `img_size`、`roi_size`、`exclude_json`、孔锚框等参数须一致。详见 `hier/precompute.sh` 与 `hier/train.sh`。

---

## 6. 结果 JSON

`inference_results*.json` 含 `metrics`（`n_penetrated`、`pct_within_*`）与 `results` 逐孔字段（`sample_path`、`pred_penetrated`、`probs` 等）。

---

## 7. 目录索引

更全的文件表见 **`CODE_INDEX.md`**。
