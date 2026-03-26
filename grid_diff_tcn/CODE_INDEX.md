# grid_diff_tcn 代码与结果索引

本目录实现 **Grid-Diff 1D-TCN** 激光钻孔穿透检测：层内融合 → 帧间差分 → 8×8 网格池化 → 因果 TCN → 训练/推理与多种决策方法对比。每个代码文件开头均有详细功能说明，此处做汇总与结果文件索引。

---

## 一、核心代码（按使用顺序）

| 文件 | 功能简述 |
|------|----------|
| `__init__.py` | 包入口，导出 GridDiffDrillingDataset、collate_fn、GridDiffTCN。 |
| `dataset.py` | 模块一：单孔数据加载，层内融合 + 帧间差 + 8×8 网格，输出 [Seq_Len, 64]；提供 collate_fn、get_layer_list_from_path。 |
| `tcn_model.py` | 模块二：因果 1D-TCN 模型，输入 (B, 64, T)，输出 (B, 2, T) 逐层 logits。 |
| `precompute_features.py` | 预计算每孔 [Seq_Len, 64] 到 .pt 文件，训练时 --precomputed_dir 可大幅提速。 |
| `split_train_test.py` | 将 samples_info.json 按比例划分为 samples_info_train.json / samples_info_test.json。 |
| `train.py` | 模块三：窗口采样训练，Focal/平衡/按孔 batch、验证、AMP、最佳权重保存；支持 --simple 简单模式。 |
| `inference.py` | 模块四：加载权重，整孔前向 + 安全锁 + S3WD 决策，输出按孔指标与结果 JSON。 |

---

## 二、推理与决策相关

| 文件 | 功能简述 |
|------|----------|
| `inference.py` | 主推理脚本：S3WD 决策，支持 --samples_info、--ckpt、--output，带进度条。 |
| `search_s3wd_thresholds.py` | 在验证集上对 S3WD 的 accept/reject/wait 做网格搜索，输出最优参数与指标（使用 cache，前向仅一次）。 |
| `compare_decision_methods.py` | 对比 Argmax、SmoothFirst、Centroid、TopKMedian、TwoStage、FirstThresh、S3WD；每种方法做阈值网格搜索，输出各方法最优参数与全局最优。 |
| `infer_missing_json_holes.py` | 对 data_drilling/train 下“有图无 JSON”的孔做推理，打印穿透结果。 |

---

## 三、数据与缓存工具

| 文件 | 功能简述 |
|------|----------|
| `link_cache_features_by_name.py` | 为已有 idx.pt 创建“样本名.pt”硬链接或复制，便于按名访问。 |
| `rename_cache_features.py` | 将 cache 目录下 0.pt, 1.pt, … 重命名为样本文件夹名.pt。 |
| `visualize_roi_simple.py` | 在样本图上绘制 CenterCrop 与 8×8 网格，保存为 PNG，用于核对 ROI。 |

---

## 四、备份与说明文档

| 文件 | 说明 |
|------|------|
| `train copy.py` | train.py 的备份/分支版本，仅作保留，日常训练请用 train.py。 |
| `README.md` | 项目概览、快速开始、目录结构、环境与依赖。 |
| `PIPELINE.md` | 从原理到实现的完整流程说明。 |
| `TRAIN_TEST_COMMANDS.md` | 训练与测试集划分、推理的常用命令。 |
| `CODE_INDEX.md` | 本文件：代码与结果文件索引。 |

---

## 五、结果与产物文件

| 文件/目录 | 说明 |
|-----------|------|
| `grid_diff_tcn.pt` | 训练得到的模型权重（或最佳验证权重），推理时 --ckpt 指定。 |
| `cache_features_train/` | 预计算特征目录，每孔一 .pt（按索引或按样本名），训练时 --precomputed_dir 指向此处。 |
| `s3wd_best_params.json` | S3WD 网格搜索得到的最优参数与对应指标（n_penetrated, pct_within_3/5, pct_over_10）。 |
| `decision_compare_results.json` | 各决策方法网格搜索后的最优参数与指标列表。 |
| `inference_results.json` | 使用 inference.py 对某次 samples_info 推理的逐孔结果 JSON（路径、真值/预测穿透层等）。 |
| `inference_results_test.json` | 对测试集推理的结果（若运行时指定了该输出名）。 |
| `inference_results_s3wd.json` | 使用 S3WD 对测试集推理的结果。 |
| `inference_results_first09.json` | 使用“首次概率≥0.9”决策推理的结果（若存在 inference_first09.py 脚本）。 |
| `roi_simple.png` | visualize_roi_simple.py 生成的 ROI/网格示意图。 |

---

## 六、验收指标（参考）

- 误差 ≤5 层占比 ≥98%  
- 误差 ≤3 层占比 ≥80%  
- 误差 >10 层占比 0%  

详见项目内 `test_3.9/evaluate_layer_diff.py` 等验收脚本。
