# -*- coding: utf-8 -*-
"""
根据 decision_compare_results_train.json 与 decision_compare_results_test.json 生成汇总指标，
并输出 RESULTS.md 与 experiment_report.html；decision_compare_results_combined.json 为训练+测试合并指标。
用法：在 grid_diff_tcn 目录下运行 python build_decision_report.py
"""

import os
import json

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_json(name):
    path = os.path.join(SCRIPT_DIR, name)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def combined_metrics(train_list, test_list):
    """按 n_penetrated 加权合并训练、测试两份指标。返回同结构列表，metrics 为合并值。"""
    by_method = {}
    for tag, lst in [("train", train_list), ("test", test_list)]:
        if not lst:
            continue
        for r in lst:
            if r["method"] not in by_method:
                by_method[r["method"]] = {"best_params": r["best_params"], "train": None, "test": None}
            by_method[r["method"]][tag] = r["metrics"]

    out = []
    for method, data in by_method.items():
        parts = [(data[k]["n_penetrated"], data[k]) for k in ("train", "test") if data.get(k) is not None]
        if not parts:
            continue
        if len(parts) == 1:
            merged = parts[0][1].copy()
        else:
            n_total = sum(n for n, _ in parts)
            merged = {
                "n_penetrated": n_total,
                "pct_within_3": sum(m["pct_within_3"] * n for n, m in parts) / n_total,
                "pct_within_5": sum(m["pct_within_5"] * n for n, m in parts) / n_total,
                "pct_over_10": sum(m["pct_over_10"] * n for n, m in parts) / n_total,
            }
        out.append({"method": method, "best_params": data["best_params"], "metrics": merged})
    return out


def main():
    train_list = load_json("decision_compare_results_train.json") or []
    test_list = load_json("decision_compare_results_test.json") or []

    combined_list = combined_metrics(train_list, test_list)
    if not combined_list:
        combined_list = train_list or test_list

    # 合同参考指标（来自 test_3.9/evaluate_layer_diff.py）
    ref = {"pct_within_5_min": 98, "pct_within_3_min": 80, "pct_over_10_max": 0}

    def row_lines(data_list, title):
        lines = [f"| 方法 | n_pen | ≤3层% | ≤5层% | >10层% | 最优参数 |", "|------|-------|-------|-------|--------|----------|"]
        for r in data_list:
            m = r["metrics"]
            params_str = json.dumps(r["best_params"], ensure_ascii=False)
            if len(params_str) > 36:
                params_str = params_str[:33] + "..."
            lines.append(f"| {r['method']} | {m['n_penetrated']} | {m['pct_within_3']:.1f} | {m['pct_within_5']:.1f} | {m['pct_over_10']:.1f} | {params_str} |")
        return lines

    # ---------- RESULTS.md ----------
    md_lines = [
        "# Grid-Diff TCN 决策方法对比结果",
        "",
        "基于 `compare_decision_methods.py` 在训练集、测试集上的网格搜索最优结果汇总。",
        "",
        "## 合同参考指标",
        "",
        f"- 误差≤5层占比 ≥ {ref['pct_within_5_min']}%",
        f"- 误差≤3层占比 ≥ {ref['pct_within_3_min']}%",
        f"- 误差>10层占比 = {ref['pct_over_10_max']}%",
        "",
    ]
    if train_list:
        md_lines.append("## 训练集（decision_compare_results_train.json）")
        md_lines.append("")
        md_lines.extend(row_lines(train_list, "train"))
        md_lines.append("")
    md_lines.append("")
    md_lines.append("## 测试集（decision_compare_results_test.json）")
    md_lines.append("")
    md_lines.extend(row_lines(test_list, "test"))
    md_lines.append("")
    md_lines.append("## 合并指标（训练+测试按穿透孔数加权）")
    md_lines.append("")
    md_lines.extend(row_lines(combined_list, "combined"))
    md_lines.append("")
    best = max(combined_list, key=lambda x: x["metrics"]["pct_within_5"]) if combined_list else None
    if best:
        md_lines.append(f"**合并最优（按≤5层）**: {best['method']}，≤5层: {best['metrics']['pct_within_5']:.1f}%")
    md_lines.append("")

    with open(os.path.join(SCRIPT_DIR, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    # ---------- experiment_report.html ----------
    def table_html(data_list, caption):
        rows = []
        for r in data_list:
            m = r["metrics"]
            params_str = json.dumps(r["best_params"], ensure_ascii=False)
            rows.append(
                f"    <tr><td>{r['method']}</td><td>{m['n_penetrated']}</td><td>{m['pct_within_3']:.1f}%</td><td>{m['pct_within_5']:.1f}%</td><td>{m['pct_over_10']:.1f}%</td><td><code>{params_str}</code></td></tr>"
            )
        return "\n".join(rows)

    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Grid-Diff TCN 决策方法实验报告</title>
  <style>
    :root {{
      --bg: #f8fafc; --card: #fff; --text: #1e293b; --text-muted: #64748b;
      --accent: #0ea5e9; --accent-dark: #0284c7; --success: #10b981; --border: #e2e8f0;
      --shadow: 0 1px 3px rgba(0,0,0,.08); --shadow-md: 0 4px 12px rgba(0,0,0,.1);
      --radius: 10px; --radius-sm: 6px;
    }}
    * {{ box-sizing: border-box; }}
    body {{ font-family: "Inter","SF Pro Text","Segoe UI",system-ui,sans-serif; margin: 0; padding: 2rem 1rem 3rem; background: var(--bg); color: var(--text); line-height: 1.65; font-size: 15px; }}
    .wrap {{ max-width: 920px; margin: 0 auto; }}
    .hero {{ background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%); color: #f1f5f9; padding: 2rem 2.5rem; border-radius: var(--radius); margin-bottom: 2rem; box-shadow: var(--shadow-md); }}
    .hero h1 {{ margin: 0 0 0.5rem; font-size: 1.75rem; font-weight: 700; letter-spacing: -0.02em; }}
    .hero p {{ margin: 0; color: #94a3b8; font-size: 0.95rem; }}
    .hero code {{ background: rgba(255,255,255,.12); color: #e2e8f0; padding: 0.2em 0.5em; border-radius: 4px; }}
    h2 {{ font-size: 1.2rem; font-weight: 600; color: var(--text); margin: 2rem 0 1rem; padding-bottom: 0.4rem; border-bottom: 2px solid var(--border); }}
    .card {{ background: var(--card); border-radius: var(--radius); padding: 1.25rem 1.5rem; margin-bottom: 1.25rem; box-shadow: var(--shadow); border: 1px solid var(--border); }}
    .card p:first-child {{ margin-top: 0; }} .card p:last-child {{ margin-bottom: 0; }}
    ol {{ padding-left: 1.4rem; margin: 0.75rem 0; }} ol li {{ margin: 0.5rem 0; }}
    ul {{ list-style: none; padding: 0; margin: 0.75rem 0; }}
    .method-desc {{ margin: 0.6rem 0; padding: 0.6rem 0.85rem; background: #f8fafc; border-left: 4px solid var(--accent); border-radius: 0 var(--radius-sm) var(--radius-sm) 0; font-size: 0.92rem; }}
    .method-desc strong {{ color: var(--accent-dark); display: inline-block; min-width: 10em; }}
    code {{ font-family: "JetBrains Mono","Fira Code",monospace; font-size: 0.85em; background: #f1f5f9; padding: 0.2em 0.45em; border-radius: 4px; color: #475569; }}
    .ref {{ background: linear-gradient(135deg, #fef9c3 0%, #fef08a 100%); border: 1px solid #facc15; padding: 1rem 1.25rem; border-radius: var(--radius); margin: 1.25rem 0; box-shadow: var(--shadow); }}
    .ref strong {{ color: #854d0e; }} .ref ul {{ margin: 0.4rem 0 0; padding-left: 1.2rem; }}
    table {{ width: 100%; border-collapse: collapse; border-radius: var(--radius); overflow: hidden; box-shadow: var(--shadow); background: var(--card); margin: 1rem 0; font-size: 0.9rem; }}
    thead {{ background: linear-gradient(180deg, #f1f5f9 0%, #e2e8f0 100%); }}
    th {{ padding: 0.75rem 1rem; text-align: left; font-weight: 600; color: #475569; border-bottom: 2px solid var(--border); }}
    td {{ padding: 0.65rem 1rem; border-bottom: 1px solid var(--border); }}
    tbody tr:hover {{ background: #f8fafc; }} tbody tr:last-child td {{ border-bottom: none; }}
    .best {{ display: inline-block; background: linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%); color: #065f46; padding: 0.6rem 1rem; border-radius: var(--radius); font-weight: 600; margin: 1rem 0; border: 1px solid #6ee7b7; }}
    .footer {{ margin-top: 2.5rem; padding-top: 1.25rem; border-top: 1px solid var(--border); color: var(--text-muted); font-size: 0.85rem; }}
    caption {{ font-size: 0.8rem; color: var(--text-muted); margin-bottom: 0.35rem; text-align: left; padding: 0 0.25rem; }}
  </style>
</head>
<body>
  <div class="wrap">
  <header class="hero">
    <h1>Grid-Diff TCN 决策方法实验报告</h1>
    <p>基于 <code>compare_decision_methods.py</code> 在训练集、测试集上的网格搜索最优结果汇总。</p>
  </header>

  <h2>模型方法步骤</h2>
  <div class="card">
  <p>本实验采用 <strong>Grid-Diff 1D-TCN</strong> 做激光钻孔穿透层定位：输入单孔逐层图像序列，经物理降维与因果时序卷积，得到每层“穿透”概率，再经决策方法输出是否穿透及穿透层号。</p>
  <ol>
    <li><strong>输入</strong>：单孔目录下按层命名的图像（如 <code>*.jpg</code>），每层可有多张图；样本列表由 <code>samples_info.json</code> 提供（含 <code>sample_path</code>、<code>is_penetrated</code>、<code>penetration_layer</code>）。</li>
    <li><strong>层内融合</strong>：同层多张图做均值，得到每层一张代表图 I<sub>n</sub>，减少层内抖动。</li>
    <li><strong>帧间绝对差分</strong>：D<sub>n</sub> = |I<sub>n</sub> − I<sub>n−1</sub>|，突出层间变化，穿透前后差异会放大。</li>
    <li><strong>ROI 与 8×8 网格池化</strong>：对差分图做中心裁剪取 ROI，再划分为 8×8 共 64 个 patch，每 patch 求均值，得到每层 64 维向量；序列形状为 <code>[Seq_Len, 64]</code>。</li>
    <li><strong>因果 1D-TCN</strong>：对 <code>(B, 64, T)</code> 做因果膨胀卷积（时刻 t 仅依赖 t 及之前），多层 TCN 块 + 分类头，每时间步输出 2 类 logits（未穿透/穿透）。</li>
    <li><strong>训练</strong>：按真实穿透层构造逐层标签，使用 Focal Loss / 交叉熵 + 辅助定位损失；支持平衡 batch 与预计算特征加速。</li>
    <li><strong>推理与安全锁</strong>：整孔前向得到每层概率曲线；前若干层（安全锁）概率置零，不参与后续决策，避免表面高亮误判。</li>
    <li><strong>决策方法</strong>：在概率曲线上应用 Argmax、SmoothFirst、Centroid、TopKMedian、TwoStage、FirstThresh、S3WD 等之一，得到最终“是否穿透”及“穿透层”预测；本报告对每种方法做网格搜索，按“误差≤5 层占比”选最优参数。</li>
  </ol>
  </div>

  <h2>决策方法说明</h2>
  <div class="card">
  <p>模型对每一层输出“穿透”概率，形成整孔概率曲线。以下七种决策方法根据该曲线判定：<em>是否穿透</em>以及<em>穿透层位置</em>（层索引）。各方法均在“安全锁”之后应用（前若干层概率置零，不参与判定）。</p>
  <ul>
    <li class="method-desc"><strong>Argmax</strong> — 取概率最大的那一层作为穿透层；若该最大概率 &lt; min_thresh 则判为未穿透。实现简单，对单峰明显时稳定；易受噪声影响。</li>
    <li class="method-desc"><strong>SmoothFirst</strong> — 先对概率做长度为 window 的滑动平均以平滑曲线，再取<em>首次</em>达到 thresh 的层为穿透层。适合强调“首次超过阈值”的时序语义，平滑可减弱抖动。</li>
    <li class="method-desc"><strong>Centroid</strong> — 在概率 &gt; thresh 的层上做<em>概率加权平均</em>索引，四舍五入得到穿透层；若最大概率 &lt; thresh 则判未穿透。利用高概率区整体重心，对多峰或平台有一定鲁棒性。</li>
    <li class="method-desc"><strong>TopKMedian</strong> — 取概率最大的 K 个层的<em>索引中位数</em>作为穿透层；若最大概率 &lt; min_thresh 则未穿透。用中位数抗野值，K 控制参与层数，常在本任务中表现较好。</li>
    <li class="method-desc"><strong>TwoStage</strong> — 两阶段：先找连续概率 &gt; region_thresh 的“高概率区间”，再在区间内取首次达到 peak_thresh 的层，若无则取区间内 argmax。兼顾区域与峰值的时序结构。</li>
    <li class="method-desc"><strong>FirstThresh</strong> — 取<em>首次</em>概率 ≥ thresh 的层为穿透层，若无则判未穿透。与 SmoothFirst 类似但不对概率平滑，只做单阈值截断，实现最简。</li>
    <li class="method-desc"><strong>S3WD</strong>（序贯三支决策）— 设接受阈 accept、拒绝阈 reject 与连续步数 wait：概率超过 accept 判穿透并记录层；低于 reject 判未穿透；介于两者则继续等待，满 wait 步再决。适合需要“延迟决策、减少误判”的场景。</li>
  </ul>
  <p>报告中“最优参数”为各方法在网格搜索下按“误差≤5 层占比”选出的超参数。</p>
  </div>

  <div class="ref">
    <strong>合同参考指标</strong>
    <ul>
      <li>误差≤5层占比 ≥ {ref['pct_within_5_min']}%</li>
      <li>误差≤3层占比 ≥ {ref['pct_within_3_min']}%</li>
      <li>误差>10层占比 = {ref['pct_over_10_max']}%</li>
    </ul>
  </div>
"""
    if train_list:
        html += """
  <h2>1. 训练集（decision_compare_results_train.json）</h2>
  <table>
    <caption>训练集各方法最优参数及指标</caption>
    <thead><tr><th>方法</th><th>n_pen</th><th>≤3层%</th><th>≤5层%</th><th>>10层%</th><th>最优参数</th></tr></thead>
    <tbody>
""" + table_html(train_list, "训练集") + """
    </tbody>
  </table>
"""
    sec = 2 if train_list else 1
    html += f"""
  <h2>{sec}. 测试集（decision_compare_results_test.json）</h2>
  <table>
    <caption>测试集各方法最优参数及指标</caption>
    <thead><tr><th>方法</th><th>n_pen</th><th>≤3层%</th><th>≤5层%</th><th>>10层%</th><th>最优参数</th></tr></thead>
    <tbody>
{table_html(test_list, "测试集")}
    </tbody>
  </table>

  <h2>{sec + 1}. 合并指标（训练+测试按穿透孔数加权）</h2>
  <p>合并 n_penetrated = 训练+测试穿透孔数之和；各占比为按孔数加权平均。</p>
  <table>
    <caption>合并后各方法指标</caption>
    <thead><tr><th>方法</th><th>n_pen</th><th>≤3层%</th><th>≤5层%</th><th>>10层%</th><th>最优参数</th></tr></thead>
    <tbody>
{table_html(combined_list, "合并")}
    </tbody>
  </table>
"""
    if best:
        html += f"""
  <p class="best">合并最优（按≤5层占比）: {best['method']} — ≤5层: {best['metrics']['pct_within_5']:.1f}%</p>
"""
    html += """
  <div class="footer">
    生成脚本: build_decision_report.py · 数据来源: decision_compare_results_train/test.json → 合并为 decision_compare_results_combined.json
  </div>
  </div>
</body>
</html>
"""

    with open(os.path.join(SCRIPT_DIR, "experiment_report.html"), "w", encoding="utf-8") as f:
        f.write(html)

    # 合并结果写回一份 JSON 供程序读取
    with open(os.path.join(SCRIPT_DIR, "decision_compare_results_combined.json"), "w", encoding="utf-8") as f:
        json.dump(combined_list, f, ensure_ascii=False, indent=2)
    print("已生成: RESULTS.md, experiment_report.html, decision_compare_results_combined.json")


if __name__ == "__main__":
    main()
