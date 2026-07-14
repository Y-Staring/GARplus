# 既有噪声图上的演绎修正

## 怎么运行

查看完整参数：

```bash
python experiments/exp1_robustness/deductive_refinement/apply_gar_rules_to_noisy_graph.py --help
```

典型运行：

```bash
python experiments/exp1_robustness/deductive_refinement/apply_gar_rules_to_noisy_graph.py \
  --dataset DDA \
  --input-noisy-csv /path/to/noisy.csv \
  --rules-file /path/to/deduped_rules.txt \
  --pattern-instances-file /path/to/pattern_instances.jsonl \
  --output-dir experiments/exp1_robustness/results/dda_10pct
```

批量探针入口：

- `gar_denoise_probe_runner.py OUTPUT_ROOT`
- `gar_denoise_noise_only_probe.py OUTPUT_ROOT DDA TI`

`negative_edge_expander.py` 是规则负边导出实现，由上述流程和消融实验复用，通常不需要单独运行。

## 结果哪里看

所选 `--output-dir` 下：

- `<dataset>_gar_cleaned.csv`：删除规则判定噪声边后的图；
- `<dataset>_gar_removed_edges.csv`：被移除的边；
- `<dataset>_gar_marked_all.csv`：所有边及规则命中标记；
- `<dataset>_gar_denoise_summary.csv`：规则命中、移除和保留统计。

批量 probe 按 `<output-root>/<dataset>/<noise-ratio>/` 保存同类文件。

`simulate_rule_deductive_refinement.py` 默认输出
`outputs/simulated_rdr/simulated_rdr_summary.csv` 和
`simulated_rdr_summary_mean_std.csv`；它是 oracle/反事实敏感性分析。

## 对应论文

- 真实规则修正前后的 Precision/Recall/F1 对应 `tab-deductive-refinement`。
- 各噪声比例下，重新训练模型得到的 Accuracy/Precision/Recall/F1 对应
  `tab-noise-robustness`。
- cleaned/removed 数量和 probe 日志是诊断数据，不直接填论文表。
- simulated RDR 的任何数值都不能作为真实实验结果。

