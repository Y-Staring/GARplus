# 翻转噪声上的规则挖掘与修正

所有命令均从仓库根目录运行。先执行 `python <脚本> --help` 查看完整参数。

## 主流程：GAR+ 挖掘并修正

### 1. 挖掘规则

入口：`flip_subgraph_garplus.py`。

```bash
python experiments/exp1_robustness/rule_mining/flip_subgraph_garplus.py \
  --dataset ti \
  --train-c experiments/exp1_robustness/rule_mining/mappings/ti20_train_c.txt \
  --node-map experiments/exp1_robustness/rule_mining/mappings/ti/node_labeled.csv \
  --big-nodes enumeration-discovery/data/node.csv \
  --big-edges enumeration-discovery/data/edges.csv \
  --output-dir experiments/exp1_robustness/results/ti_20pct \
  --hops 1 --overfit-noise --enable-bn
```

结果目录中查看：

- `deduped_rules.txt`：最终 GAR+ 规则；
- `pattern_instances.jsonl`：规则对应的 pattern 匹配实例；
- `subgraph_edges.csv`、`subgraph_nodes.csv`：实际挖掘子图；
- `target_edges_mapped.csv`：翻转训练边到大图的映射；
- `subgraph_summary.json`：子图规模和映射统计。

### 2. 应用多分类规则

入口：`apply_flip_garplus_rules.py`，输入为上一步的同一结果目录。

```bash
python experiments/exp1_robustness/rule_mining/apply_flip_garplus_rules.py \
  --result-dir experiments/exp1_robustness/results/ti_20pct \
  --min-confidence 0.75 --score-margin 1.2
```

结果目录中查看：

- `garplus_denoise_predictions.csv`：逐边原标签、噪声标签、规则命中和修正标签；
- `train_garplus_denoised.txt`：供下游模型重新训练的修正训练集；
- `garplus_denoise_summary.json`：修正前后正确率、净增益和命中统计。

### 3. 应用定向负规则

入口：`apply_negative_targeted_garplus_rules.py`。

```bash
python experiments/exp1_robustness/rule_mining/apply_negative_targeted_garplus_rules.py \
  --result-dir experiments/exp1_robustness/results/ti_20pct \
  --min-confidence 0.80 --min-lift 1.50 --min-matched-rules 1
```

重点结果：`negative_targeted_summary.json`、
`negative_targeted_predictions.csv`、`per_rule_negative_summary.csv`。

### 4. 一次运行 TI/DDA × 5%/10%/20%

```bash
GARPLUS_ROOT=/home/yyyy/codework/GARplus \
GARPLUS_PYTHON=/home/yyyy/anaconda3/envs/digress/bin/python \
bash experiments/exp1_robustness/rule_mining/run_all_flip_garplus.sh
```

定向负规则批量入口为 `run_all_negative_targeted_garplus.sh`。批量汇总文件分别为
`all_denoise_summaries.json` 和 `all_negative_targeted_summaries.json`。

## 对照诊断流程

`mine_flip_denoise_rules.py` 是决策树/OOF 对照，不是 GAR+ 主算法。输出为
`rules.txt`、`denoise_predictions.csv`、`train_denoised_oof.txt` 和
`summary.json`。它只能作为诊断或 baseline，不能作为 GAR+ 的论文结果。

## 与论文对应关系

- 规则数量、置信度和规则实例可辅助核对 `tab-rule-statistics`，但只有采用论文统一配置的正式运行才能填表。
- `garplus_denoise_summary.json` 和 `negative_targeted_summary.json` 给出规则修正本身的效果诊断。
- 论文 `tab-noise-robustness` 需要使用修正后的训练集重新训练 TransE、RotatE 或 NBFNet，再填写 Accuracy、Precision、Recall、F1；不能直接把训练标签修正率写进该表。
- `subgraph_summary.json`、脚本日志和模拟/oracle 字段只用于完整性检查，不进入论文表格。

