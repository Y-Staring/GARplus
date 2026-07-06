# Agent Handoff: GARplusMiner Performance Experiments

本文是给后续 agent 的交接文档。目标是：后续 agent 在不重新阅读整个工程的情况下，只读本文件就能继续做 GARplusMiner 的性能实验、参数修改、运行、日志排查和结果解释。

使用前提：先进入 `GARplusMiner` 工程根目录。本文后续所有路径均以该目录为当前工作目录，除非特别说明。

```text
.
```

当前结论以 `GARplusMiner` 工程根目录为准。不要把其他临时目录中的额外 baseline 文件当作目标工程文件；目标工程当前没有 `gar_batch_pattern_size_*.py`、`gfd_batch_pattern_size_*.py`、`gar_demo_runner.py`、`gfd_demo_runner.py`。

## 1. 任务背景

用户要求整理“代码运行性能实验”的完整方法，并进一步要求生成一个 `agent.md`，让后续 agent 可以不读工程、只读该文档就开始工作。

本工程中的主要后续工作是修改 GARplus 图规则挖掘逻辑，包括 pattern 生成、predicate 生成、规则过滤/去重、BN 剪枝和 global rematch 等。性能实验脚本主要作为修改后的回归验证工具，用来检查不同数据集、不同 pattern size、不同规则阈值下的运行时间和规则产出是否符合预期。

本工程不是 LLM 推理 benchmark：

- 没有 prompt length / output length / tokens/s。
- 没有 profiling delay / profiling duration。
- 没有多卡并行参数。
- 没有大模型路径参数。

如果用户提到“模型路径”，在这个性能实验中通常应理解为 sampled graph `.pt` 路径或 `similarity.py` 的 embedding 模型路径；但 `similarity.py` 不在当前 GARplus 性能实验主链路中。

## 2. 工程定位

`GARplusMiner` 是 BN-guided GAR+ 的 Python 实现，主要流程如下：

1. 读取 PPI / DDA / TI 图数据。
2. 从 sampled `.pt` 或 CSV 构造 `DataGraph`。
3. 用 `GraphSpawn` 做 VSpawn-style pattern extension。
4. 可选使用 Pattern BN 剪枝 pattern 扩展候选。
5. 可选注入 sampled frequent patterns。
6. 可选做 global rematch。
7. 用 decision-tree 或 FP-Growth 做 predicate rule mining。
8. 可选使用 Predicate BN 剪枝 predicate features。
9. 输出日志、汇总 CSV、deduped rules、pattern instances。

核心 runner：

```text
garplus_demo_runner.py
```

核心配置入口：

```text
ppi_demo.py
dda_demo.py
ti_demo.py
```

核心批量实验入口：

```text
batch_pattern_size_ppi.py
batch_pattern_size_dda.py
batch_pattern_size_ti.py
batch_sigma_confidence_ppi.py
batch_sigma_confidence_dda.py
batch_sigma_confidence_ti.py
```

## 3. 已确认的目标目录文件

`GARplusMiner` 工程根目录中，与性能实验最相关的文件如下：

| 文件 | 作用 | 后续 agent 是否常改 |
| --- | --- | --- |
| `batch_pattern_size_ppi.py` | PPI 上跑 pattern size sweep。 | 是 |
| `batch_pattern_size_dda.py` | DDA 上跑 pattern size sweep。 | 是 |
| `batch_pattern_size_ti.py` | TI 上跑 pattern size sweep。 | 是 |
| `batch_sigma_confidence_ppi.py` | PPI 上跑 sigma / confidence sweep。 | 是 |
| `batch_sigma_confidence_dda.py` | DDA 上跑 sigma / confidence sweep。 | 是 |
| `batch_sigma_confidence_ti.py` | TI 上跑 sigma / confidence sweep。 | 是 |
| `garplus_demo_runner.py` | GARplus 主流程和绝大多数参数定义。 | 高级修改 |
| `ppi_demo.py` | PPI 数据集配置，默认 `mode="fp-growth"`。 | 是 |
| `dda_demo.py` | DDA 数据集配置，默认根据 debug 环境变量选择 `pattern-only` 或 `decision-tree`。 | 是 |
| `ti_demo.py` | TI 数据集配置，默认 `mode="fp-growth"`。 | 是 |
| `graph_types.py` | 图、pattern、instance 等数据结构。 | 通常不改 |
| `pattern_extension.py` | VSpawn pattern extension。 | 高级修改 |
| `pattern_bn.py` | Pattern BN 训练和剪枝。 | 高级修改 |
| `predicate_bn.py` | Predicate BN 训练和特征剪枝。 | 高级修改 |
| `predicate_selection.py` | DecisionTree / FPGrowth 规则挖掘。 | 高级修改 |
| `sampled_pt_loader.py` | sampled `.pt` 图加载。 | 路径/数据结构变动时改 |
| `relation_sampled_loader.py` | DDA/TI 关系图加载。 | 路径/数据结构变动时改 |
| `ppi_loader.py` | PPI CSV 和 PPI sampled graph 加载。 | 路径/数据结构变动时改 |
| `garplus_ml_predicates.py` | ML predicate 注入。 | 需要改离线 predicate 时改 |
| `predicate_enrichment.py` | 额外 predicate enrichment。 | 高级修改 |
| `negative_edge_expander.py` | 负边扩展辅助脚本，不是本轮性能主入口。 | 仅相关任务改 |
| `build_three_gnn_edge_csv.py` | GNN edge CSV 构建辅助脚本，不是本轮性能主入口。 | 仅相关任务改 |
| `README.md` / `GAR_FLOW.md` | 早期说明文档；PowerShell 默认读取时可能出现中文乱码。 | 只做参考 |

## 4. 两类性能实验

### 4.1 Pattern Size Sweep

入口脚本：

```text
batch_pattern_size_ppi.py
batch_pattern_size_dda.py
batch_pattern_size_ti.py
```

默认参数：

```python
ALGORITHM = "GARplus"
PATTERN_SIZES = (2, 4, 6, 8)
MAX_RADIUS = 8
MAX_ADD_EDGE = 8
NODE_MAX_ADD_EDGE = 8
PATTERN_SUPPORT = 1
PATTERN_EXTENSION_DEBUG = True
PATTERN_EXTENSION_DEBUG_LIMIT = 200
```

每轮运行时，脚本会调用：

```python
run_demo(config_for_pattern_size(pattern_size))
```

`config_for_pattern_size()` 会基于对应 demo 文件里的 `CONFIG` 做 `dataclasses.replace()`，主要覆盖：

```python
min_pattern_nodes=pattern_size
max_pattern_nodes=pattern_size
max_radius=MAX_RADIUS
max_add_edge=MAX_ADD_EDGE
node_max_add_edge=NODE_MAX_ADD_EDGE
pattern_support=PATTERN_SUPPORT
pattern_extension_debug=PATTERN_EXTENSION_DEBUG
pattern_extension_debug_limit=PATTERN_EXTENSION_DEBUG_LIMIT
deduped_rules_output_path=...
pattern_instances_output_path=...
```

因此 pattern size sweep 的核心含义是：固定每轮只挖节点数等于 `pattern_size` 的 pattern，然后比较不同 pattern 节点规模下的耗时和 pattern 数量。

输出目录：

```text
batch_results\pattern_size_ppi
batch_results\pattern_size_dda
batch_results\pattern_size_ti
```

日志命名：

```text
garplus_<dataset>_n<pattern_size>.log
```

汇总 CSV：

```text
garplus_<dataset>_pattern_size_timing.csv
```

CSV 字段包括：

```text
algorithm,dataset,pattern_size,min_pattern_nodes,max_pattern_nodes,
max_radius,max_add_edge,node_max_add_edge,pattern_support,
patterns_mined,rule_mining_seconds,wall_seconds,status,error,log_path
```

### 4.2 Sigma / Confidence Sweep

入口脚本：

```text
batch_sigma_confidence_ppi.py
batch_sigma_confidence_dda.py
batch_sigma_confidence_ti.py
```

默认参数：

```python
PATTERN_SIZE = 4
DEFAULT_SIGMA = 50
DEFAULT_CONFIDENCE = 0.7
SIGMAS = (50, 100, 150, 200)
CONFIDENCES = (0.3, 0.5, 0.7, 0.9)
MAX_RADIUS = 8
MAX_ADD_EDGE = 8
NODE_MAX_ADD_EDGE = 8
PATTERN_SUPPORT = 1
```

运行组合：

```python
runs = (
    [("sigma", sigma, DEFAULT_CONFIDENCE) for sigma in SIGMAS]
    + [("confidence", DEFAULT_SIGMA, confidence) for confidence in CONFIDENCES]
)
```

也就是说它不是完整网格搜索，而是两条单变量曲线：

- 固定 `confidence=0.7`，扫 `sigma=50,100,150,200`。
- 固定 `sigma=50`，扫 `confidence=0.3,0.5,0.7,0.9`。

每轮会覆盖：

```python
min_pattern_nodes=PATTERN_SIZE
max_pattern_nodes=PATTERN_SIZE
min_support=sigma
min_confidence=confidence
tau_x=confidence
```

注意：这里 `confidence` 同时影响 `min_confidence` 和 `tau_x`。这意味着 confidence sweep 同时改变规则置信度阈值和 Predicate BN 阈值，报告实验时需要说明。

输出目录：

```text
batch_results\sigma_confidence_ppi
batch_results\sigma_confidence_dda
batch_results\sigma_confidence_ti
```

日志命名：

```text
garplus_<dataset>_<varying_param>_s<sigma>_c<confidence_tag>_n<pattern_size>.log
```

例如：

```text
garplus_ppi_sigma_s100_c0p7_n4.log
garplus_ppi_confidence_s50_c0p9_n4.log
```

汇总 CSV：

```text
garplus_<dataset>_sigma_confidence_timing.csv
```

CSV 字段包括：

```text
algorithm,dataset,pattern_size,varying_param,sigma,confidence,
min_pattern_nodes,max_pattern_nodes,max_radius,max_add_edge,node_max_add_edge,
pattern_support,min_support,min_confidence,tau_x,patterns_mined,
raw_rules,deduped_rules,rule_mining_seconds,wall_seconds,status,error,log_path
```

## 5. 数据路径和环境变量

三个 demo 文件都使用类似路径逻辑：

```python
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_DATA_SUBDIR = "去病图数据"
DATA_DIR = Path(os.environ.get("GARPLUS_DATA_DIR", str(BASE_DIR / DEFAULT_DATA_SUBDIR)))
PROCESSED_DIR = Path(os.environ.get("GARPLUS_PROCESSED_DIR", str(BASE_DIR / "processed")))
```

工程根目录是：

```text
.
```

所以默认父目录是：

```text
..
```

默认数据目录：

```text
..\去病图数据
```

默认 processed 目录：

```text
..\processed
```

推荐显式设置环境变量：

```powershell
$env:GARPLUS_DATA_DIR="<DATA_DIR>"
$env:GARPLUS_PROCESSED_DIR="<PROCESSED_DIR>"
```

常用环境变量：

| 环境变量 | 作用 | 默认值/说明 |
| --- | --- | --- |
| `GARPLUS_DATA_DIR` | CSV 数据目录。 | 父目录下 `去病图数据`。 |
| `GARPLUS_PROCESSED_DIR` | sampled `.pt`、BN cache、离线 predicate、规则输出目录。 | 父目录下 `processed`。 |
| `GARPLUS_PATTERN_DEBUG` | DDA/TI 中控制是否用 `pattern-only` debug 模式。 | 默认 `0`。DDA/TI demo 会读取。 |
| `GARPLUS_PATTERN_DEBUG_LIMIT` | pattern debug 输出上限。 | 默认 `500`。 |
| `GARPLUS_DEBUG_MATCH_EXPANSION` | DDA 中控制 match expansion debug。 | 默认 `1`。 |
| `GARPLUS_DEBUG_TRANSACTION_COST` | DDA 中控制 transaction cost debug。 | 默认 `1`。 |
| `GARPLUS_DEBUG_SAMPLE_MATCHES` | DDA 中 sample match 数量。 | 默认 `3`。 |
| `GARPLUS_STRUCTURAL_EDGE_LABEL` | 是否启用 structural edge label。 | 默认 `0`。PPI/DDA/TI 都读取。 |
| `GARPLUS_STRUCTURAL_EDGE_LABEL_ATTR` | structural edge label 使用的属性名。 | PPI 默认 `experimental_system_type`；DDA/TI 默认 `direct_evidence_category`。 |

数据文件：

| 数据集 | CSV 边文件 | 节点文件 | sampled `.pt` |
| --- | --- | --- | --- |
| PPI | `protein_protein_signed.csv` | `protein.csv` | `processed\ppi\ppi_selected.pt` |
| DDA | `drug_disease_signed.csv` | `drug.csv`, `disease.csv` | `processed\dda\dda_selected.pt` |
| TI | `gene_disease_signed.csv` | `gene.csv`, `disease.csv` | `processed\ti\ti_selected.pt` |

PPI 的 `ppi_demo.py` 还配置了：

```python
ml_predicates=MLPredicateConfig(
    enabled=True,
    equivalence_threshold=0.80,
    similarity_threshold=0.85,
    precomputed_edge_csv_path="<PRECOMPUTED_EDGE_CSV>",
    offline_csv_path=str(PROCESSED_DIR / "ppi" / "ml_predicates.csv"),
)
```

DDA/TI 类似，也有 Linux 风格的 `precomputed_edge_csv_path`，但同时有 `offline_csv_path` 指向 `PROCESSED_DIR`。如果 Windows 下运行报找不到预计算 predicate，优先检查 `processed\<dataset>\ml_predicates.csv` 是否存在，以及 `garplus_ml_predicates.py` 的读取优先级。

## 6. Dataset Config 摘要

### 6.1 PPI: `ppi_demo.py`

关键默认配置：

```python
dataset_name="PPI"
mode="fp-growth"
interaction_csv_path=DATA_DIR / "protein_protein_signed.csv"
node_csv_path=DATA_DIR / "protein.csv"
sampled_pt_path=PROCESSED_DIR / "ppi" / "ppi_selected.pt"
force_edge_label="candidate_interaction"
edge_label_column="Experimental System"
include_ml_predicate_targets=False
include_edge_existing_target=True
inject_sampled_frequent_patterns=True
topology_only_pattern_dedup=True
global_rematch_max_pattern_edges=3
pattern_dedup_prefer_target_value="negative"
enable_target_recall=False
enable_rule_payload_generation=False
max_radius=2
max_add_edge=2
filter_degree_predicates=True
```

注意：批量脚本会把 `max_radius` 和 `max_add_edge` 覆盖成 `8`，把 `pattern_support` 覆盖成 `1`，把 `min_pattern_nodes/max_pattern_nodes` 覆盖成当前实验值。

### 6.2 DDA: `dda_demo.py`

关键默认配置：

```python
dataset_name="DDA"
mode="pattern-only" if PATTERN_DEBUG else "decision-tree"
interaction_csv_path=DATA_DIR / "drug_disease_signed.csv"
sampled_pt_path=PROCESSED_DIR / "dda" / "dda_selected.pt"
force_edge_label="drug_disease"
edge_label_column="EdgeLabel"
include_ml_predicate_targets=False
include_edge_existing_target=True
undirected=False
undirected_pattern=False
decision_tree_max_depth=4
max_radius=2
max_add_edge=2
topology_only_pattern_dedup=True
topology_dedupe_respect_direction=True
drop_target_entity_features=True
ignored_target_values=("unknown", "neutral")
drop_ignored_target_edges=True
```

DDA 的 `RELATION` 会加载 drug / disease 节点属性，并排除大量非规则挖掘字段。DDA 还配置了 `PredicateEnrichmentConfig(inference_edge_predicates=True, inference_presence_key="inferencechemicalname")`。

### 6.3 TI: `ti_demo.py`

关键默认配置：

```python
dataset_name="TI"
mode="fp-growth"
fp_growth_max_itemset_size=4
interaction_csv_path=DATA_DIR / "gene_disease_signed.csv"
sampled_pt_path=PROCESSED_DIR / "ti" / "ti_selected.pt"
force_edge_label="gene_disease"
edge_label_column="EdgeLabel"
include_ml_predicate_targets=False
include_edge_existing_target=False
undirected=False
undirected_pattern=False
topology_only_pattern_dedup=True
topology_dedupe_respect_direction=True
global_match_scope="sampled"
max_radius=3
max_add_edge=2
enable_rule_payload_generation=False
ignored_target_values=("unknown", "neutral")
drop_ignored_target_edges=True
```

TI 也配置了 `PredicateEnrichmentConfig(inference_edge_predicates=True, inference_presence_key="inferencegenesymbol")`。

## 7. Runner 内部逻辑

核心函数：

```python
garplus_demo_runner.run_demo(cfg: GarplusRunConfig)
```

可以把 `run_demo()` 理解为 6 个阶段：路径与配置确认、图加载、pattern 生成、pattern 全局重匹配、规则挖掘、结果汇总。后续 agent 排错时应按这个顺序看日志。

### 7.1 路径与配置确认

`run_demo()` 开始后先打印：

```text
=== GAR <dataset> Demo ===
[RunStart] dataset=<dataset>
```

然后解析输入路径：

```python
interaction_csv_path = resolve_path(cfg.interaction_csv_path, cfg.fallback_interaction_name, cfg.auto_discover_if_missing)
node_csv_path = resolve_path(cfg.node_csv_path, cfg.fallback_node_name, cfg.auto_discover_if_missing)
```

路径解析规则：

- 如果 `cfg.interaction_csv_path` / `cfg.node_csv_path` 已有值，直接使用。
- 如果路径为空且 `auto_discover_if_missing=False`，会抛出找不到文件的异常。
- 如果 `auto_discover_if_missing=True`，才会按 fallback 文件名搜索。

随后打印配置快照：

```text
[Input] interaction_csv=...
[Input] <node_csv_label>=...
[Config] dataset=... mode=... max_rows=... y_key=... min_value_support_count=...
[Pruning] tau_p=... tau_x=... predicate_focus=...
[PatternConfig] support=... max_radius=... max_add_edge=... node_max_add_edge=...
[BN] pattern_bn=... predicate_bn=...
[PatternDebug] only=... enabled=... event_limit=...
[PatternMatching] global_rematch_patterns=... global_match_scope=...
[Targets] include_ml_predicate_targets=...
[PatternMode] undirected_pattern=...
[StructuralEdgeLabel] enabled=... base=... attr=...
```

这些日志是判断批量脚本覆盖参数是否生效的第一入口。例如 pattern size sweep 会把 `[PatternConfig]` 里的 `min_pattern_nodes/max_pattern_nodes` 改成当前 `pattern_size`。

### 7.2 图加载与图预处理

`run_demo()` 调用 `load_graph(cfg, interaction_csv_path, node_csv_path)`。

`load_graph()` 的分支：

```python
if cfg.use_sampled_pt_graph:
    cfg.sampled_graph_loader(...)
else:
    cfg.csv_graph_loader(...)
```

当 `cfg.use_sampled_pt_graph=True`：

- 必须配置 `cfg.sampled_pt_path`。
- 必须配置 `cfg.sampled_graph_loader`。
- 日志会打印 `[Input] sampled_pt=...`。
- loader 会收到 `interaction_path`、`protein_path`、`edge_label_column`、`force_edge_label`、`augment_negative_edges`、`negative_edge_limit`、`balance_edge_labels`、`structural_edge_label_enabled`、`structural_edge_label_attr` 等参数。

当 `cfg.use_sampled_pt_graph=False`：

- 必须配置 `cfg.csv_graph_loader`。
- 直接从 CSV 构建图。

图加载后，可能发生这些处理：

- 如果 `cfg.drop_ignored_target_edges=True`，会删除 target 值属于 `cfg.ignored_target_values` 的边，并打印 `[TargetEdgeFilter]`。
- 如果不是 `pattern_extension_only`，会调用 `inject_ml_predicates(graph, cfg.dataset_name, cfg.ml_predicates)` 注入 ML predicate。
- 之后调用 `enrich_numeric_bin_predicates(graph, cfg.predicate_enrichment)` 做 predicate enrichment。
- 再调用 `build_target_y_list(graph, cfg)` 生成规则挖掘目标 `target_y_list`。

相关日志：

```text
[MLPredicate] ...
[PredicateEnrichment] ...
[YList] targets=[...]
[Graph] vertices=... out_edge_lists=... isolated_vertices=...
```

如果日志没有 `[YList]`，通常说明在图加载或 predicate 注入阶段已经失败，或者当前是 `pattern_extension_only=True`。

### 7.3 Sampled Frequent Patterns 和 Pattern BN

如果 `cfg.enable_sampled_frequent_patterns=True`，runner 会先调用：

```python
mine_sampled_frequent_patterns(...)
edge_priors_from_frequent_patterns(...)
```

这些结果会作为 Pattern BN 的 edge priors。日志会出现：

```text
[SampledFSM] min_graph_support=... frequent=... edge_priors=...
```

如果 `cfg.enable_pattern_bn=True`，会调用：

```python
PatternBayesianNetwork.fit_graph(
    graph,
    PatternBNConfig(
        min_score=cfg.tau_p,
        top_k_per_spawn_node=cfg.pattern_bn_top_k_per_spawn_node,
        cache_path=cfg.pattern_bn_cache_path,
        retrain=cfg.retrain_pattern_bn,
        ...
    ),
)
```

影响：

- `tau_p` 越高，Pattern BN 剪枝越强。
- `pattern_bn_top_k_per_spawn_node` 如果不为空，会限制每个扩展节点保留的候选结构动作数。
- `pattern_bn_cache_path` 和 `retrain_pattern_bn` 决定是否复用缓存。

### 7.4 VSpawn Pattern Extension

runner 先由数据集配置里的 `seed_builder` 构造 seed：

```python
seed = cfg.seed_builder(graph)
```

然后构造：

```python
spawn = GraphSpawn(
    graph,
    [seed],
    options=PatternOptions(
        pattern_support_threshold=cfg.pattern_support,
        max_radius=cfg.max_radius,
        max_add_edge=cfg.max_add_edge,
        node_max_add_edge=cfg.node_max_add_edge,
        max_pattern_nodes=cfg.max_pattern_nodes,
        full_solution=cfg.full_solution,
        max_multi_support=cfg.max_multi_support,
        undirected_pattern=cfg.undirected_pattern,
        topology_only_dedup=cfg.topology_only_pattern_dedup,
        topology_dedupe_respect_direction=cfg.topology_dedupe_respect_direction,
        global_vspawn_instances=cfg.global_vspawn_instances,
        extension_debug=cfg.pattern_extension_debug,
        extension_debug_limit=cfg.pattern_extension_debug_limit,
    ),
    pattern_bn=pattern_bn,
)
```

核心循环：

```python
while spawn.unstoppable():
    round_generated = spawn.vspawn()
    generated.extend(round_generated)
```

每轮打印：

```text
[VSpawn] round=... generated=... total=...
[VSpawnStats] round=... candidates_seen=... bn_pruned=... duplicate_pruned=... constraint_pruned=... no_match_pruned=... support_pruned=...
```

排错含义：

- `candidates_seen` 很大：搜索空间大，优先降低 `MAX_RADIUS/MAX_ADD_EDGE/NODE_MAX_ADD_EDGE`。
- `bn_pruned` 很大：Pattern BN 剪枝强，检查 `tau_p`。
- `support_pruned` 很大：`pattern_support` 过滤强。
- `generated` 一直为 0：可能 pattern support 太高，或者图/edge label 配置不匹配。

### 7.5 注入 Sampled Structural Patterns

如果 `cfg.inject_sampled_frequent_patterns=True` 且前面挖到了 `frequent_sampled`，runner 会调用：

```python
build_directed_frequent_patterns(...)
```

然后把注入的 structural patterns 加入 `generated`：

```python
generated.extend(sampled_structural_patterns)
```

相关日志：

```text
[SampledFSMInject] injected=... limit=... include_edge=... materialized=...
```

如果 `generated` 为空，runner 会直接抛错：

```text
No pattern generated. Try lowering pattern_support or increasing max_radius/max_add_edge.
```

### 7.6 Global Rematch

如果 `cfg.global_rematch_patterns=True`，VSpawn 生成的 pattern 会被重新匹配：

```python
find_matches_with_limit(
    item.pattern,
    rematch_graph,
    rematch_limit,
    target_edge_index=cfg.global_rematch_target_edge_index,
    max_instances_per_target_edge=cfg.global_rematch_max_instances_per_target_edge,
    target_edge_undirected=cfg.undirected_pattern,
)
```

`rematch_graph` 的选择：

- `cfg.global_match_scope == "sampled"`：用前面加载的 sampled graph。
- `cfg.global_match_scope == "original"`：调用 `load_verification_graph()` 重新加载原始 CSV 图。

相关控制参数：

- `global_rematch_max_instances`：全局匹配实例上限。
- `global_rematch_max_pattern_edges`：pattern 边数超过该值时可跳过 global rematch。
- `global_rematch_target_edge_index`：作为目标边的 pattern edge index。
- `global_rematch_max_instances_per_target_edge`：每条目标边最多保留多少实例。

相关日志：

```text
[GlobalRematchGraph] mode=... vertices=... edges=...
[GlobalRematch] pattern_id=... kept=True/False backend=vf3_linux incremental_multi=... global_multi=... global_single=...
```

如果很多 pattern 被 `support<...` 过滤，说明 global rematch 后支持度不足。

### 7.7 Pattern 去重、过滤和实例保存

global rematch 后，runner 会对 pattern 去重：

- 如果 `cfg.topology_only_pattern_dedup=True`，使用 `topology_pattern_code(...)`。
- 否则根据 `cfg.undirected_pattern` 使用 `undirected_canonical_code()` 或 `canonical_code()`。
- 如果 `cfg.pattern_dedup_prefer_target_value` 不为空，会优先保留覆盖该 target value 更多的 pattern。

然后按节点数过滤：

```python
cfg.min_pattern_nodes <= item.pattern.node_count() <= cfg.max_pattern_nodes
```

最后得到：

```python
patterns_to_mine
```

关键日志：

```text
[Patterns] generated_total=... unique_total=... mining_total=... deduped=... size_filtered=... min_pattern_nodes=... max_pattern_nodes=...
```

批量脚本的 `patterns_mined` 就是从这行的 `mining_total` 提取。

如果 `cfg.save_pattern_instances=True`，会调用 `save_pattern_instances(...)` 写 jsonl：

```text
[PatternInstances] wrote=... path=...
```

如果 `cfg.pattern_extension_only=True`，runner 在这里打印 pattern extension result 后直接返回，不进入规则挖掘。因此批量脚本可能提取不到 `rule_mining_seconds`。

### 7.8 规则挖掘

进入规则挖掘前，runner 记录：

```python
rule_mining_started = time.perf_counter()
```

规则挖掘循环结构是：

```python
for pattern_index, target_pattern in enumerate(patterns_to_mine, start=1):
    for y_key in target_y_list:
        focus_items = predicate_focus_items_for_y_key(cfg, y_key)
        for focus_item in focus_items:
            ...
```

每个 pattern 开始时打印：

```text
[PatternStart] dataset=... pattern_index=... pattern_id=... labels=... edges=... single_support=... multi_support=...
```

每个 y target 打印：

```text
[YTarget] dataset=... pattern_id=... y_key=...
[PredicateFocus] dataset=... pattern_id=... y_key=... focus=...
```

如果 `cfg.enable_predicate_bn=True`，每个 `(y_key, focus_item)` 会创建或复用 `PredicateBayesianNetwork`：

```python
PredicateBNConfig(
    target_key=y_key,
    min_score=cfg.tau_x,
    top_k_features=cfg.predicate_bn_top_k_features,
    focus_target_item=focus_item,
    cache_path=predicate_bn_cache_for_y_key(cfg, y_key, focus_item),
    retrain=cfg.retrain_predicate_bn,
    ...
)
```

然后根据 `cfg.mode` 选择 selector：

- `cfg.mode == "decision-tree"`：使用 `DecisionTreePredicateSelector`。
- `cfg.mode == "fp-growth"`：使用 `FPGrowthPredicateSelector`，并额外使用 `cfg.fp_growth_max_itemset_size`。

两个 selector 都会使用：

```python
min_support=cfg.min_support
min_confidence=cfg.min_confidence
min_value_support_count=cfg.min_value_support_count
drop_target_values=cfg.ignored_target_values
drop_feature_key_tokens=cfg.ignored_predicate_key_tokens
drop_target_entity_features=cfg.drop_target_entity_features
```

相关日志：

```text
[PredicateSelection/DecisionTree] ...
[PredicateSelection/FPGrowth] ...
[PredicateFilter] ...
[PredicateFocusMerge] ...
[TargetRecall] ...
[RuleGeneration] ...
```

sigma/confidence sweep 中：

- `sigma` 覆盖 `cfg.min_support`。
- `confidence` 覆盖 `cfg.min_confidence`。
- `confidence` 同时覆盖 `cfg.tau_x`。

因此 confidence sweep 会同时影响规则过滤和 Predicate BN 特征剪枝。

### 7.9 规则 payload、去重和最终汇总

每个 focus 生成的规则会按 `(antecedent, consequent)` 合并，保留 `(confidence, support, lift)` 更高的版本。

如果 `cfg.enable_rule_payload_generation=True`：

1. 用 `predicate_rule_to_zl(...)` 转成 ZLRule。
2. 用 `zl_rule_filter(...)` 过滤。
3. 用 `send_zl_rules(...)` 写入 `RuleSender`。
4. 打印 `[RuleGeneration] ... filtered=... sent=...`。

如果 `enable_rule_payload_generation=False`，会跳过 payload 生成，但仍然统计 raw rules / deduped rules。

规则挖掘结束后打印：

```text
[Timing] stage=rule_mining_total algorithm=GARplus dataset=... patterns=... raw_rules=... seconds=...
```

批量脚本的 `rule_mining_seconds` 就是从这行提取。

最后：

1. 调用 `print_deduped_rules(all_pattern_rules, cfg.print_deduped_rule_limit, cfg.deduped_rules_output_path)`。
2. 统计 raw / deduped consequent distribution。
3. 打印 discovered rule stats table。
4. 打印最终 `[Summary]`：

```text
[Summary] dataset=... patterns_mined=... raw_rules=... deduped_rules=... positive_rules=... negative_rules=... negative_ratio=... avg_antecedent_size=... avg_pattern_size=... avg_confidence=... total_sent=... pattern_instances=... pattern_instances_path=...
```

sigma/confidence sweep 的 `raw_rules` 和 `deduped_rules` 就是从这行提取。

批量脚本依靠正则从日志提取：

```python
TIMING_RE = re.compile(r"\[Timing\] stage=rule_mining_total .*?seconds=([0-9.]+)")
PATTERNS_RE = re.compile(r"\[Patterns\].*?mining_total=(\d+)")
RULES_RE = re.compile(r"\[Summary\].*?raw_rules=(\d+).*?deduped_rules=(\d+)")
```

`RULES_RE` 只在 sigma/confidence sweep 中使用。

## 8. 运行命令

后续 agent 应优先进入工程根目录运行：

```powershell
cd "<GARPLUSMINER_DIR>"
```

设置环境变量：

```powershell
$env:GARPLUS_DATA_DIR="<DATA_DIR>"
$env:GARPLUS_PROCESSED_DIR="<PROCESSED_DIR>"
```

如果要减少 debug 噪声：

```powershell
$env:GARPLUS_PATTERN_DEBUG="0"
$env:GARPLUS_DEBUG_MATCH_EXPANSION="0"
$env:GARPLUS_DEBUG_TRANSACTION_COST="0"
```

### 8.1 最小可运行命令

建议先跑 PPI pattern size sweep：

```powershell
cd "<GARPLUSMINER_DIR>"
$env:GARPLUS_DATA_DIR="<DATA_DIR>"
$env:GARPLUS_PROCESSED_DIR="<PROCESSED_DIR>"
python batch_pattern_size_ppi.py
```

如果想更快验证，先把 `batch_pattern_size_ppi.py` 里的：

```python
PATTERN_SIZES = (2, 4, 6, 8)
```

改成：

```python
PATTERN_SIZES = (2,)
```

### 8.2 Pattern Size Sweep 全部命令

```powershell
cd "<GARPLUSMINER_DIR>"
$env:GARPLUS_DATA_DIR="<DATA_DIR>"
$env:GARPLUS_PROCESSED_DIR="<PROCESSED_DIR>"

python batch_pattern_size_ppi.py
python batch_pattern_size_dda.py
python batch_pattern_size_ti.py
```

### 8.3 Sigma / Confidence Sweep 全部命令

```powershell
cd "<GARPLUSMINER_DIR>"
$env:GARPLUS_DATA_DIR="<DATA_DIR>"
$env:GARPLUS_PROCESSED_DIR="<PROCESSED_DIR>"

python batch_sigma_confidence_ppi.py
python batch_sigma_confidence_dda.py
python batch_sigma_confidence_ti.py
```

### 8.4 完整推荐运行命令

```powershell
cd "<GARPLUSMINER_DIR>"
$env:GARPLUS_DATA_DIR="<DATA_DIR>"
$env:GARPLUS_PROCESSED_DIR="<PROCESSED_DIR>"
$env:GARPLUS_PATTERN_DEBUG="0"
$env:GARPLUS_DEBUG_MATCH_EXPANSION="0"
$env:GARPLUS_DEBUG_TRANSACTION_COST="0"

python batch_pattern_size_ppi.py
python batch_pattern_size_dda.py
python batch_pattern_size_ti.py

python batch_sigma_confidence_ppi.py
python batch_sigma_confidence_dda.py
python batch_sigma_confidence_ti.py
```

## 9. 输出和结果解释

### 9.1 Pattern Size Sweep 输出

| 输出 | 位置 | 内容 |
| --- | --- | --- |
| 日志 | `batch_results\pattern_size_<dataset>\garplus_<dataset>_n<size>.log` | 单轮完整 stdout/stderr。 |
| CSV | `batch_results\pattern_size_<dataset>\garplus_<dataset>_pattern_size_timing.csv` | 每个 pattern size 的耗时汇总。 |
| 规则 | `batch_results\pattern_size_<dataset>\deduped_rules_n<size>.txt` | 去重后的规则。 |
| pattern instances | `batch_results\pattern_size_<dataset>\pattern_instances_n<size>.jsonl` | pattern 匹配实例。 |

### 9.2 Sigma / Confidence Sweep 输出

| 输出 | 位置 | 内容 |
| --- | --- | --- |
| 日志 | `batch_results\sigma_confidence_<dataset>\garplus_<dataset>_<tag>.log` | 单轮完整 stdout/stderr。 |
| CSV | `batch_results\sigma_confidence_<dataset>\garplus_<dataset>_sigma_confidence_timing.csv` | sigma/confidence 单变量 sweep 汇总。 |
| 规则 | `batch_results\sigma_confidence_<dataset>\deduped_rules_s<sigma>_c<confidence>_n<size>.txt` | 去重规则。 |
| pattern instances | `batch_results\sigma_confidence_<dataset>\pattern_instances_s<sigma>_c<confidence>_n<size>.jsonl` | pattern 匹配实例。 |

### 9.3 关键字段解释

| 字段 | 含义 |
| --- | --- |
| `wall_seconds` | 批量脚本外层测量的单轮端到端时间，包括图加载、pattern extension、规则挖掘、日志写入等。 |
| `rule_mining_seconds` | runner 内部 `[Timing] stage=rule_mining_total` 的耗时，只覆盖规则挖掘阶段。 |
| `patterns_mined` | 进入规则挖掘的 pattern 数量，即 `[Patterns] ... mining_total=...`。 |
| `raw_rules` | 未去重规则数，只在 sigma/confidence sweep CSV 中自动提取。 |
| `deduped_rules` | 去重后规则数，只在 sigma/confidence sweep CSV 中自动提取。 |
| `status` | `ok` 或 `error`。 |
| `error` | 异常类型和消息。 |
| `log_path` | 对应完整日志。 |

实验成功判断：

- CSV 中 `status=ok`。
- `error` 为空。
- `rule_mining_seconds` 不是空字符串。
- `patterns_mined` 不是空字符串。
- 对应日志末尾有 `[BatchTiming]` 和 `[BatchSummary]`。
- 日志中没有 traceback。

PowerShell 查看：

```powershell
Get-ChildItem .\batch_results\pattern_size_ppi
Get-Content .\batch_results\pattern_size_ppi\garplus_ppi_n2.log -Tail 80
Get-Content .\batch_results\pattern_size_ppi\garplus_ppi_pattern_size_timing.csv
```

注意：PowerShell 默认编码显示中文可能乱码；这通常不影响 Python 读取 UTF-8 日志/文档。若需要人工阅读中文，使用支持 UTF-8 的编辑器打开。

## 10. 挖掘逻辑修改指南

后续工作重点不是只调批量脚本参数，而是修改规则挖掘算法本身。建议先判断需求落在哪个阶段，再改对应模块。

| 想修改的内容 | 首选文件/函数 | 影响范围 | 验证方式 |
| --- | --- | --- | --- |
| pattern 如何扩展、何时停止、候选如何剪枝 | `pattern_extension.py` 的 `GraphSpawn`、`PatternOptions`；`garplus_demo_runner.py` 构造 `GraphSpawn` 的位置 | 影响 `[VSpawn]`、`[VSpawnStats]`、`[Patterns]`、后续所有规则数量 | 先跑 `PATTERN_SIZES=(2,)`，看 `generated_total/unique_total/mining_total` 和日志 traceback。 |
| pattern 去重方式 | `pattern_extension.py` 的 canonical/topology code；`garplus_demo_runner.py` 的 unique pattern 选择逻辑 | 影响 `unique_total`、`deduped`、`patterns_mined` | 对比修改前后的 `[Patterns]` 和 `pattern_instances_n<size>.jsonl`。 |
| Pattern BN 剪枝逻辑 | `pattern_bn.py`、`garplus_demo_runner.py` 中 `PatternBayesianNetwork.fit_graph(...)` | 影响 `bn_pruned` 和 pattern 候选保留 | 看 `[VSpawnStats] bn_pruned=...` 和 `[SampledFSM]`。 |
| sampled frequent patterns 注入 | `sampled_frequent_patterns.py`、`garplus_demo_runner.py` 的 `[SampledFSMInject]` 阶段 | 影响额外注入的 structural patterns | 看 `[SampledFSMInject] injected=...`。 |
| global rematch 规则 | `vf3_linux.py`、`garplus_demo_runner.py` 的 `find_matches_with_limit(...)` 调用 | 影响 pattern 实例数、支持度过滤、规则覆盖 | 看 `[GlobalRematch] kept=True/False global_multi/global_single`。 |
| predicate 候选列如何生成、过滤、转 transaction/table | `predicate_selection.py` 的 `DecisionTreePredicateSelector` / `FPGrowthPredicateSelector` | 影响候选 literal、规则数和运行时间 | 看 `[PredicateFilter]`、候选诊断、`raw_rules`。 |
| decision-tree 规则生成逻辑 | `predicate_selection.py` 的 `DecisionTreePredicateSelector.generate_rules(...)` | 影响 `mode="decision-tree"` 的规则产出 | DDA 默认走 decision-tree，可优先用 DDA smoke test。 |
| FP-Growth 规则生成逻辑 | `predicate_selection.py` 的 `FPGrowthPredicateSelector.generate_rules(...)` | 影响 `mode="fp-growth"` 的规则产出 | PPI/TI 默认走 fp-growth，可优先用 PPI smoke test。 |
| Predicate BN 特征剪枝 | `predicate_bn.py`、`garplus_demo_runner.py` 构造 `PredicateBayesianNetwork` 的位置 | 影响 predicate feature 选择和规则数量 | 看 `tau_x`、`predicate_bn_*` 参数和 `[PredicateFilter]`。 |
| target Y 的选择 | `garplus_demo_runner.py` 的 `build_target_y_list(...)`、`predicate_focus_items_for_y_key(...)`；各 `*_demo.py` 的 `include_*` 配置 | 影响挖哪些 consequent | 看 `[YList] targets=[...]` 和 `[YTarget]`。 |
| 规则 payload 生成、过滤、去重和输出 | `rulegeneration.py`、`garplus_demo_runner.py` 的 `predicate_rule_to_zl(...)`、`print_deduped_rules(...)` | 影响 `deduped_rules_*.txt` 和 final `[Summary]` | 看 `[RuleGeneration]`、`[RuleConsequentDistribution]`、`deduped_rules`。 |

### 10.1 修改 pattern 生成逻辑

优先阅读和修改：

```text
pattern_extension.py
graph_types.py
pattern_bn.py
sampled_frequent_patterns.py
garplus_demo_runner.py
```

`pattern_extension.py` 是主要入口。`garplus_demo_runner.py` 负责把配置传给 `PatternOptions`，例如：

```python
PatternOptions(
    pattern_support_threshold=cfg.pattern_support,
    max_radius=cfg.max_radius,
    max_add_edge=cfg.max_add_edge,
    node_max_add_edge=cfg.node_max_add_edge,
    max_pattern_nodes=cfg.max_pattern_nodes,
    max_multi_support=cfg.max_multi_support,
    undirected_pattern=cfg.undirected_pattern,
    topology_only_dedup=cfg.topology_only_pattern_dedup,
    topology_dedupe_respect_direction=cfg.topology_dedupe_respect_direction,
    extension_debug=cfg.pattern_extension_debug,
)
```

修改建议：

- 改候选扩展策略时，优先在 `GraphSpawn.vspawn()` 附近定位候选生成、剪枝和支持度检查。
- 改 pattern 支持度定义时，同时检查 `FrequentPattern.single_support()` / `multi_support()` 的使用点。
- 改有向/无向逻辑时，同时检查 `undirected_pattern`、`topology_only_pattern_dedup`、`topology_dedupe_respect_direction`。
- 改完后优先看 `[VSpawnStats]`，确认候选被哪个环节剪掉。

最小验证：

```python
PATTERN_SIZES = (2,)
MAX_RADIUS = 2
MAX_ADD_EDGE = 2
NODE_MAX_ADD_EDGE = 2
PATTERN_SUPPORT = 1
PATTERN_EXTENSION_DEBUG = True
```

然后运行：

```powershell
python batch_pattern_size_ppi.py
```

重点看：

```text
[VSpawn]
[VSpawnStats]
[SampledFSMInject]
[GlobalRematch]
[Patterns]
[PatternInstances]
```

### 10.2 修改 predicate 生成逻辑

优先阅读和修改：

```text
predicate_selection.py
predicate_bn.py
predicate_enrichment.py
garplus_ml_predicates.py
garplus_demo_runner.py
```

`predicate_selection.py` 是 predicate/rule 生成的主要入口。当前 runner 根据 `cfg.mode` 选择：

```python
DecisionTreePredicateSelector(...)
FPGrowthPredicateSelector(...)
```

两个 selector 都会受到这些参数影响：

```python
min_support
min_confidence
min_value_support_count
drop_target_values
allowed_consequent_values
drop_feature_key_tokens
drop_target_entity_features
predicate_bn
```

修改建议：

- 改候选 predicate 列如何产生：看 selector 构造 instance rows / transactions 的逻辑。
- 改低频值过滤：看 `min_value_support_count` 和 value support pruning。
- 改 feature 过滤：看 `drop_feature_key_tokens`、`filter_degree_predicates` 和 `ignored_predicate_key_tokens`。
- 改 consequent 允许值：看 `allowed_consequent_values_for_y_key(...)`。
- 改 target 选择：看 `build_target_y_list(...)` 和 demo 文件里的 `include_edge_existing_target`、`include_ml_predicate_targets`。
- 改 Predicate BN 剪枝：看 `predicate_bn.py` 和 `PredicateBNConfig(min_score=cfg.tau_x, ...)`。

验证建议：

- 改 decision-tree：优先跑 DDA，因为 `dda_demo.py` 默认 `mode="decision-tree"`。
- 改 fp-growth：优先跑 PPI，因为 `ppi_demo.py` 默认 `mode="fp-growth"`。
- 如果只想验证 predicate 逻辑，不建议一开始跑 `PATTERN_SIZES=(2,4,6,8)`，先固定一个小 pattern size。

重点日志：

```text
[YList]
[YTarget]
[PredicateFocus]
[PredicateSelection/DecisionTree]
[PredicateSelection/FPGrowth]
[PredicateFilter]
[PredicateFocusMerge]
[RuleConsequentDistribution]
[Summary]
```

### 10.3 修改规则过滤、去重和输出逻辑

优先阅读和修改：

```text
garplus_demo_runner.py
rulegeneration.py
```

关键函数/阶段：

- `predicate_rule_to_zl(...)`：把 selector 规则转成 ZLRule。
- `zl_rule_filter(...)`：规则过滤。
- `send_zl_rules(...)`：payload 发送到 `RuleSender`。
- `print_deduped_rules(...)`：最终规则去重和输出。
- `rule_consequent_distribution(...)`：统计 consequent 分布。
- `discovered_rule_table_stats(...)`：最终 summary 表格统计。

如果用户要“规则更多/更少/更准”，不要只改 batch 脚本里的 `min_support`，还要判断目标是：

- predicate 候选变多/变少；
- selector 生成逻辑变化；
- consequent 值过滤变化；
- rule dedupe key 变化；
- payload 过滤变化。

### 10.4 修改后的回归验证顺序

建议每次算法逻辑改动后按这个顺序验证：

1. 只跑单数据集、单 pattern size：

```powershell
python batch_pattern_size_ppi.py
```

并临时设置：

```python
PATTERN_SIZES = (2,)
```

2. 看日志是否走完整链路：

```text
[RunStart] -> [Graph] -> [VSpawn] -> [Patterns] -> [PatternStart] -> [Timing] -> [Summary]
```

3. 对比关键数量：

```text
generated_total
unique_total
mining_total
raw_rules
deduped_rules
rule_mining_seconds
```

4. 再跑对应模式：

- 改 FP-Growth：跑 PPI 或 TI。
- 改 decision-tree：跑 DDA。
- 改 pattern extension：三个数据集都至少 smoke test 一次。
- 改 rule output/dedupe：检查 `deduped_rules_*.txt`。

5. 最后再跑完整 sweep：

```powershell
python batch_pattern_size_ppi.py
python batch_pattern_size_dda.py
python batch_pattern_size_ti.py
```

## 11. 常见参数修改任务

### 11.1 只跑一个 pattern size

修改对应 `batch_pattern_size_*.py`：

```python
PATTERN_SIZES = (2,)
```

然后运行对应脚本：

```powershell
python batch_pattern_size_ppi.py
```

### 11.2 增加 pattern size

修改：

```python
PATTERN_SIZES = (2, 4, 6, 8, 10)
```

注意：`MAX_RADIUS=8`、`MAX_ADD_EDGE=8`、`PATTERN_SUPPORT=1` 时搜索空间可能很大，新增更大 pattern size 可能非常慢。

### 11.3 修改搜索复杂度

修改批量脚本中的：

```python
MAX_RADIUS = 8
MAX_ADD_EDGE = 8
NODE_MAX_ADD_EDGE = 8
PATTERN_SUPPORT = 1
```

经验：

- 降低 `MAX_RADIUS` 通常会减少候选扩展范围。
- 降低 `MAX_ADD_EDGE` / `NODE_MAX_ADD_EDGE` 会减少候选边。
- 提高 `PATTERN_SUPPORT` 会过滤更多 pattern，通常更快。

快速 smoke test 可设：

```python
PATTERN_SIZES = (2,)
MAX_RADIUS = 2
MAX_ADD_EDGE = 2
NODE_MAX_ADD_EDGE = 2
PATTERN_SUPPORT = 5
```

### 11.4 修改 sigma

`sigma` 在脚本中对应 `min_support`。修改 `batch_sigma_confidence_*.py`：

```python
SIGMAS = (20, 50, 100)
DEFAULT_SIGMA = 50
```

`min_support` 越大，规则越少，通常规则挖掘更快。

### 11.5 修改 confidence

修改：

```python
CONFIDENCES = (0.3, 0.5, 0.7, 0.9)
DEFAULT_CONFIDENCE = 0.7
```

注意脚本会同时设置：

```python
min_confidence=confidence
tau_x=confidence
```

所以该实验不是只改变规则置信度，也同时改变 Predicate BN pruning 阈值。

### 11.6 修改数据目录

优先用环境变量，不要改代码：

```powershell
$env:GARPLUS_DATA_DIR="<YOUR_DATA_DIR>"
$env:GARPLUS_PROCESSED_DIR="<YOUR_PROCESSED_DIR>"
```

如果要永久写死路径，改 `ppi_demo.py`、`dda_demo.py`、`ti_demo.py` 中：

```python
DATA_DIR = ...
PROCESSED_DIR = ...
```

### 11.7 修改 sampled `.pt` 路径

改对应 demo：

```python
sampled_pt_path=str(PROCESSED_DIR / "ppi" / "ppi_selected.pt")
sampled_pt_path=str(PROCESSED_DIR / "dda" / "dda_selected.pt")
sampled_pt_path=str(PROCESSED_DIR / "ti" / "ti_selected.pt")
```

### 11.8 修改规则挖掘模式

PPI/TI 当前默认 `fp-growth`，DDA 默认 `decision-tree`。

修改对应 demo 的 `CONFIG`：

```python
mode="decision-tree"
```

或：

```python
mode="fp-growth"
fp_growth_max_itemset_size=4
```

### 11.9 关闭 debug 输出

批量脚本中：

```python
PATTERN_EXTENSION_DEBUG = False
```

环境变量：

```powershell
$env:GARPLUS_PATTERN_DEBUG="0"
$env:GARPLUS_DEBUG_MATCH_EXPANSION="0"
$env:GARPLUS_DEBUG_TRANSACTION_COST="0"
```

### 11.10 避免结果覆盖

批量脚本会覆盖同名日志和 CSV。要保留不同 run，改：

```python
RESULT_DIR = Path(__file__).resolve().parent / "batch_results" / "pattern_size_ppi_run2"
```

或运行后复制整个 `batch_results`。

## 12. 常见问题

### 问题：找不到 CSV 数据

现象：

```text
FileNotFoundError
[Input] interaction_csv=... 指向不存在路径
```

原因：

`GARPLUS_DATA_DIR` 未设置，默认路径下没有数据。

解决：

```powershell
$env:GARPLUS_DATA_DIR="<DATA_DIR>"
Test-Path "$env:GARPLUS_DATA_DIR\protein_protein_signed.csv"
```

### 问题：找不到 sampled `.pt`

现象：

加载 sampled graph 时失败。

原因：

GARplus 默认 `use_sampled_pt_graph=True`，需要 `processed\<dataset>\<dataset>_selected.pt`。

解决：

```powershell
$env:GARPLUS_PROCESSED_DIR="<PROCESSED_DIR>"
Test-Path "$env:GARPLUS_PROCESSED_DIR\ppi\ppi_selected.pt"
```

### 问题：运行很慢

原因：

默认批量脚本把搜索参数提高到：

```python
MAX_RADIUS = 8
MAX_ADD_EDGE = 8
NODE_MAX_ADD_EDGE = 8
PATTERN_SUPPORT = 1
```

这比 demo 文件里的默认 `max_radius=2/3`、`max_add_edge=2` 更重。

解决：

先 smoke test：

```python
PATTERN_SIZES = (2,)
MAX_RADIUS = 2
MAX_ADD_EDGE = 2
NODE_MAX_ADD_EDGE = 2
PATTERN_SUPPORT = 5
```

### 问题：`patterns_mined` 为空

原因：

批量脚本从日志中用正则提取 `[Patterns] ... mining_total=...`。如果 runner 在打印 `[Patterns]` 前报错，或日志格式被改了，就会为空。

解决：

查看对应日志：

```powershell
Get-Content <log_path> -Tail 120
```

确认是否存在 `[Patterns]`，以及是否有 traceback。

### 问题：`rule_mining_seconds` 为空

原因：

正则没有匹配到：

```text
[Timing] stage=rule_mining_total ... seconds=...
```

可能是规则挖掘前报错，或 `pattern_extension_only=True` 提前返回。

解决：

检查 demo 中是否开启了：

```powershell
$env:GARPLUS_PATTERN_DEBUG="1"
```

DDA/TI 在 debug 模式下可能走 `pattern-only`。

### 问题：sigma/confidence CSV 中 raw_rules 为空

原因：

脚本依赖 `[Summary] ... raw_rules=... deduped_rules=...`。如果 runner 没走到 final summary 或 summary 格式被改，会提取失败。

解决：

检查日志末尾是否有 `[Summary]`。

### 问题：Windows 路径和 Linux 路径混用

现象：

日志中出现某个硬编码的外部路径找不到。

原因：

`ppi_demo.py`、`dda_demo.py`、`ti_demo.py` 的 `MLPredicateConfig.precomputed_edge_csv_path` 里可能有非当前机器可用的硬编码路径。

解决：

优先确认 `offline_csv_path=str(PROCESSED_DIR / "<dataset>" / "ml_predicates.csv")` 是否存在。如果读取逻辑仍使用 Linux 路径，检查 `garplus_ml_predicates.py`，或把 `precomputed_edge_csv_path` 改成 Windows 下真实路径。

## 13. 后续 agent 工作建议

后续 agent 的主要任务预期是改规则挖掘逻辑，而不是只跑性能脚本。接手时建议先把用户需求归类到 pattern 生成、predicate 生成、规则过滤/去重、BN 剪枝、global rematch 或数据配置其中一类。

如果用户要求“修改 pattern 生成”：

1. 先看 `pattern_extension.py` 和 `garplus_demo_runner.py` 的 `GraphSpawn` 构造。
2. 开启小规模 debug，先跑 `PATTERN_SIZES=(2,)`。
3. 对比 `[VSpawnStats]` 和 `[Patterns]`，确认候选数量、剪枝数量、最终 `mining_total` 是否符合预期。

如果用户要求“修改 predicate 生成/规则生成”：

1. 先看 `predicate_selection.py`，再看 `garplus_demo_runner.py` 中 selector 的构造参数。
2. 根据模式选择验证数据集：FP-Growth 优先 PPI/TI，decision-tree 优先 DDA。
3. 对比 `[PredicateFilter]`、`[PredicateFocusMerge]`、`raw_rules`、`deduped_rules`。

如果用户要求“修改规则输出、过滤、去重”：

1. 先看 `garplus_demo_runner.py` 的 `print_deduped_rules(...)`、`predicate_rule_to_zl(...)`。
2. 再看 `rulegeneration.py` 的 `zl_rule_filter(...)`、`send_zl_rules(...)`。
3. 验证 `deduped_rules_*.txt` 和 final `[Summary]`。

如果用户要求“继续完善文档”：

1. 优先编辑本文件 `agent.md`。
2. 保持以当前 `GARplusMiner` 工程根目录为准。
3. 不要把临时目录里的 baseline 文件写进目标工程说明，除非用户明确要求跨目录比较。

如果用户要求“跑实验”：

1. 先运行最小 smoke test，建议只跑 PPI 的 `PATTERN_SIZES=(2,)`。
2. 检查 CSV `status` 和日志 traceback。
3. 再恢复完整 `PATTERN_SIZES=(2,4,6,8)`。
4. 最后跑 DDA/TI 和 sigma/confidence sweep。

如果用户要求“只改参数”：

1. pattern size / radius / add_edge / support：改对应批量脚本。
2. 数据路径 / mode / 数据集特定选项：改 `ppi_demo.py`、`dda_demo.py`、`ti_demo.py`。
3. 算法内部逻辑不要只改参数，按第 10 节定位具体模块。

如果用户要求“生成报告图表”：

1. 读取 `batch_results\pattern_size_*` 下的 `*_pattern_size_timing.csv`。
2. 读取 `batch_results\sigma_confidence_*` 下的 `*_sigma_confidence_timing.csv`。
3. 以 `wall_seconds` 做端到端时间图。
4. 以 `rule_mining_seconds` 做规则挖掘阶段时间图。
5. 同时报告 `patterns_mined`、`raw_rules`、`deduped_rules`，避免只比较时间造成误读。

## 14. 本次 agent 已做过的排查

本次已检查：

- `GARplusMiner` 工程根目录的文件列表。
- `batch_pattern_size_ppi.py` 的完整逻辑。
- `batch_sigma_confidence_ppi.py` 的完整逻辑。
- `batch_sigma_confidence_dda.py` / `batch_sigma_confidence_ti.py` 的关键参数。
- `ppi_demo.py`、`dda_demo.py`、`ti_demo.py` 的关键配置。
- `garplus_demo_runner.py` 的 `GarplusRunConfig`、`load_graph()`、`run_demo()` 日志关键点。
- `README.md`、`GAR_FLOW.md` 的概要内容。

已确认：

- 目标工程中存在完整内部模块，如 `graph_types.py`、`pattern_extension.py`、`ppi_loader.py`、`sampled_pt_loader.py` 等。
- 目标工程中没有 GAR/GFD baseline 批量脚本；当前性能入口是 GARplus。
- 当前批量脚本会自动创建 `RESULT_DIR`。
- 当前日志和 CSV 文件名固定，重复运行会覆盖同名结果。

后续 agent 如果要修改挖掘逻辑，优先从第 10 节定位模块，再用第 8/9/11 节的脚本和输出做小规模回归验证。
