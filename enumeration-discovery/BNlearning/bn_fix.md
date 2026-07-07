# GAR+ Pattern-BN / Predicate-BN 修复说明

> 修复在线挖掘中 BN 变量退化导致「几乎剪不掉」的问题。  
> **子图匹配语义不变**，仅 BN 训练与打分使用独立特征。

## 1. 问题根因

PPI 采样图中：

- 所有顶点 `label = "Protein"`
- 所有边被 `force_edge_label = "candidate_interaction"` 折叠（匹配用）

旧 Pattern-BN 用 `src_label → edge_label → dst_label` 建 CPD 时，条件概率恒为 **1.0**，`bn_score ≡ 1.0`，`tau_p=0.5` 无法剪掉任何 VSpawn 候选。

`BNlearning/build_pattern_edge_node_bn.py` 中已有 role / clustering / core 等结构特征，但 `GARplusMiner` 在线挖掘未接入。

## 2. 修复方案

### 2.1 特征解耦（`BNlearning/pattern_bn_features.py`）

| 用途 | 标签来源 |
|------|----------|
| 子图匹配 | `vertex.label`、`force_edge_label`（不变） |
| BN 训练/打分 | `bn_edge_label_fn`、`bn_node_label_fn` 回调映射 |

**边 BN 标签**（优先级）：

1. `experimental_system`（CSV 列 `Experimental System`）
2. `score_bin`（边 score 三分箱）
3. 回退 `struct:{edge.label}`

**禁止**使用 `interaction_label` 作为 BN 边标签（RHS 目标，防泄漏）。

**节点 BN 标签**：

- `role`：度三分箱 `leaf` / `mid` / `hub`
- `clustering_bin`、`core_bin`
- VSpawn 时对 pattern 节点按实例绑定做**多数投票**（`aggregate_spawn_node_label`）

图加载后调用 `augment_graph_structural_features(graph)` 注入上述属性。

### 2.2 剪枝策略（`GARplusMiner/pattern_bn.py`）

`rank_spawn_edges` 使用：

```
cutoff = max(tau_p, relative_tau × max_score)
保留 top_k_per_spawn_node，且至少 min_keep_per_spawn_node 个
```

外部扩展边（尚无具体目标节点）使用**边际化打分**（`use_marginal_edge_score=True`）。

### 2.3 Predicate-BN（`GARplusMiner/predicate_bn.py`）

- 修复 pgmpy 1.1.2：`model.fit(..., estimator=DiscreteMLE())` / `DiscreteBayesianEstimator(...)`
- 配合结构角色 + `predicate_enrichment` 数值分箱 + ML 谓词候选

### 2.4 Strict 预设（`BNlearning/bn_config.py`）

| 参数 | Strict 值 | 含义 |
|------|-----------|------|
| `tau_p` | `0.0` | Pattern-BN 绝对下限 |
| `pattern_bn_relative_tau` | `0.5` | 自适应阈值 α×max_score |
| `pattern_bn_top_k_per_spawn_node` | `4` | 每 spawn 节点最多保留候选 |
| `pattern_bn_min_keep_per_spawn_node` | `1` | 每 spawn 节点至少保留 |
| `augment_structural_features` | `True` | 注入结构特征 |
| `tau_x` | `0.05` | Predicate-BN 特征剪枝阈值 |
| `predicate_bn_min_keep_features` | `6` | 谓词特征 min_keep |

`ppi_demo.py` 已通过 `**STRICT_PATTERN_BN, **STRICT_PREDICATE_BN` 启用 strict 配置。

## 3. 代码结构

```
enumeration-discovery/
├── BNlearning/
│   ├── pattern_bn_features.py   # BN 特征工程（核心逻辑）
│   ├── bn_config.py             # strict 预设
│   └── bn_fix.md                # 本文档
└── GARplusMiner/
    ├── pattern_bn.py            # Pattern-BN 训练与 rank_spawn_edges
    ├── predicate_bn.py          # Predicate-BN
    ├── pattern_extension.py     # VSpawn 调用 BN 时传入 graph / frequent_pattern
    ├── garplus_demo_runner.py   # 加载图后 augment、构造 PatternBNConfig
    └── ppi_demo.py              # PPI 入口，引用 strict 预设
```

## 4. 环境依赖

```bash
pip install pgmpy>=1.1.2 pandas networkx scikit-learn>=1.5
```

GARplusMiner 其余依赖（torch、torch-geometric 等）按原项目要求安装。

## 5. 数据准备

1. 将 PPI 数据放到可访问目录（含 `protein.csv`、`protein_protein_signed.csv`、`processed/ppi/ppi_selected.pt` 等）。
2. 通过环境变量指定路径（推荐）：

```bash
export GARPLUS_DATA_DIR=/path/to/your/data
export GARPLUS_PROCESSED_DIR=/path/to/processed
```

Windows PowerShell：

```powershell
$env:GARPLUS_DATA_DIR = "D:\path\to\data"
$env:GARPLUS_PROCESSED_DIR = "D:\path\to\processed"
```

若未设置 `GARPLUS_DATA_DIR`，默认使用 `enumeration-discovery/去病图数据/`（与仓库原布局一致）。

## 6. 运行

```bash
cd enumeration-discovery/GARplusMiner
python ppi_demo.py
```

日志中关注：

```
[Graph] augmented structural features: role, clustering_bin, core_bin, score_bin
[Pruning] tau_p=0.0 relative_tau=0.5 tau_x=0.05 ...
[PatternBN] ... seen=... kept=... pruned=... relative_tau=0.5 topk_pruned=...
[PredicateBN] ... seen=... pruned=...
```

**剪枝生效标志**：`[PatternBN]` 中 `pruned > 0` 或 `topk_pruned > 0`；`bn_states` 中各变量状态数 > 1。

首次运行会在 `processed/ppi/pattern_bn_strict.pkl` 缓存 Pattern-BN。

## 7. 在其他 Demo 中启用 strict BN

```python
from dataclasses import replace
from BNlearning.bn_config import STRICT_PATTERN_BN, STRICT_PREDICATE_BN

cfg = replace(CONFIG, **STRICT_PATTERN_BN, **STRICT_PREDICATE_BN)
```

或：

```python
from BNlearning.bn_config import apply_strict_bn_config

cfg = apply_strict_bn_config(CONFIG)
```

## 8. 调参建议

| 目标 | 调整 |
|------|------|
| 更强 Pattern 剪枝 | 降低 `pattern_bn_relative_tau`（如 0.3→0.5），减小 `top_k`（如 8→4） |
| 保护召回 | 增大 `min_keep_per_spawn_node`，或略降 `relative_tau` |
| 更强 Predicate 剪枝 | 提高 `tau_x`（如 0.05→0.1） |
| 搜索空间大时 BN 才明显 | 增大 `max_radius` / `max_add_edge`，候选多时 top-k 才有区分 |

## 9. PPI 验证参考（strict）

`pattern_extension_only` 对比（注入 12 个采样频繁模式）：

| 配置 | bn_pruned | prune_rate | unique_total |
|------|-----------|------------|--------------|
| legacy（退化 BN） | 0 | 0% | 7 |
| strict | 21 | 28.8% | 7 |

完整挖掘：PredicateBN 约 2817 seen / 2775 pruned（特征级剪枝）。

## 10. 与旧版差异摘要

- **旧**：BN 变量 = 匹配标签 → CPD 退化 → 剪枝无效  
- **新**：BN 变量 = 结构/语义特征 → 自适应阈值 + top-k → 可有效剪枝  
- **不变**：`force_edge_label`、顶点 `label` 仍只用于子图匹配
