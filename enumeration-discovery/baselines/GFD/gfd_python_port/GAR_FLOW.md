# GAR Python Port Flow

## GARplusMiner Sync Notes

The structural mining layer now follows the current `GARplusMiner` VSpawn
implementation: frontier-based rounds, topology-only deduplication, optional
global parent rematching, incremental add-edge instance extension, debug
statistics, and the same `PatternOptions` knobs used by the GARplus demo.

This port intentionally keeps GFD semantics above that shared structure. The
rule layer mines functional dependencies over matched pattern instances
(`Q[x](X -> l)` and optional `Q[x](X -> false)`) instead of GAR/GAR+
association rules. Synthetic negative-edge expansion and the GARplus pattern/
predicate BN optimizations are not enabled by the demo.

## Overall pipeline

The current Python GAR port follows this order:

1. `ppi_loader.py`
   - Load `protein_protein.csv` into a large `DataGraph`
   - Optionally merge `protein.csv` attributes into vertices already present in interactions
   - Derive helper attributes such as `degree`, `degree_bucket`, `high_degree`

2. `graph_types.py`
   - Define core graph / pattern / instance structures
   - Provide `instance_literals(...)` to expand a matched pattern instance into attribute literals

3. `pattern_extension.py`
   - Start from a 1-node seed pattern
   - Run VSpawn-style pattern growth
   - For every grown pattern, re-match it on the big graph and keep frequent ones

4. `vf3_like.py`
   - Perform the actual subgraph matching with a lightweight backtracking matcher
   - Return `GraphInstance` objects for every successful match

5. `predicate_selection.py`
   - Turn matched instances into a table (`DecisionTree`) or transactions (`FP-Growth`)
   - Prune low-support values / columns
   - Generate `X -> Y` predicate rules

6. `rulegeneration.py`
   - Convert mined rules into a GAR-like payload
   - Organize `x_info`, `y_info`, `segmentation`, `zl_col`, support stats, and instances

7. `ppi_demo.py`
   - Wire all modules together for a runnable end-to-end demo

## File roles

- `graph_types.py`
  - Core data model layer
- `ppi_loader.py`
  - Data ingestion layer
- `vf3_like.py`
  - Pattern matching layer
- `pattern_extension.py`
  - Structural pattern mining layer
- `predicate_selection.py`
  - Predicate / rule mining layer
- `rulegeneration.py`
  - Output serialization layer
- `ppi_demo.py`
  - End-to-end driver

## Key concepts

- `DataGraph`
  - The large input graph
- `GraphPattern`
  - A candidate frequent subgraph pattern
- `GraphInstance`
  - One successful embedding of a pattern in the data graph
- `FrequentPattern`
  - A pattern plus all currently matched instances
- `Literal`
  - A vertex or edge attribute expanded from a matched instance
- `Y`
  - The target predicate column to explain or predict
- `X`
  - The antecedent predicates used to imply `Y`
