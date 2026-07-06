# GFD Python Port

This directory is derived from `baselines/GAR/gar_python_port`, but the rule-mining stage is changed from GAR-style association rules to GFD-style graph functional dependency discovery.

## 2026-06 GARplusMiner Architecture Sync

This baseline now reuses the GARplusMiner VSpawn architecture while keeping GFD
semantics:

- Synchronized from `GARplusMiner`:
  - `graph_types.py`: topology-aware canonical codes, undirected-pattern
    options, global-vspawn flags, extension debug flags, edge binding helpers,
    and edge-existence literal extraction.
  - `pattern_extension.py`: round-frontier VSpawn, topology-only dedupe,
    optional global parent rematching, incremental instance extension after
    adding one edge, richer `SpawnStats`, and the Go-aligned radius edge-count
    constraint.
  - `ppi_demo.py`: the same VSpawn knobs are exposed and passed into
    `PatternOptions`.
- Kept different from `GARplusMiner`:
  - GFD mines dependencies of the form `Q[x](X -> l)` or optional
    `Q[x](X -> false)`, not GAR/GAR+ association rules.
  - The default demo keeps `DISCOVER_NEGATIVE = False`; it does not use
    GARplusMiner's synthetic negative-edge expansion.
  - The GARplusMiner pattern BN and predicate BN optimizations are not wired in
    by this baseline demo.
  - Predicate generation is functional-dependency oriented: constant RHS
    (`v.A=c`) and equality RHS (`v.A=w.B`) are evaluated per matched pattern
    instance.

## What Is Reused

- `graph_types.py`: graph, pattern, instance, and literal extraction structures
- `vf3_like.py` / `vf3_linux.py`: pattern matching helpers
- `pattern_extension.py`: VSpawn-style frequent graph pattern extension
- `ppi_loader.py`: PPI CSV loading and seed-pattern construction

## What Is Different

- `gfd_mining.py` implements `GFDDependencyMiner` with the GFD paper's literal implication semantics.
- `ppi_demo.py` runs multi-round VSpawn and mines GFDs for every generated frequent graph pattern.

For a fixed graph pattern `Q`, the miner builds one row per matched instance. It then mines GFDs in normal form:

- positive constant GFD: `Q[x](X -> v.A=c)`
- positive equality GFD: `Q[x](X -> v.A=w.B)`
- optional negative GFD candidate: `Q[x](X -> false)`

A row satisfies `X -> l` when either it does not satisfy all literals in `X`, or it also satisfies RHS literal `l`. With `MIN_CONFIDENCE=1.0`, only strict GFDs with no violating matches are kept. Lower values allow approximate candidates and print example conflicts.

## Run

From this directory:

```bash
python ppi_demo.py
```

Important knobs in `ppi_demo.py`:

- `Y_KEY`: RHS attribute. Use a concrete key such as `v0.high_degree`, or set it to `None` to mine RHS literals over all attributes.
- `DISCOVER_CONSTANT_RHS`: mine `v.A=c` RHS literals.
- `DISCOVER_EQUALITY_RHS`: mine `v.A=w.B` RHS literals.
- `DISCOVER_NEGATIVE`: also generate simple `X -> false` candidates.
- `MIN_CONFIDENCE`: `1.0` means strict GFD; lower values allow approximate GFDs.
- `MIN_SUPPORT_COUNT`: minimum number of matches satisfying both `X` and the RHS literal.
- `MAX_LHS_SIZE`: maximum number of literals in `X`.
- `MIN_VALUE_SUPPORT_COUNT`: drops rare literal values before candidate generation.
- `MAX_CANDIDATE_LITERALS`: caps horizontal literal expansion cost.

## Difference From GAR

GAR mines association-style rules and keeps candidates by support/confidence over itemsets. GFD validates graph dependencies on each match of a graph pattern: once a match satisfies the LHS literal set `X`, the RHS literal must hold in that same match. This lets GFD express constant bindings, equality constraints such as `v0.name=v1.name`, and negative constraints with `false`.
