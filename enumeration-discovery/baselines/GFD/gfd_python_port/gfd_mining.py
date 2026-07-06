from __future__ import annotations

"""GFD dependency mining on top of discovered graph-pattern instances.

A GFD in the paper has the form Q[x](X -> l) or Q[x](X -> false),
where X is a set of literals and l is one literal. A literal is either
constant-valued (x.A = c) or variable-valued (x.A = y.B). For a fixed graph
pattern Q, the dependency is checked on every match h of Q: if h satisfies all
literals in X, then h must satisfy l (or must not exist for false).

This module implements that literal-implication semantics over the materialized
matches of one frequent pattern. It is intentionally small and local so the GFD
baseline can reuse the GAR pattern mining/matching pipeline.
"""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Dict, Iterable, List, Sequence, Tuple

from graph_types import DataGraph, FrequentPattern, instance_literals


Row = Dict[str, object]


@dataclass(frozen=True)
class GFDLiteral:
    """One GFD literal: key=value or left_key=right_key."""

    kind: str
    left: str
    right: object

    def format(self) -> str:
        return f"{self.left}={self.right}"

    def is_satisfied(self, row: Row) -> bool:
        if self.kind == "constant":
            return row.get(self.left) == self.right
        if self.kind == "equality":
            right_key = str(self.right)
            return self.left in row and right_key in row and row[self.left] == row[right_key]
        if self.kind == "false":
            return False
        raise ValueError(f"Unsupported literal kind: {self.kind}")


@dataclass
class GFDConflict:
    """Example rows that satisfy X but violate the RHS literal."""

    row_indices: List[int]
    examples: List[Row]

    def to_dict(self, determinant_keys: Sequence[str], dependent: str) -> Dict[str, object]:
        return {
            "x": list(determinant_keys),
            "y": dependent,
            "row_indices": list(self.row_indices),
            "examples": [dict(row) for row in self.examples],
        }


@dataclass
class GFDDependency:
    """A mined graph functional dependency candidate."""

    determinant: Tuple[str, ...]
    dependent: str
    support_count: int
    group_count: int
    violation_group_count: int
    violating_row_count: int
    confidence: float
    strict: bool
    conflicts: List[GFDConflict] = field(default_factory=list)
    kind: str = "positive"
    antecedent_support_count: int = 0
    pattern_id: int = -1
    pattern_node_count: int = 0
    pattern_edge_count: int = 0

    def to_dict(self) -> Dict[str, object]:
        return {
            "kind": self.kind,
            "determinant": list(self.determinant),
            "dependent": self.dependent,
            "support_count": self.support_count,
            "antecedent_support_count": self.antecedent_support_count,
            "group_count": self.group_count,
            "violation_group_count": self.violation_group_count,
            "violating_row_count": self.violating_row_count,
            "confidence": self.confidence,
            "strict": self.strict,
            "pattern_id": self.pattern_id,
            "pattern_node_count": self.pattern_node_count,
            "pattern_edge_count": self.pattern_edge_count,
            "conflicts": [conflict.to_dict(self.determinant, self.dependent) for conflict in self.conflicts],
        }

    def format(self) -> str:
        lhs = " and ".join(self.determinant) if self.determinant else "empty"
        mode = "strict" if self.strict else "approx"
        return (
            f"[{self.kind}/{mode}] Q{self.pattern_id}({lhs}) -> {self.dependent} "
            f"support={self.support_count}/{self.antecedent_support_count} "
            f"confidence={self.confidence:.3f} violations={self.violating_row_count}"
        )


class GFDDependencyMiner:
    """Mine GFDs from matched instances of one frequent graph pattern."""

    def __init__(
        self,
        min_support_count: int = 2,
        min_confidence: float = 1.0,
        min_value_support_count: int = 1,
        max_lhs_size: int = 2,
        max_conflicts: int = 5,
        max_constant_literals_per_key: int = 5,
        max_candidate_literals: int = 200,
        discover_constant_rhs: bool = True,
        discover_equality_rhs: bool = True,
        discover_negative: bool = False,
    ) -> None:
        self.min_support_count = max(1, int(min_support_count))
        self.min_confidence = float(min_confidence)
        self.min_value_support_count = max(1, int(min_value_support_count))
        self.max_lhs_size = max(0, int(max_lhs_size))
        self.max_conflicts = max(0, int(max_conflicts))
        self.max_constant_literals_per_key = max(1, int(max_constant_literals_per_key))
        self.max_candidate_literals = max(1, int(max_candidate_literals))
        self.discover_constant_rhs = bool(discover_constant_rhs)
        self.discover_equality_rhs = bool(discover_equality_rhs)
        self.discover_negative = bool(discover_negative)

    def build_instance_rows(self, graph: DataGraph, frequent_pattern: FrequentPattern) -> List[Row]:
        """Convert matched pattern instances into attribute rows."""

        rows: List[Row] = []
        for instance in frequent_pattern.instances:
            row: Row = {}
            for record in instance_literals(graph, frequent_pattern.pattern, instance):
                literal_key = f"{record.entity}.{record.key}"
                if literal_key not in row:
                    row[literal_key] = record.value
                    continue
                existing = row[literal_key]
                if isinstance(existing, list):
                    if record.value not in existing:
                        existing.append(record.value)
                elif existing != record.value:
                    row[literal_key] = [existing, record.value]
            rows.append({key: self._normalize_value(value) for key, value in row.items()})
        return rows

    def prune_rows_by_value_support(self, rows: List[Row]) -> List[Row]:
        """Drop rare literal values before GFD checks."""

        if not rows:
            return []
        value_counts: Dict[str, Counter] = defaultdict(Counter)
        for row in rows:
            for key, value in row.items():
                value_counts[key][value] += 1

        allowed = {
            key: {value for value, count in counts.items() if count >= self.min_value_support_count}
            for key, counts in value_counts.items()
        }
        return [
            {key: value for key, value in row.items() if value in allowed.get(key, set())}
            for row in rows
        ]

    def discover_dependencies(
        self,
        graph: DataGraph,
        frequent_pattern: FrequentPattern,
        y_key: str | None = None,
        candidate_keys: Sequence[str] | None = None,
    ) -> List[GFDDependency]:
        """Discover GFDs Q[x](X -> l) for one fixed pattern Q.

        If ``y_key`` is provided, RHS literals are restricted to that attribute:
        constant RHS ``y_key=c`` and equality RHS ``y_key=other_key``. If omitted,
        all frequent literals can be RHS candidates.
        """

        rows = self.prune_rows_by_value_support(self.build_instance_rows(graph, frequent_pattern))
        rows = [row for row in rows if row]
        if not rows:
            return []

        all_keys = sorted({key for row in rows for key in row})
        if candidate_keys is not None:
            key_pool = [key for key in candidate_keys if key in all_keys]
        else:
            key_pool = all_keys
        rhs_literals = self._rhs_literals(rows, all_keys, y_key)
        antecedent_literals = self._antecedent_literals(rows, key_pool)

        dependencies: List[GFDDependency] = []
        for rhs in rhs_literals:
            lhs_pool = [literal for literal in antecedent_literals if literal.format() != rhs.format()]
            for size in range(0, min(self.max_lhs_size, len(lhs_pool)) + 1):
                for lhs in combinations(lhs_pool, size):
                    dep = self.evaluate_literal_dependency(rows, lhs, rhs, frequent_pattern)
                    if dep is None:
                        continue
                    if dep.support_count >= self.min_support_count and dep.confidence >= self.min_confidence:
                        dependencies.append(dep)
                if size == 0 and self.max_lhs_size == 0:
                    break

        if self.discover_negative:
            dependencies.extend(self.discover_negative_dependencies(rows, antecedent_literals, frequent_pattern))

        dependencies.sort(
            key=lambda item: (
                item.kind != "positive",
                -int(item.strict),
                -item.confidence,
                -item.support_count,
                len(item.determinant),
                item.dependent,
                item.determinant,
            )
        )
        return self._remove_redundant_dependencies(dependencies)

    def evaluate_literal_dependency(
        self,
        rows: List[Row],
        lhs: Sequence[GFDLiteral],
        rhs: GFDLiteral,
        frequent_pattern: FrequentPattern,
    ) -> GFDDependency | None:
        """Evaluate one positive GFD candidate X -> l under match-wise semantics."""

        antecedent_indices = [idx for idx, row in enumerate(rows) if all(literal.is_satisfied(row) for literal in lhs)]
        antecedent_support = len(antecedent_indices)
        if antecedent_support < self.min_support_count:
            return None

        satisfied = [idx for idx in antecedent_indices if rhs.is_satisfied(rows[idx])]
        violating = [idx for idx in antecedent_indices if idx not in set(satisfied)]
        support_count = len(satisfied)
        confidence = support_count / antecedent_support if antecedent_support else 0.0
        conflicts = self._conflicts(rows, violating)
        return GFDDependency(
            determinant=tuple(literal.format() for literal in lhs),
            dependent=rhs.format(),
            support_count=support_count,
            antecedent_support_count=antecedent_support,
            group_count=antecedent_support,
            violation_group_count=len(violating),
            violating_row_count=len(violating),
            confidence=confidence,
            strict=len(violating) == 0,
            conflicts=conflicts,
            kind="positive",
            pattern_id=frequent_pattern.pattern.pattern_id,
            pattern_node_count=frequent_pattern.pattern.node_count(),
            pattern_edge_count=frequent_pattern.pattern.edge_count(),
        )

    def discover_negative_dependencies(
        self,
        rows: List[Row],
        antecedent_literals: Sequence[GFDLiteral],
        frequent_pattern: FrequentPattern,
    ) -> List[GFDDependency]:
        """Generate simple negative GFD candidates X -> false.

        A strict negative GFD is satisfied when no match satisfies X. Since we only
        have literals observed in the graph, this practical baseline considers
        combinations of frequent literals whose conjunction has zero matches.
        """

        dependencies: List[GFDDependency] = []
        for size in range(1, min(self.max_lhs_size + 1, len(antecedent_literals)) + 1):
            for lhs in combinations(antecedent_literals, size):
                support = sum(1 for row in rows if all(literal.is_satisfied(row) for literal in lhs))
                if support != 0:
                    continue
                dependencies.append(
                    GFDDependency(
                        determinant=tuple(literal.format() for literal in lhs),
                        dependent="false",
                        support_count=0,
                        antecedent_support_count=0,
                        group_count=0,
                        violation_group_count=0,
                        violating_row_count=0,
                        confidence=1.0,
                        strict=True,
                        kind="negative",
                        pattern_id=frequent_pattern.pattern.pattern_id,
                        pattern_node_count=frequent_pattern.pattern.node_count(),
                        pattern_edge_count=frequent_pattern.pattern.edge_count(),
                    )
                )
                if len(dependencies) >= self.max_candidate_literals:
                    return dependencies
        return dependencies

    def _rhs_literals(self, rows: List[Row], all_keys: Sequence[str], y_key: str | None) -> List[GFDLiteral]:
        keys = [y_key] if y_key else list(all_keys)
        keys = [key for key in keys if key in all_keys]
        literals: List[GFDLiteral] = []
        if self.discover_constant_rhs:
            for key in keys:
                counts = Counter(row[key] for row in rows if key in row)
                for value, count in counts.most_common(self.max_constant_literals_per_key):
                    if count >= self.min_support_count:
                        literals.append(GFDLiteral("constant", key, value))
        if self.discover_equality_rhs:
            for key in keys:
                for other in all_keys:
                    if other == key:
                        continue
                    support = sum(1 for row in rows if key in row and other in row and row[key] == row[other])
                    if support >= self.min_support_count:
                        literals.append(GFDLiteral("equality", key, other))
        return self._dedupe_literals(literals)[: self.max_candidate_literals]

    def _antecedent_literals(self, rows: List[Row], key_pool: Sequence[str]) -> List[GFDLiteral]:
        literals: List[GFDLiteral] = []
        for key in key_pool:
            counts = Counter(row[key] for row in rows if key in row)
            for value, count in counts.most_common(self.max_constant_literals_per_key):
                if count >= self.min_value_support_count:
                    literals.append(GFDLiteral("constant", key, value))
        for left, right in combinations(key_pool, 2):
            support = sum(1 for row in rows if left in row and right in row and row[left] == row[right])
            if support >= self.min_value_support_count:
                literals.append(GFDLiteral("equality", left, right))
        literals = self._dedupe_literals(literals)
        literals.sort(key=lambda literal: (-self._literal_support(rows, literal), literal.format()))
        return literals[: self.max_candidate_literals]

    def _literal_support(self, rows: List[Row], literal: GFDLiteral) -> int:
        return sum(1 for row in rows if literal.is_satisfied(row))

    def _conflicts(self, rows: List[Row], violating: Sequence[int]) -> List[GFDConflict]:
        if not violating or self.max_conflicts <= 0:
            return []
        chosen = list(violating[: self.max_conflicts])
        return [GFDConflict(row_indices=chosen, examples=[rows[idx] for idx in chosen])]

    def _remove_redundant_dependencies(self, dependencies: Iterable[GFDDependency]) -> List[GFDDependency]:
        """Drop larger LHS dependencies already implied by an equal-quality subset."""

        kept: List[GFDDependency] = []
        best_by_rhs: Dict[Tuple[str, int], List[GFDDependency]] = defaultdict(list)
        for dep in dependencies:
            lhs = set(dep.determinant)
            bucket_key = (dep.dependent, dep.pattern_id)
            redundant = False
            for existing in best_by_rhs[bucket_key]:
                if (
                    existing.kind == dep.kind
                    and set(existing.determinant).issubset(lhs)
                    and existing.confidence >= dep.confidence
                    and existing.strict == dep.strict
                ):
                    redundant = True
                    break
            if redundant:
                continue
            kept.append(dep)
            best_by_rhs[bucket_key].append(dep)
        return kept

    @staticmethod
    def _dedupe_literals(literals: Sequence[GFDLiteral]) -> List[GFDLiteral]:
        seen = set()
        result: List[GFDLiteral] = []
        for literal in literals:
            key = (literal.kind, literal.left, literal.right)
            if key in seen:
                continue
            seen.add(key)
            result.append(literal)
        return result

    @staticmethod
    def _normalize_value(value: object) -> object:
        if isinstance(value, list):
            return "|".join(str(item) for item in sorted(value, key=str))
        return value
