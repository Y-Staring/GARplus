from __future__ import annotations

"""pgmpy-based Pattern Bayesian Network for GARplusMiner.

Pattern BN is trained from data-graph edges. Each directed training case records:

    SRC_LABEL, DIRECTION, EDGE_LABEL, DST_LABEL

During VSpawn, each candidate expansion is scored by querying the learned CPDs,
mainly `P(EDGE_LABEL | SRC_LABEL, DIRECTION)` and
`P(DST_LABEL | SRC_LABEL, DIRECTION, EDGE_LABEL)`. The score is then used to
rank or prune structural expansions before expensive subgraph matching.
"""

import os
import pickle
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Tuple

_BN_ROOT = Path(__file__).resolve().parents[1]
if str(_BN_ROOT) not in sys.path:
    sys.path.insert(0, str(_BN_ROOT))

from BNlearning.pattern_bn_features import (  # noqa: E402
    BNEdgeLabelFn,
    BNNodeLabelFn,
    aggregate_spawn_node_label,
    build_default_bn_label_fns,
    default_bn_edge_label,
    default_bn_node_label,
    default_bn_node_label_for_pattern_node,
)
from graph_types import DataGraph, FrequentPattern, GraphPattern, SpawnEdge


CandidateScore = Tuple[float, SpawnEdge]


@dataclass
class PatternBNConfig:
    """Controls pgmpy Pattern BN training and pruning."""

    enabled: bool = True
    top_k_per_spawn_node: Optional[int] = None
    min_score: float = 0.0
    relative_tau: Optional[float] = None
    min_keep_per_spawn_node: int = 1
    estimator: str = "bayesian"  # bayesian | maximum_likelihood
    equivalent_sample_size: float = 5.0
    cache_path: Optional[str] = None
    retrain: bool = False
    frequent_edge_priors: Dict[Tuple[str, str, str], float] = field(default_factory=dict)
    frequent_prior_weight: float = 0.25
    bn_edge_label_fn: Optional[BNEdgeLabelFn] = None
    bn_node_label_fn: Optional[BNNodeLabelFn] = None
    use_marginal_edge_score: bool = True


class PatternBayesianNetwork:
    """Pattern BN backed by pgmpy CPDs."""

    SRC_LABEL = "src_label"
    DIRECTION = "direction"
    EDGE_LABEL = "edge_label"
    DST_LABEL = "dst_label"

    def __init__(self, config: Optional[PatternBNConfig] = None) -> None:
        self.config = config or PatternBNConfig()
        self.model = None
        self.data = None
        self.state_names: Dict[str, List[object]] = {}
        self.total_rank_calls = 0
        self.total_candidates_seen = 0
        self.total_candidates_kept = 0
        self.total_threshold_pruned = 0
        self.total_topk_pruned = 0
        self.total_min_keep_rescued = 0
        self.last_rank_snapshot: List[Tuple[float, str]] = []

    @classmethod
    def fit_graph(cls, graph: DataGraph, config: Optional[PatternBNConfig] = None) -> "PatternBayesianNetwork":
        bn = cls(config=config)
        bn.fit(graph)
        return bn

    def _edge_label_fn(self) -> BNEdgeLabelFn:
        return self.config.bn_edge_label_fn or default_bn_edge_label

    def _node_label_fn(self) -> BNNodeLabelFn:
        return self.config.bn_node_label_fn or default_bn_node_label

    def fit(self, graph: DataGraph) -> None:
        """Train the Pattern BN with pgmpy from directed graph-edge samples."""

        if self.config.cache_path and os.path.exists(self.config.cache_path) and not self.config.retrain:
            with open(self.config.cache_path, "rb") as handle:
                cached = pickle.load(handle)
            self.__dict__.update(cached.__dict__)
            self.config = cached.config
            return
        if self.config.bn_edge_label_fn is None or self.config.bn_node_label_fn is None:
            edge_fn, node_fn = build_default_bn_label_fns(graph)
            if self.config.bn_edge_label_fn is None:
                self.config.bn_edge_label_fn = edge_fn
            if self.config.bn_node_label_fn is None:
                self.config.bn_node_label_fn = node_fn

        edge_fn = self._edge_label_fn()
        node_fn = self._node_label_fn()
        pd, model_cls, _ = _load_pgmpy(self.config.estimator)
        rows = []
        for edge in graph.all_edges():
            src_label = node_fn(edge.src, graph)
            dst_label = node_fn(edge.dst, graph)
            edge_label = edge_fn(edge, graph)
            rows.append(
                {
                    self.SRC_LABEL: src_label,
                    self.DIRECTION: "out",
                    self.EDGE_LABEL: edge_label,
                    self.DST_LABEL: dst_label,
                }
            )
            rows.append(
                {
                    self.SRC_LABEL: dst_label,
                    self.DIRECTION: "in",
                    self.EDGE_LABEL: edge_label,
                    self.DST_LABEL: src_label,
                }
            )
        if not rows:
            raise ValueError("Pattern BN cannot be trained because the graph has no edges")

        self.data = pd.DataFrame(rows).astype(str)
        self.state_names = {column: sorted(self.data[column].unique().tolist()) for column in self.data.columns}
        self.model = model_cls(
            [
                (self.SRC_LABEL, self.EDGE_LABEL),
                (self.DIRECTION, self.EDGE_LABEL),
                (self.SRC_LABEL, self.DST_LABEL),
                (self.DIRECTION, self.DST_LABEL),
                (self.EDGE_LABEL, self.DST_LABEL),
            ]
        )
        _fit_pgmpy_model(
            self.model,
            self.data,
            self.config.estimator,
            self.config.equivalent_sample_size,
        )

        self._save_cache_if_needed()

    def _save_cache_if_needed(self) -> None:
        if not self.config.cache_path:
            return
        cache_dir = os.path.dirname(self.config.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(self.config.cache_path, "wb") as handle:
            pickle.dump(self, handle)

    def score_spawn_edge(
        self,
        pattern: GraphPattern,
        spawn_node: int,
        spawn_edge: SpawnEdge,
        frequent_pattern: Optional[FrequentPattern] = None,
        graph: Optional[DataGraph] = None,
    ) -> float:
        """Score one candidate expansion from learned pgmpy CPDs."""

        return self.score_spawn_edge_components(
            pattern,
            spawn_node,
            spawn_edge,
            frequent_pattern=frequent_pattern,
            graph=graph,
        )["final_score"]

    def score_spawn_edge_components(
        self,
        pattern: GraphPattern,
        spawn_node: int,
        spawn_edge: SpawnEdge,
        frequent_pattern: Optional[FrequentPattern] = None,
        graph: Optional[DataGraph] = None,
    ) -> Dict[str, float]:
        """Return CPD and sampled-frequency contributions to one spawn score."""

        if not self.config.enabled:
            return {
                "edge_prob": 1.0,
                "dst_prob": 1.0,
                "bn_score": 1.0,
                "frequent_prior": 0.0,
                "final_score": 1.0,
            }
        if self.model is None:
            return {
                "edge_prob": 0.0,
                "dst_prob": 0.0,
                "bn_score": 0.0,
                "frequent_prior": 0.0,
                "final_score": 0.0,
            }
        node_fn = self._node_label_fn()
        src_label = self._resolve_src_label(pattern, spawn_node, frequent_pattern, graph)
        direction = str(spawn_edge.direction)
        if spawn_edge.external or spawn_edge.to_node == -1:
            dst_label = f"plabel:{spawn_edge.target_label}"
        elif graph is not None and frequent_pattern is not None:
            dst_label = aggregate_spawn_node_label(pattern, spawn_edge.to_node, graph, frequent_pattern, node_fn)
        else:
            dst_label = default_bn_node_label_for_pattern_node(
                pattern,
                spawn_edge.to_node,
                graph or DataGraph(vertices={}),
                node_fn,
            )

        structural_edge_label = str(spawn_edge.edge_label)
        if self.config.use_marginal_edge_score:
            edge_prob, dst_prob, bn_score = self._marginal_spawn_score(src_label, direction, dst_label)
        else:
            bn_edge_label = f"struct:{structural_edge_label}"
            edge_prob = _cpd_probability(
                self.model,
                self.EDGE_LABEL,
                bn_edge_label,
                {self.SRC_LABEL: src_label, self.DIRECTION: direction},
            )
            dst_prob = _cpd_probability(
                self.model,
                self.DST_LABEL,
                dst_label,
                {self.SRC_LABEL: src_label, self.DIRECTION: direction, self.EDGE_LABEL: bn_edge_label},
            )
            bn_score = edge_prob * dst_prob

        frequent_prior = self._frequent_edge_prior(src_label, dst_label, structural_edge_label)
        prior_weight = min(1.0, max(0.0, float(self.config.frequent_prior_weight)))
        if self.config.frequent_edge_priors:
            final_score = (1.0 - prior_weight) * bn_score + prior_weight * frequent_prior
        else:
            final_score = bn_score
        return {
            "edge_prob": edge_prob,
            "dst_prob": dst_prob,
            "bn_score": bn_score,
            "frequent_prior": frequent_prior,
            "final_score": final_score,
        }

    def _dst_probability(self, src_label: str, direction: str, edge_state: str, dst_label: str) -> float:
        dst_states = [str(state) for state in self.state_names.get(self.DST_LABEL, [])]
        if not dst_states:
            return 0.0
        evidence = {
            self.SRC_LABEL: src_label,
            self.DIRECTION: direction,
            self.EDGE_LABEL: edge_state,
        }
        if dst_label in dst_states:
            return _cpd_probability(self.model, self.DST_LABEL, dst_label, evidence)
        best = 0.0
        for state in dst_states:
            best = max(best, _cpd_probability(self.model, self.DST_LABEL, state, evidence))
        return best

    def _edge_probability(self, src_label: str, direction: str, edge_state: str) -> float:
        return _cpd_probability(
            self.model,
            self.EDGE_LABEL,
            edge_state,
            {self.SRC_LABEL: src_label, self.DIRECTION: direction},
        )

    def _resolve_src_label(
        self,
        pattern: GraphPattern,
        spawn_node: int,
        frequent_pattern: Optional[FrequentPattern],
        graph: Optional[DataGraph],
    ) -> str:
        node_fn = self._node_label_fn()
        if graph is not None and frequent_pattern is not None and frequent_pattern.instances:
            return aggregate_spawn_node_label(pattern, spawn_node, graph, frequent_pattern, node_fn)
        states = [str(state) for state in self.state_names.get(self.SRC_LABEL, [])]
        if states:
            return Counter(self.data[self.SRC_LABEL].tolist()).most_common(1)[0][0] if self.data is not None else states[0]
        return default_bn_node_label_for_pattern_node(pattern, spawn_node, graph or DataGraph(vertices={}), node_fn)

    def _marginal_spawn_score(self, src_label: str, direction: str, dst_label: str) -> Tuple[float, float, float]:
        edge_states = [str(state) for state in self.state_names.get(self.EDGE_LABEL, [])]
        if not edge_states:
            return 0.0, 0.0, 0.0
        best_edge = 0.0
        best_dst = 0.0
        best_joint = 0.0
        for edge_state in edge_states:
            p_edge = self._edge_probability(src_label, direction, edge_state)
            p_dst = self._dst_probability(src_label, direction, edge_state, dst_label)
            joint = p_edge * p_dst
            best_edge = max(best_edge, p_edge)
            best_dst = max(best_dst, p_dst)
            best_joint = max(best_joint, joint)
        return best_edge, best_dst, best_joint

    def _frequent_edge_prior(self, src_label: str, dst_label: str, edge_label: str) -> float:
        left, right = sorted([str(src_label), str(dst_label)])
        return float(self.config.frequent_edge_priors.get((left, right, str(edge_label)), 0.0))

    def _effective_min_score(self, scored: List[Tuple[float, SpawnEdge]]) -> float:
        if not scored:
            return self.config.min_score
        max_score = max(item[0] for item in scored)
        relative = self.config.relative_tau
        if relative is not None and relative > 0.0:
            return max(self.config.min_score, relative * max_score)
        return self.config.min_score

    def rank_spawn_edges(
        self,
        pattern: GraphPattern,
        spawn_node: int,
        candidates: Iterable[SpawnEdge],
        frequent_pattern: Optional[FrequentPattern] = None,
        graph: Optional[DataGraph] = None,
    ) -> List[CandidateScore]:
        """Rank and optionally prune VSpawn actions with pgmpy CPDs."""

        candidate_list = list(candidates)
        scored = [
            (
                self.score_spawn_edge(
                    pattern,
                    spawn_node,
                    candidate,
                    frequent_pattern=frequent_pattern,
                    graph=graph,
                ),
                candidate,
            )
            for candidate in candidate_list
        ]
        scored.sort(key=lambda item: item[0], reverse=True)
        cutoff = self._effective_min_score(scored)
        thresholded = [item for item in scored if item[0] >= cutoff]
        threshold_pruned = len(scored) - len(thresholded)
        min_keep = max(0, self.config.min_keep_per_spawn_node)
        min_keep_target = min(min_keep, len(scored))
        min_keep_rescued = 0
        if min_keep and len(thresholded) < min_keep_target:
            min_keep_rescued = min_keep_target - len(thresholded)
            thresholded = scored[:min_keep_target]
        before_topk = len(thresholded)
        scored = thresholded
        if self.config.top_k_per_spawn_node is not None:
            scored = scored[: max(0, self.config.top_k_per_spawn_node)]
        topk_pruned = before_topk - len(scored)
        self.total_rank_calls += 1
        self.total_candidates_seen += len(candidate_list)
        self.total_candidates_kept += len(scored)
        self.total_threshold_pruned += threshold_pruned
        self.total_topk_pruned += topk_pruned
        self.total_min_keep_rescued += min_keep_rescued
        self.last_rank_snapshot = [
            (score, f"{edge.from_node}->{edge.to_node} {edge.direction}:{edge.edge_label}->{edge.target_label}")
            for score, edge in scored[:5]
        ]
        return scored

    def pruning_summary(self) -> Dict[str, object]:
        """Return observable pruning statistics for demo/debug printing."""

        return {
            "backend": "pgmpy",
            "rank_calls": self.total_rank_calls,
            "candidates_seen": self.total_candidates_seen,
            "candidates_kept": self.total_candidates_kept,
            "candidates_pruned": self.total_candidates_seen - self.total_candidates_kept,
            "tau_p": self.config.min_score,
            "relative_tau": self.config.relative_tau,
            "threshold_pruned": self.total_threshold_pruned,
            "topk_pruned": self.total_topk_pruned,
            "min_keep_rescued": self.total_min_keep_rescued,
            "frequent_edge_prior_count": len(self.config.frequent_edge_priors),
            "frequent_prior_weight": self.config.frequent_prior_weight,
            "bn_state_counts": {key: len(values) for key, values in self.state_names.items()},
            "top_snapshot": self.last_rank_snapshot,
        }


def _load_pgmpy(estimator: str):
    try:
        import pandas as pd
        try:
            from pgmpy.models import DiscreteBayesianNetwork as ModelCls
        except ImportError:
            from pgmpy.models import BayesianNetwork as ModelCls
        if estimator == "maximum_likelihood":
            from pgmpy.estimators import MaximumLikelihoodEstimator as EstimatorCls
        else:
            from pgmpy.estimators import BayesianEstimator as EstimatorCls
    except ImportError as exc:
        raise ImportError(
            "GARplusMiner Pattern BN now requires pgmpy and pandas. "
            "Install them in this environment, e.g. `pip install pgmpy pandas`."
        ) from exc
    return pd, ModelCls, EstimatorCls


def _fit_pgmpy_model(model, data, estimator: str, equivalent_sample_size: float) -> None:
    """Use pgmpy 1.1 discrete estimators when available, otherwise fall back to pgmpy<=1.0."""

    try:
        from pgmpy.parameter_estimator import DiscreteBayesianEstimator, DiscreteMLE
    except ModuleNotFoundError:
        from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator

        if estimator == "maximum_likelihood":
            model.fit(data, estimator=MaximumLikelihoodEstimator)
        else:
            model.fit(
                data,
                estimator=BayesianEstimator,
                prior_type="BDeu",
                equivalent_sample_size=equivalent_sample_size,
            )
        return

    if estimator == "maximum_likelihood":
        model.fit(data, estimator=DiscreteMLE())
    else:
        model.fit(
            data,
            estimator=DiscreteBayesianEstimator(
                prior_type="BDeu",
                equivalent_sample_size=equivalent_sample_size,
            ),
        )


def _cpd_probability(model, variable: str, state: str, evidence: Dict[str, str]) -> float:
    """Read a local CPD probability with graceful zero for unseen states."""

    cpd = model.get_cpds(variable)
    if cpd is None:
        return 0.0
    try:
        variable_states = list(cpd.state_names.get(variable, []))
        if state not in variable_states:
            return 0.0
        state_index = variable_states.index(state)
        values = cpd.values
        for evidence_var in cpd.variables[1:]:
            evidence_states = list(cpd.state_names.get(evidence_var, []))
            evidence_value = evidence.get(evidence_var)
            if evidence_value not in evidence_states:
                return 0.0
            values = values.take(evidence_states.index(evidence_value), axis=1)
        return float(values[state_index])
    except Exception:
        return 0.0









