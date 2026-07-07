from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

from graph_types import DataGraph, FrequentPattern, GraphInstance, GraphPattern, Vertex
from ppi_loader import _assign_degree_features, _edge_attrs_from_row, _merge_attr, _merge_vertex, _normalize_edge_label, _normalize_key, _normalize_scalar


def _raise_csv_field_limit() -> None:
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit //= 10


DISEASE_NODE_OFFSET = 1_000_000_000


@dataclass(frozen=True)
class RelationGraphConfig:
    relation_name: str
    source_label: str
    target_label: str
    source_index_column: str
    target_index_column: str
    default_edge_label: str
    edge_csv_path: str
    source_node_csv_path: Optional[str] = None
    target_node_csv_path: Optional[str] = None
    source_node_index_column: str = "index"
    target_node_index_column: str = "index"
    target_node_offset: int = DISEASE_NODE_OFFSET
    load_node_attributes: bool = False
    excluded_node_attr_columns: Tuple[str, ...] = ()
    source_edge_attr_columns: Tuple[str, ...] = ()
    target_edge_attr_columns: Tuple[str, ...] = ()
    excluded_edge_attr_columns: Tuple[str, ...] = ()


def _load_node_attrs(
    path: Optional[str],
    label: str,
    index_column: str = "index",
    offset: int = 0,
    excluded_columns: Tuple[str, ...] = (),
) -> Dict[int, Vertex]:
    _raise_csv_field_limit()
    result: Dict[int, Vertex] = {}
    excluded = {_normalize_key(column) for column in excluded_columns}
    if not path or not Path(path).exists():
        return result
    with Path(path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_id = _normalize_scalar(row.get(index_column))
            if raw_id is None:
                continue
            node_id = int(raw_id) + offset
            attrs = {}
            for column, value in row.items():
                key = _normalize_key(column)
                if column == index_column or key in excluded:
                    continue
                normalized = _normalize_scalar(value)
                if normalized is not None:
                    attrs[key] = normalized
            result[node_id] = Vertex(id=node_id, label=label, attrs=attrs)
    return result


def _load_relation_node_attrs(cfg: RelationGraphConfig) -> Dict[int, Vertex]:
    if not cfg.load_node_attributes:
        return {}
    source_attrs = _load_node_attrs(
        cfg.source_node_csv_path,
        cfg.source_label,
        index_column=cfg.source_node_index_column,
        offset=0,
        excluded_columns=cfg.excluded_node_attr_columns,
    )
    target_attrs = _load_node_attrs(
        cfg.target_node_csv_path,
        cfg.target_label,
        index_column=cfg.target_node_index_column,
        offset=cfg.target_node_offset,
        excluded_columns=cfg.excluded_node_attr_columns,
    )
    attrs = dict(source_attrs)
    attrs.update(target_attrs)
    print(
        f"[NodeAttrs/{cfg.relation_name}] source={len(source_attrs)} target={len(target_attrs)} "
        f"source_path={cfg.source_node_csv_path} target_path={cfg.target_node_csv_path}"
    )
    return attrs


def _promote_edge_attrs_to_nodes(source: Vertex, target: Vertex, attrs: Dict[str, object], cfg: RelationGraphConfig) -> None:
    for columns, vertex in ((cfg.source_edge_attr_columns, source), (cfg.target_edge_attr_columns, target)):
        for column in columns:
            key = _normalize_key(column)
            if key not in attrs:
                continue
            vertex.attrs[key] = _merge_attr(vertex.attrs.get(key), attrs[key])
            attrs.pop(key, None)
    for column in cfg.excluded_edge_attr_columns:
        attrs.pop(_normalize_key(column), None)


def load_relation_csv_graph(
    relation_config: RelationGraphConfig,
    interaction_path: str,
    max_rows: Optional[int] = None,
    undirected: bool = False,
    protein_path: Optional[str] = None,
    protein_index_column: str = "index",
    edge_label_column: str = "EdgeLabel",
    force_edge_label: Optional[str] = None,
) -> DataGraph:
    _raise_csv_field_limit()
    vertices = _load_relation_node_attrs(relation_config)
    graph = DataGraph(vertices=vertices)
    with Path(interaction_path).open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row_index, row in enumerate(reader):
            if max_rows is not None and row_index >= max_rows:
                break
            source_value = _normalize_scalar(row.get(relation_config.source_index_column))
            target_value = _normalize_scalar(row.get(relation_config.target_index_column))
            if source_value is None or target_value is None:
                continue
            source_id = int(source_value)
            target_id = int(target_value) + relation_config.target_node_offset
            if source_id == target_id:
                continue
            vertices.setdefault(source_id, Vertex(id=source_id, label=relation_config.source_label))
            vertices.setdefault(target_id, Vertex(id=target_id, label=relation_config.target_label))
            attrs = _edge_attrs_from_row(row)
            attrs.setdefault("source_row_id", row_index)
            attrs.setdefault("interaction_label", str(row.get("interaction_label", "unknown")).strip().lower() or "unknown")
            _promote_edge_attrs_to_nodes(vertices[source_id], vertices[target_id], attrs, relation_config)
            edge_label = force_edge_label or _normalize_edge_label(row.get(edge_label_column, relation_config.default_edge_label))
            graph.add_edge(source_id, target_id, edge_label, attrs)
            if undirected:
                graph.add_edge(target_id, source_id, edge_label, dict(attrs, direction_role="reverse_copy"))
    _assign_degree_features(graph)
    return graph


def build_source_seed_pattern(graph: DataGraph, source_label: str) -> FrequentPattern:
    pattern = GraphPattern(node_labels=[source_label])
    instances = [
        GraphInstance(node_map={0: node_id}, edge_ids=(), pivot=node_id)
        for node_id, vertex in graph.vertices.items()
        if vertex.label == source_label
    ]
    return FrequentPattern(pattern=pattern, instances=instances)
