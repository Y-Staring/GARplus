from __future__ import annotations

"""Helpers for optionally using edge attributes as structural pattern labels."""

from typing import Mapping, Optional

MISSING_VALUES = {"", "-", "null", "none", "nan", "na", "n/a", "inf", "-inf"}


def _normalize_edge_label(raw: object) -> str:
    text = str(raw or "interacts_with").strip() or "interacts_with"
    return text.replace(" ", "_").replace("/", "_")


def direct_evidence_category(attrs: Mapping[str, object]) -> str:
    raw = str(attrs.get("direct_evidence_category") or attrs.get("directevidence") or "").strip().lower()
    if not raw or raw in MISSING_VALUES:
        return "inference_evidence"
    if raw == "marker/mechanism":
        return "marker_mechanism"
    return "other"


def structural_edge_label(
    base_label: str,
    attrs: Mapping[str, object],
    enabled: bool = False,
    attr_key: Optional[str] = None,
    separator: str = ":",
) -> str:
    """Return either the base relation label or a base+attribute structural label."""

    base = _normalize_edge_label(base_label)
    if not enabled or not attr_key:
        return base
    key = attr_key.strip().lower()
    if key == "direct_evidence_category":
        value = direct_evidence_category(attrs)
    else:
        value = attrs.get(key)
    value_text = str(value or "").strip().lower()
    if not value_text or value_text in MISSING_VALUES:
        value_text = "missing"
    return f"{base}{separator}{_normalize_edge_label(value_text)}"
