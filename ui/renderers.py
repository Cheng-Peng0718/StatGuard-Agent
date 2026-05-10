from __future__ import annotations

from typing import Any, Dict, Iterable, List

import streamlit as st


def render_json_expander(
    title: str,
    payload: Any,
    *,
    expanded: bool = False,
) -> None:
    with st.expander(title, expanded=expanded):
        st.json(payload)


def rows_from_column_profile(columns: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []

    for name, meta in columns.items():
        rows.append({
            "column": name,
            "semantic_type": meta.get("semantic_type", "unknown"),
            "dtype": meta.get("dtype", "unknown"),
            "missing_rate": meta.get("missing_rate", 0.0),
            "n_unique": meta.get("n_unique"),
        })

    return rows


def render_key_value_captions(items: Iterable[tuple[str, Any]]) -> None:
    for label, value in items:
        st.caption(f"{label}: `{value}`")