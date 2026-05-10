from __future__ import annotations

from typing import Any, Dict, Optional


def _as_state_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)

    if hasattr(value, "dict"):
        dumped = value.dict()
        if isinstance(dumped, dict):
            return dict(dumped)

    raise TypeError(
        "Graph runner expected the LangGraph app to return a dict-like state."
    )


def run_graph_once(
    state: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run one invocation of the canonical LangGraph backend.

    This module is intentionally thin:
    - it does not import workflow nodes;
    - it does not manually route between nodes;
    - it does not build UI snapshots;
    - it does not execute tools outside the graph.

    The only runtime authority is core.graph.create_graph_app().
    """
    from core.graph import create_graph_app

    app = create_graph_app()
    input_state = dict(state or {})

    if config is None:
        result = app.invoke(input_state)
    else:
        result = app.invoke(input_state, config=config)

    return _as_state_dict(result)