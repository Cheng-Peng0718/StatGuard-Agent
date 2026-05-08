from pathlib import Path


def test_basic_routes_live_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    routes_text = Path("core/workflow/routes.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def route_after_intent",
        "def route_after_execute_pending_plan",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    required_defs = [
        "def route_after_intent",
        "def route_after_execute_pending_plan",
    ]

    for required in required_defs:
        assert required in routes_text


def test_core_graph_imports_basic_routes_from_workflow_routes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.routes import" in graph_text
    assert "route_after_intent" in graph_text
    assert "route_after_execute_pending_plan" in graph_text