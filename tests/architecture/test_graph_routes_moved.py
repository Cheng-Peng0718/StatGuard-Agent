from pathlib import Path


def test_route_after_intent_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    routes_text = Path("core/workflow/routes.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "def route_after_intent" not in graph_text
    assert "def route_after_intent" in routes_text


def test_core_graph_imports_route_after_intent_from_workflow_routes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.routes import route_after_intent" in graph_text