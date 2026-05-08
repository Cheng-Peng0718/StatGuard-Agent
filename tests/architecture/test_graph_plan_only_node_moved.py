from pathlib import Path


def test_plan_only_node_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    planning_text = Path("core/workflow/nodes/planning.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "def plan_only_node" not in graph_text
    assert "def plan_only_node" in planning_text


def test_core_graph_imports_plan_only_node_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.nodes.planning import plan_only_node" in graph_text

    forbidden_imports = [
        "from core.planning.planner import build_plan_from_capability_map",
        "from core.planning.verifier import verify_plan",
        "from core.planning.renderer import render_plan_for_user",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_text