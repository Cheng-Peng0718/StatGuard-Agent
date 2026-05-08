from pathlib import Path


def test_execute_pending_plan_node_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    node_text = Path("core/workflow/nodes/plan_execution.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "def execute_pending_plan_node" not in graph_text
    assert "def execute_pending_plan_node" in node_text


def test_core_graph_imports_execute_pending_plan_node_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert (
        "from core.workflow.nodes.plan_execution import execute_pending_plan_node"
        in graph_text
    )

    forbidden_imports = [
        "find_next_executable_step",
        "mark_plan_step_started",
        "modeling_blocked_by_pending_cleaning",
    ]

    graph_import_area = graph_text[:2500]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_import_area