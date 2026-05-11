from pathlib import Path


def test_execute_node_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    execution_text = Path("core/workflow/nodes/execution.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "def execute_node" not in graph_text
    assert "def execute_node" in execution_text


def test_core_graph_imports_execute_node_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.nodes.execution import execute_node" in graph_text

    forbidden_imports = [
        "from core.analysis_tool_plugins.execution import execute_analysis_tool",
        "from core.workflow.execution_fingerprints import has_duplicate_executed_action",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_text

    assert "sanitize_results" not in graph_text