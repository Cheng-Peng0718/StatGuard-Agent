from pathlib import Path


def test_supervisor_node_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    supervisor_text = Path("core/workflow/nodes/supervisor.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "def supervisor_node" not in graph_text
    assert "def supervisor_node" in supervisor_text


def test_core_graph_imports_supervisor_node_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.nodes.supervisor import supervisor_node" in graph_text
    assert "from agents.supervisor import call_supervisor" not in graph_text