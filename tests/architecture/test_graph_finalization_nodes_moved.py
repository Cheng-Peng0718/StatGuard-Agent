from pathlib import Path


def test_finalization_nodes_live_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    finalization_text = Path("core/workflow/nodes/finalization.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def final_response_node",
        "def deliverable_gate_node",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    required_defs = [
        "def final_response_node",
        "def deliverable_gate_node",
    ]

    for required in required_defs:
        assert required in finalization_text


def test_core_graph_imports_finalization_nodes_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.nodes.finalization import" in graph_text
    assert "deliverable_gate_node" in graph_text
    assert "final_response_node" in graph_text

    forbidden_imports = [
        "from core.deliverables.gate import evaluate_deliverable_gate_state",
        "from core.deliverables.evidence import extract_final_answer_content_from_state",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_text