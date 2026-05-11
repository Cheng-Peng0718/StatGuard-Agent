from pathlib import Path


def test_deliverable_gate_node_uses_backend_evaluator_from_finalization_module():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    finalization_text = Path("core/workflow/nodes/finalization.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "evaluate_deliverable_gate_state" not in graph_text
    assert "def deliverable_gate_node" not in graph_text

    assert "evaluate_deliverable_gate_state" in finalization_text
    assert "def deliverable_gate_node" in finalization_text