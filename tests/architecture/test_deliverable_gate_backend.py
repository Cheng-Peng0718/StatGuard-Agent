from pathlib import Path


def test_deliverable_gate_node_uses_backend_evaluator():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    assert "evaluate_deliverable_gate_state" in graph_text
    assert "def deliverable_gate_node" in graph_text