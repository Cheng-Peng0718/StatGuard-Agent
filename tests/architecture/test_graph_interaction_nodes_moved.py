from pathlib import Path


def test_interaction_nodes_live_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    interaction_text = Path("core/workflow/nodes/interaction.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def intent_router_node",
        "def advisory_answer_node",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    required_defs = [
        "def intent_router_node",
        "def advisory_answer_node",
    ]

    for required in required_defs:
        assert required in interaction_text


def test_core_graph_imports_interaction_nodes_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.nodes.interaction import" in graph_text
    assert "intent_router_node" in graph_text
    assert "advisory_answer_node" in graph_text
    assert "from core.interaction_intent import classify_interaction_intent" not in graph_text