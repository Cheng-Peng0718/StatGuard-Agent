from pathlib import Path


def test_human_review_node_uses_action_and_verification_codecs():
    text = Path("core/workflow/nodes/human_review.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.action_codec import action_to_state_dict" in text
    assert "from core.verification_codec import verification_to_state_dict" in text

    forbidden_patterns = [
        "source_action_id=action.action_id",
        "action.model_dump() if hasattr(action, \"model_dump\")",
        "vr.model_dump() if hasattr(vr, \"model_dump\")",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in text


def test_core_graph_imports_human_review_node_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert (
        "from core.workflow.nodes.human_review import human_review_node"
        in graph_text
    )
    assert "def human_review_node" not in graph_text
    assert "from core.action_codec import action_to_state_dict" not in graph_text
    assert "from core.verification_codec import verification_to_state_dict" not in graph_text