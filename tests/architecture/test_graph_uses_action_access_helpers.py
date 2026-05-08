from pathlib import Path


def test_core_graph_does_not_directly_read_action_attributes():
    text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_patterns = [
        "action.tool_name",
        "action.arguments",
        "action.action_type",
        "action.reasoning_summary",
        "current_action.tool_name",
        "current_action.arguments",
        "current_action.action_type",
        "current_action.reasoning_summary",
    ]

    for pattern in forbidden_patterns:
        assert pattern not in text