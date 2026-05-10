from pathlib import Path


def test_interaction_node_uses_llm_interaction_parser_boundary():
    text = Path("core/workflow/nodes/interaction.py").read_text(encoding="utf-8")

    assert "core.services.llm_interaction_parser" in text
    assert "decide_llm_interaction_intent" in text
    assert "core.services.interaction_router" not in text
    assert "core.interaction_intent" not in text


def test_llm_interaction_parser_defines_structured_output_boundary():
    text = Path("core/services/llm_interaction_parser.py").read_text(encoding="utf-8")

    assert "def generate_llm_interaction_draft" in text
    assert "with_structured_output(" in text
    assert "LLMInteractionDraft" in text
    assert "method=" in text
    assert "function_calling" in text


def test_llm_interaction_parser_does_not_import_workflow_or_ui():
    text = Path("core/services/llm_interaction_parser.py").read_text(encoding="utf-8")

    forbidden = [
        "core.workflow.nodes",
        "core.graph",
        "core.runtime",
        "core.ui_adapter",
        "streamlit",
    ]

    for item in forbidden:
        assert item not in text