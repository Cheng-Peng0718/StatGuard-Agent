from pathlib import Path


def test_plan_only_node_uses_llm_planner_boundary():
    text = Path("core/workflow/nodes/planning.py").read_text(encoding="utf-8")

    assert "from core.services.llm_planner import create_llm_plan_from_state" in text
    assert "create_llm_plan_from_state(state)" in text
    assert "core.services.intelligent_planner" not in text
    assert "create_plan_from_state(state)" not in text


def test_llm_planner_temporarily_delegates_to_deterministic_fallback():
    text = Path("core/services/llm_planner.py").read_text(encoding="utf-8")

    assert "from core.services.intelligent_planner import create_plan_from_state" in text
    assert "return create_plan_from_state(state)" in text

def test_llm_planner_defines_input_contract_boundary():
    text = Path("core/services/llm_planner.py").read_text(encoding="utf-8")

    assert "def build_llm_planner_input" in text
    assert "LLMPlannerInput" in text
    assert "build_tool_manifests" in text
    assert "PLUGIN_REGISTRY" in text


def test_llm_planner_contracts_do_not_import_runtime_or_ui():
    text = Path("core/services/llm_planner_contracts.py").read_text(encoding="utf-8")

    forbidden = [
        "core.workflow",
        "core.graph",
        "core.runtime",
        "core.ui",
        "streamlit",
        "ChatOpenAI",
    ]

    for item in forbidden:
        assert item not in text