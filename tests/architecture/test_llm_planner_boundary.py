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