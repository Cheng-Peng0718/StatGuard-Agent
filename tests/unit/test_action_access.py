from core.action_access import (
    get_action_arguments,
    get_action_id,
    get_action_reasoning_summary,
    get_action_tool_name,
    get_action_type,
    has_action_tool_name,
)
from core.schema import ActionProposal


def test_action_access_supports_action_proposal():
    action = ActionProposal(
        action_id="act_1",
        action_type="tool_call",
        tool_name="get_summary_stats",
        arguments={"columns": ["GPA"]},
        reasoning_summary="Compute GPA summary.",
    )

    assert get_action_id(action) == "act_1"
    assert get_action_type(action) == "tool_call"
    assert get_action_tool_name(action) == "get_summary_stats"
    assert get_action_arguments(action) == {"columns": ["GPA"]}
    assert get_action_reasoning_summary(action) == "Compute GPA summary."
    assert has_action_tool_name(action) is True


def test_action_access_supports_dict_action():
    action = {
        "action_id": "act_2",
        "action_type": "tool_call",
        "tool_name": "run_multiple_regression",
        "arguments": {"target_col": "GPA", "feature_cols": ["SATM"]},
        "reasoning_summary": "Run regression.",
    }

    assert get_action_id(action) == "act_2"
    assert get_action_type(action) == "tool_call"
    assert get_action_tool_name(action) == "run_multiple_regression"
    assert get_action_arguments(action) == {
        "target_col": "GPA",
        "feature_cols": ["SATM"],
    }
    assert get_action_reasoning_summary(action) == "Run regression."
    assert has_action_tool_name(action) is True


def test_action_access_handles_missing_or_bad_arguments():
    assert get_action_arguments(None) == {}
    assert get_action_arguments({"arguments": None}) == {}
    assert get_action_arguments({"arguments": ["bad"]}) == {}
    assert has_action_tool_name(None) is False


def test_action_reasoning_summary_falls_back_to_legacy_fields():
    assert get_action_reasoning_summary({"summary": "Legacy summary."}) == "Legacy summary."
    assert get_action_reasoning_summary({"message": "Legacy message."}) == "Legacy message."