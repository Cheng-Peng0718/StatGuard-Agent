from core.analysis_tool_plugins.validation import validate_plugin_action
from core.schema import ActionProposal


def test_clean_data_drop_drop_is_canonicalized_to_drop_rows():
    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "drop",
            "strategy": "drop",
            "columns": ["GPA", "SATM"],
        },
        reasoning_summary="Drop missing rows.",
    )

    result = validate_plugin_action(action, profile=None)

    assert result.status == "needs_review"
    assert action.arguments["strategy"] == "rows"
    assert result.details["canonical_arguments"]["strategy"] == "rows"


def test_clean_data_invalid_strategy_is_rejected_before_review():
    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "impute",
            "strategy": "rows",
            "columns": ["GPA"],
        },
        reasoning_summary="Invalid imputation.",
    )

    result = validate_plugin_action(action, profile=None)

    assert result.status == "rejected_recoverable"
    assert result.error_code == "INVALID_TOOL_ARGUMENTS"
    assert result.details["conditional_violations"]


def test_unknown_tool_is_rejected_terminal():
    action = ActionProposal(
        action_id="act_test",
        action_type="tool_call",
        tool_name="not_a_real_tool",
        arguments={},
        reasoning_summary="Invalid tool.",
    )

    result = validate_plugin_action(action, profile=None)

    assert result.status == "rejected_terminal"
    assert result.error_code == "TOOL_NOT_REGISTERED"