from types import SimpleNamespace

from core.repair.proposal import (
    make_argument_repair_proposal,
    make_ask_user_repair_proposal,
    make_method_fallback_repair_proposal,
    make_no_op_repair_proposal,
)


def make_action(
    *,
    action_id="act_1",
    tool_name="clean_data",
    arguments=None,
):
    return SimpleNamespace(
        action_id=action_id,
        action_type="tool_call",
        tool_name=tool_name,
        arguments=arguments or {},
    )


def test_make_argument_repair_proposal_records_source_and_arguments():
    action = make_action(
        action_id="act_clean",
        tool_name="clean_data",
    )

    repair_decision = {
        "status": "repairable",
        "tool_name": "clean_data",
        "error_code": "INVALID_TOOL_ARGUMENTS",
    }

    proposal = make_argument_repair_proposal(
        repair_decision=repair_decision,
        current_action=action,
        proposed_arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA"],
        },
        reason="Normalize clean_data arguments.",
        requires_confirmation=True,
        risk_level="medium",
    )

    assert proposal["repair_proposal_id"].startswith("repair_prop_")
    assert proposal["source_action_id"] == "act_clean"
    assert proposal["source_tool_name"] == "clean_data"
    assert proposal["proposal_type"] == "argument_repair"
    assert proposal["proposed_tool_name"] == "clean_data"
    assert proposal["proposed_arguments"]["strategy"] == "rows"
    assert proposal["requires_confirmation"] is True
    assert proposal["source_error_code"] == "INVALID_TOOL_ARGUMENTS"


def test_make_ask_user_repair_proposal_requires_user():
    action = make_action(
        action_id="act_reg",
        tool_name="run_multiple_regression",
    )

    repair_decision = {
        "status": "needs_user",
        "tool_name": "run_multiple_regression",
        "error_code": "MISSING_COLUMNS",
    }

    proposal = make_ask_user_repair_proposal(
        repair_decision=repair_decision,
        current_action=action,
        prompt="Please choose predictor columns.",
        missing_fields=["feature_cols"],
    )

    assert proposal["proposal_type"] == "ask_user"
    assert proposal["requires_user"] is True
    assert proposal["requires_confirmation"] is False
    assert proposal["metadata"]["missing_fields"] == ["feature_cols"]


def test_make_method_fallback_repair_proposal_changes_tool():
    action = make_action(
        action_id="act_corr",
        tool_name="run_correlation_test",
    )

    repair_decision = {
        "status": "repairable",
        "tool_name": "run_correlation_test",
        "error_code": "METHOD_NOT_APPLICABLE",
    }

    proposal = make_method_fallback_repair_proposal(
        repair_decision=repair_decision,
        current_action=action,
        fallback_tool_name="get_correlation_matrix",
        proposed_arguments={
            "columns": ["GPA", "SATM"],
        },
        reason="Fallback to correlation matrix screening.",
    )

    assert proposal["proposal_type"] == "method_fallback"
    assert proposal["source_tool_name"] == "run_correlation_test"
    assert proposal["proposed_tool_name"] == "get_correlation_matrix"
    assert proposal["proposed_arguments"]["columns"] == ["GPA", "SATM"]


def test_make_no_op_repair_proposal_records_terminal_reason():
    action = make_action(
        action_id="act_reg",
        tool_name="run_multiple_regression",
    )

    repair_decision = {
        "status": "terminal",
        "tool_name": "run_multiple_regression",
        "error_code": "INTERNAL_PLUGIN_ERROR",
    }

    proposal = make_no_op_repair_proposal(
        repair_decision=repair_decision,
        current_action=action,
        reason="Internal plugin error is terminal.",
    )

    assert proposal["proposal_type"] == "no_op"
    assert proposal["requires_user"] is False
    assert proposal["requires_confirmation"] is False
    assert proposal["source_error_code"] == "INTERNAL_PLUGIN_ERROR"