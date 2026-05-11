from types import SimpleNamespace

from core.repair.decision import evaluate_repair_decision


def make_action(tool_name="clean_data"):
    return SimpleNamespace(
        action_id="act_1",
        action_type="tool_call",
        tool_name=tool_name,
        arguments={},
    )


def test_repair_decision_no_repair_needed_when_no_failure():
    result = evaluate_repair_decision({
        "current_action": make_action("get_summary_stats"),
        "current_verification": {
            "status": "allowed",
        },
        "current_execution": {
            "status": "ok",
            "success": True,
        },
    })

    assert result.status == "no_repair_needed"


def test_repair_decision_terminal_for_rejected_terminal_verification():
    result = evaluate_repair_decision({
        "current_action": make_action("clean_data"),
        "current_verification": {
            "status": "rejected_terminal",
            "error_code": "MALFORMED_TOOL_CONTRACT",
        },
    })

    assert result.status == "terminal"
    assert result.error_code == "MALFORMED_TOOL_CONTRACT"


def test_repair_decision_repairable_for_rejected_recoverable_clean_data():
    result = evaluate_repair_decision({
        "current_action": make_action("clean_data"),
        "current_verification": {
            "status": "rejected_recoverable",
            "error_code": "INVALID_TOOL_ARGUMENTS",
        },
    })

    assert result.status in {"repairable", "needs_user"}
    assert result.tool_name == "clean_data"
    assert result.error_code == "INVALID_TOOL_ARGUMENTS"


def test_repair_decision_terminal_for_internal_plugin_error():
    result = evaluate_repair_decision({
        "current_action": make_action("run_multiple_regression"),
        "current_execution": {
            "status": "failed",
            "success": False,
            "error_code": "INTERNAL_PLUGIN_ERROR",
        },
    })

    assert result.status == "terminal"
    assert result.error_code == "INTERNAL_PLUGIN_ERROR"


def test_repair_decision_needs_user_for_missing_columns_when_policy_requires_user():
    result = evaluate_repair_decision({
        "current_action": make_action("run_multiple_regression"),
        "current_verification": {
            "status": "rejected_recoverable",
            "error_code": "MISSING_COLUMNS",
        },
    })

    assert result.status in {"needs_user", "repairable"}
    assert result.error_code == "MISSING_COLUMNS"


def test_repair_decision_terminal_when_tool_has_no_plugin():
    result = evaluate_repair_decision({
        "current_action": make_action("not_a_real_tool"),
        "current_execution": {
            "status": "failed",
            "success": False,
            "error_code": "SOME_ERROR",
        },
    })

    assert result.status == "terminal"
    assert result.tool_name == "not_a_real_tool"