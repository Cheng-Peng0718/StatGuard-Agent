from types import SimpleNamespace

from core.graph import _attach_repair_decision, summarize_node


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


def test_attach_repair_decision_records_recoverable_verification_failure():
    state = {
        "current_action": make_action(
            tool_name="clean_data",
            arguments={},
        ),
    }

    updates = {
        "current_verification": {
            "status": "rejected_recoverable",
            "error_code": "INVALID_TOOL_ARGUMENTS",
            "feedback": "Invalid clean_data arguments.",
        }
    }

    result = _attach_repair_decision(state, updates)

    assert "repair_decision" in result
    assert result["repair_decision"]["status"] in {"repairable", "needs_user"}
    assert result["repair_decision"]["tool_name"] == "clean_data"
    assert result["repair_decision"]["error_code"] == "INVALID_TOOL_ARGUMENTS"


def test_attach_repair_decision_records_terminal_execution_failure():
    state = {
        "current_action": make_action(
            tool_name="run_multiple_regression",
            arguments={
                "target_col": "GPA",
                "feature_cols": ["SATM"],
            },
        ),
    }

    updates = {
        "current_execution": {
            "status": "failed",
            "success": False,
            "error_code": "INTERNAL_PLUGIN_ERROR",
            "message": "Plugin crashed.",
        }
    }

    result = _attach_repair_decision(state, updates)

    assert "repair_decision" in result
    assert result["repair_decision"]["status"] == "terminal"
    assert result["repair_decision"]["tool_name"] == "run_multiple_regression"
    assert result["repair_decision"]["error_code"] == "INTERNAL_PLUGIN_ERROR"


def test_summarize_node_attaches_repair_decision_for_failed_execution():
    state = {
        "current_action": make_action(
            action_id="act_failed",
            tool_name="run_multiple_regression",
            arguments={
                "target_col": "GPA",
                "feature_cols": ["SATM"],
            },
        ),
        "current_execution": {
            "execution_id": "exec_failed",
            "status": "failed",
            "success": False,
            "error_code": "INTERNAL_PLUGIN_ERROR",
            "message": "Regression plugin crashed.",
            "artifacts": [],
            "payload": {},
        },
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
    }

    result = summarize_node(state)

    assert "repair_decision" in result
    assert result["repair_decision"]["status"] == "terminal"
    assert result["repair_decision"]["tool_name"] == "run_multiple_regression"
    assert result["repair_decision"]["error_code"] == "INTERNAL_PLUGIN_ERROR"

    # S13B is observe-only: summarize still records the failed run.
    assert len(result["analysis_runs"]) == 1
    assert result["analysis_runs"][0]["status"] == "failed"