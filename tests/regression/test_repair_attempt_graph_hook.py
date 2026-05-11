from types import SimpleNamespace

from core.workflow.nodes.summarization import summarize_node
from core.workflow.repair_runtime import attach_repair_decision

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


def test_attach_repair_decision_creates_repair_attempt_for_recoverable_failure():
    state = {
        "current_action": make_action(
            action_id="act_clean",
            tool_name="clean_data",
            arguments={},
        ),
        "repair_attempts": [],
    }

    updates = {
        "current_verification": {
            "status": "rejected_recoverable",
            "error_code": "INVALID_TOOL_ARGUMENTS",
            "feedback": "Invalid arguments.",
        }
    }

    result = attach_repair_decision(state, updates)

    assert "repair_decision" in result
    assert result["repair_decision"]["status"] in {"repairable", "needs_user"}

    assert "repair_attempts" in result
    assert len(result["repair_attempts"]) == 1

    attempt = result["repair_attempts"][0]

    assert attempt["source_action_id"] == "act_clean"
    assert attempt["source_tool_name"] == "clean_data"
    assert attempt["decision_status"] in {"repairable", "needs_user"}
    assert attempt["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert attempt["repair_status"] == "proposed"
    assert attempt["metadata"]["observe_only"] is True


def test_attach_repair_decision_does_not_create_attempt_for_terminal_failure():
    state = {
        "current_action": make_action(
            action_id="act_reg",
            tool_name="run_multiple_regression",
            arguments={
                "target_col": "GPA",
                "feature_cols": ["SATM"],
            },
        ),
        "repair_attempts": [],
    }

    updates = {
        "current_execution": {
            "status": "failed",
            "success": False,
            "error_code": "INTERNAL_PLUGIN_ERROR",
            "message": "Plugin crashed.",
        }
    }

    result = attach_repair_decision(state, updates)

    assert "repair_decision" in result
    assert result["repair_decision"]["status"] == "terminal"
    assert "repair_attempts" not in result


def test_attach_repair_decision_respects_max_attempts():
    state = {
        "current_action": make_action(
            action_id="act_clean",
            tool_name="clean_data",
            arguments={},
        ),
        "repair_attempts": [
            {
                "repair_attempt_id": "repair_1",
                "source_action_id": "act_clean",
                "source_tool_name": "clean_data",
            },
            {
                "repair_attempt_id": "repair_2",
                "source_action_id": "act_clean",
                "source_tool_name": "clean_data",
            },
        ],
    }

    updates = {
        "current_verification": {
            "status": "rejected_recoverable",
            "error_code": "INVALID_TOOL_ARGUMENTS",
        }
    }

    result = attach_repair_decision(state, updates)

    assert "repair_decision" in result

    # clean_data policy currently has finite max_attempts.
    # If max_attempts is 2, this should not append a third attempt.
    if "repair_attempts" in result:
        assert len(result["repair_attempts"]) <= 2


def test_summarize_node_records_repair_attempt_for_repairable_failed_execution():
    state = {
        "current_action": make_action(
            action_id="act_clean_failed",
            tool_name="clean_data",
            arguments={
                "action_type": "drop",
                "strategy": "bad_strategy",
            },
        ),
        "current_execution": {
            "execution_id": "exec_failed",
            "status": "failed",
            "success": False,
            "error_code": "INVALID_TOOL_ARGUMENTS",
            "message": "Invalid clean_data strategy.",
            "artifacts": [],
            "payload": {},
        },
        "observations": [],
        "analysis_runs": [],
        "repair_attempts": [],
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
    assert result["repair_decision"]["tool_name"] == "clean_data"

    if result["repair_decision"]["status"] in {"repairable", "needs_user"}:
        assert "repair_attempts" in result
        assert len(result["repair_attempts"]) == 1
        assert result["repair_attempts"][0]["source_action_id"] == "act_clean_failed"

    # Still observe-only: failed run is recorded, not retried.
    assert len(result["analysis_runs"]) == 1
    assert result["analysis_runs"][0]["status"] == "failed"