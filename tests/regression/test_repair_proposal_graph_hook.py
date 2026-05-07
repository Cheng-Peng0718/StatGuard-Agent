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


def test_attach_repair_decision_creates_argument_repair_proposal():
    state = {
        "current_action": make_action(
            action_id="act_clean",
            tool_name="clean_data",
            arguments={
                "action_type": "drop rows",
                "strategy": "drop",
                "columns": ["GPA"],
            },
        ),
        "repair_attempts": [],
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

    assert "repair_proposal" in result
    proposal = result["repair_proposal"]

    assert proposal["source_action_id"] == "act_clean"
    assert proposal["source_tool_name"] == "clean_data"
    assert proposal["proposal_type"] in {"argument_repair", "ask_user", "no_op"}

    if proposal["proposal_type"] == "argument_repair":
        assert proposal["proposed_arguments"]["action_type"] == "drop"
        assert proposal["proposed_arguments"]["strategy"] == "rows"


def test_attach_repair_decision_creates_no_op_proposal_for_terminal_failure():
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

    result = _attach_repair_decision(state, updates)

    assert "repair_decision" in result
    assert result["repair_decision"]["status"] == "terminal"

    assert "repair_proposal" in result
    assert result["repair_proposal"]["proposal_type"] == "no_op"
    assert result["repair_proposal"]["source_action_id"] == "act_reg"

    # Terminal failure should not create repair attempts.
    assert "repair_attempts" not in result


def test_summarize_node_attaches_repair_proposal_for_failed_execution():
    state = {
        "current_action": make_action(
            action_id="act_clean_failed",
            tool_name="clean_data",
            arguments={
                "action_type": "drop rows",
                "strategy": "drop",
                "columns": ["GPA"],
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

    assert "repair_proposal" in result
    assert result["repair_proposal"]["source_action_id"] == "act_clean_failed"
    assert result["repair_proposal"]["source_tool_name"] == "clean_data"

    # Still observe-only: failed run is recorded, not retried.
    assert len(result["analysis_runs"]) == 1
    assert result["analysis_runs"][0]["status"] == "failed"


def test_no_repair_needed_does_not_create_repair_proposal():
    state = {
        "current_action": make_action(
            action_id="act_summary",
            tool_name="get_summary_stats",
            arguments={},
        ),
    }

    updates = {
        "current_verification": {
            "status": "allowed",
        },
        "current_execution": {
            "status": "ok",
            "success": True,
        },
    }

    result = _attach_repair_decision(state, updates)

    assert result["repair_decision"]["status"] == "no_repair_needed"
    assert "repair_proposal" not in result