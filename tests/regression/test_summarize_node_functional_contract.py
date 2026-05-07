from types import SimpleNamespace

from core.graph import summarize_node


def make_action(
    *,
    action_id="act_1",
    tool_name="test_tool",
    arguments=None,
):
    return SimpleNamespace(
        action_id=action_id,
        tool_name=tool_name,
        arguments=arguments or {},
    )


def test_summarize_node_creates_observation_analysis_run_and_audit_for_success():
    state = {
        "current_action": make_action(
            action_id="act_summary",
            tool_name="test_summary_tool",
            arguments={"columns": ["GPA"]},
        ),
        "current_execution": {
            "execution_id": "exec_1",
            "status": "ok",
            "success": True,
            "error_code": None,
            "message": "Tool completed successfully.",
            "artifacts": [],
            "payload": {
                "rows": 226,
            },
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

    assert "observations" in result
    assert len(result["observations"]) == 1

    obs = result["observations"][0]

    assert obs["tool_name"] == "test_summary_tool"
    assert obs["arguments"] == {"columns": ["GPA"]}
    assert obs["status"] == "ok"
    assert obs["success"] is True
    assert obs["data_version_id"] == "raw_v1"

    assert "analysis_runs" in result
    assert len(result["analysis_runs"]) == 1

    run = result["analysis_runs"][0]

    assert run["observation_id"] == obs["observation_id"]
    assert run["tool_name"] == "test_summary_tool"
    assert run["arguments"] == {"columns": ["GPA"]}
    assert run["status"] == "ok"
    assert run["success"] is True
    assert run["data_version_id"] == "raw_v1"

    assert "execution_audit" in result
    assert result["execution_audit"]["status"] == "ok"

    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None


def test_summarize_node_records_failed_execution_as_analysis_run():
    state = {
        "current_action": make_action(
            action_id="act_failed",
            tool_name="test_failed_tool",
            arguments={"target_col": "GPA"},
        ),
        "current_execution": {
            "execution_id": "exec_failed",
            "status": "failed",
            "success": False,
            "error_code": "MODEL_FIT_FAILED",
            "message": "The model failed to fit.",
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

    assert len(result["observations"]) == 1
    assert len(result["analysis_runs"]) == 1

    obs = result["observations"][0]
    run = result["analysis_runs"][0]

    assert obs["status"] == "failed"
    assert obs["success"] is False
    assert obs["error_code"] == "MODEL_FIT_FAILED"

    assert run["observation_id"] == obs["observation_id"]
    assert run["tool_name"] == "test_failed_tool"
    assert run["status"] == "failed"
    assert run["success"] is False
    assert run["error_code"] == "MODEL_FIT_FAILED"

    assert "execution_audit" in result
    assert result["execution_audit"]["status"] == "ok"


def test_summarize_node_marks_pending_plan_step_completed_on_success():
    state = {
        "current_action": make_action(
            action_id="act_plan_step",
            tool_name="test_plan_tool",
            arguments={},
        ),
        "current_execution": {
            "execution_id": "exec_plan_step",
            "status": "ok",
            "success": True,
            "error_code": None,
            "message": "Plan step completed.",
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
        "current_plan_step_id": "s1",
        "action_origin": "pending_plan",
        "pending_plan": {
            "plan_id": "plan_1",
            "status": "executing",
            "steps": [
                {
                    "step_id": "s1",
                    "tool_name": "test_plan_tool",
                    "status": "ready",
                    "execution_ready": True,
                    "execution_status": "running",
                }
            ],
        },
    }

    result = summarize_node(state)

    assert result["current_plan_step_id"] is None
    assert result["action_origin"] is None

    updated_plan = result["pending_plan"]
    step = updated_plan["steps"][0]

    assert step["step_id"] == "s1"
    assert step["execution_status"] == "completed"
    assert step["last_execution_id"] == "exec_plan_step"
    assert step["last_execution_message"] == "Plan step completed."

    assert result["plan_status"] in {
        "completed",
        "partially_executed",
        "partially_failed",
    }


def test_summarize_node_marks_pending_plan_step_failed_on_failure():
    state = {
        "current_action": make_action(
            action_id="act_plan_failed",
            tool_name="test_plan_failed_tool",
            arguments={},
        ),
        "current_execution": {
            "execution_id": "exec_plan_failed",
            "status": "failed",
            "success": False,
            "error_code": "TOOL_FAILED",
            "message": "Plan step failed.",
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
        "current_plan_step_id": "s1",
        "action_origin": "pending_plan",
        "pending_plan": {
            "plan_id": "plan_1",
            "status": "executing",
            "steps": [
                {
                    "step_id": "s1",
                    "tool_name": "test_plan_failed_tool",
                    "status": "ready",
                    "execution_ready": True,
                    "execution_status": "running",
                }
            ],
        },
    }

    result = summarize_node(state)

    updated_plan = result["pending_plan"]
    step = updated_plan["steps"][0]

    assert step["step_id"] == "s1"
    assert step["execution_status"] == "failed"
    assert step["last_execution_id"] == "exec_plan_failed"
    assert step["last_execution_message"] == "Plan step failed."

    assert len(result["analysis_runs"]) == 1
    assert result["analysis_runs"][0]["status"] == "failed"