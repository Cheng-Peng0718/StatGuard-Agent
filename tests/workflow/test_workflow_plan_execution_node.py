from core.action_access import get_action_arguments, get_action_tool_name
from core.workflow.nodes.plan_execution import execute_pending_plan_node


def test_execute_pending_plan_without_plan_returns_response_only():
    result = execute_pending_plan_node({
        "pending_plan": None,
        "active_data_version_id": "raw_v1",
    })

    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None
    assert result["plan_execution_status"] == "no_pending_plan"
    assert result["assistant_response"]["response_type"] == "plan_execution_status"


def test_execute_pending_plan_with_no_ready_step_does_not_create_action():
    state = {
        "active_data_version_id": "raw_v1",
        "pending_plan": {
            "plan_id": "plan_1",
            "status": "partially_ready",
            "steps": [
                {
                    "step_id": "s1",
                    "tool_name": "run_multiple_regression",
                    "status": "needs_user_choice",
                    "execution_ready": False,
                    "execution_status": "not_started",
                    "arguments": {},
                    "variables": {},
                    "required_user_choices": ["target_col"],
                }
            ],
        },
    }

    result = execute_pending_plan_node(state)

    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None
    assert result["plan_execution_status"] == "blocked_no_ready_steps"


def test_execute_pending_plan_creates_action_for_ready_step():
    state = {
        "active_data_version_id": "raw_v1",
        "pending_plan": {
            "plan_id": "plan_1",
            "status": "partially_ready",
            "steps": [
                {
                    "step_id": "s1",
                    "tool_name": "get_summary_stats",
                    "status": "ready",
                    "execution_ready": True,
                    "execution_status": "not_started",
                    "arguments": {
                        "columns": ["GPA"],
                    },
                    "variables": {},
                    "required_user_choices": [],
                }
            ],
        },
    }

    result = execute_pending_plan_node(state)

    assert result["current_action"] is not None
    assert get_action_tool_name(result["current_action"]) == "get_summary_stats"
    assert get_action_arguments(result["current_action"]) == {"columns": ["GPA"]}
    assert result["current_plan_step_id"] == "s1"
    assert result["plan_execution_status"] == "started_step"
    assert result["action_origin"] == "pending_plan"