from core.graph import execute_pending_plan_node


def test_execute_pending_plan_without_plan_returns_response_only():
    state = {
        "active_data_version_id": "raw_v1",
        "pending_plan": None,
    }

    result = execute_pending_plan_node(state)

    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None
    assert result["assistant_response"]["response_type"] == "plan_execution_status"


def test_execute_pending_plan_with_no_ready_step_does_not_create_action():
    state = {
        "active_data_version_id": "raw_v1",
        "pending_plan": {
            "plan_id": "plan_test",
            "status": "partially_ready",
            "steps": [
                {
                    "step_id": "s1",
                    "title": "Needs variables",
                    "tool_name": "get_summary_stats",
                    "status": "needs_user_choice",
                    "execution_ready": False,
                    "required_user_choices": ["outcome"],
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
            "plan_id": "plan_test",
            "status": "verified",
            "steps": [
                {
                    "step_id": "s1",
                    "title": "Ready Step",
                    "tool_name": "get_summary_stats",
                    "status": "ready",
                    "execution_ready": True,
                    "arguments": {},
                }
            ],
        },
    }

    result = execute_pending_plan_node(state)

    assert result["current_action"] is not None
    assert result["current_action"].tool_name == "get_summary_stats"
    assert result["current_action"].arguments == {}
    assert result["current_plan_step_id"] == "s1"
    assert result["plan_execution_status"] == "started_step"
    assert result["action_origin"] == "pending_plan"

from core.graph import route_after_verify


class DummyVerification:
    def __init__(self, status):
        self.status = status


def test_route_after_verify_ends_for_rejected_pending_plan_action():
    state = {
        "action_origin": "pending_plan",
        "current_verification": DummyVerification("rejected_recoverable"),
    }

    assert route_after_verify(state) == "end"