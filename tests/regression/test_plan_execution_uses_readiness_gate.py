from core.planning.execution_queue import find_next_executable_step


def test_execution_queue_does_not_select_clean_data_with_empty_arguments():
    plan = {
        "plan_id": "plan_test",
        "steps": [
            {
                "step_id": "s_clean",
                "tool_name": "clean_data",
                "status": "ready",
                "execution_ready": True,
                "arguments": {},
            }
        ],
    }

    step, readiness = find_next_executable_step(plan)

    assert step is None
    assert readiness is not None
    assert readiness.executable is False
    assert readiness.status == "needs_user_choice"
    assert "action_type" in readiness.missing_required_arguments
    assert "strategy" in readiness.missing_required_arguments


def test_execution_queue_selects_summary_stats_ready_step():
    plan = {
        "plan_id": "plan_test",
        "steps": [
            {
                "step_id": "s_summary",
                "tool_name": "get_summary_stats",
                "status": "ready",
                "execution_ready": True,
                "arguments": {},
            }
        ],
    }

    step, readiness = find_next_executable_step(plan)

    assert step is not None
    assert step["step_id"] == "s_summary"
    assert readiness is not None
    assert readiness.executable is True
    assert readiness.action.tool_name == "get_summary_stats"


def test_execution_queue_skips_bad_step_and_selects_next_executable_step():
    plan = {
        "plan_id": "plan_test",
        "steps": [
            {
                "step_id": "s_clean",
                "tool_name": "clean_data",
                "status": "ready",
                "execution_ready": True,
                "arguments": {},
            },
            {
                "step_id": "s_summary",
                "tool_name": "get_summary_stats",
                "status": "ready",
                "execution_ready": True,
                "arguments": {},
            },
        ],
    }

    step, readiness = find_next_executable_step(plan)

    assert step is not None
    assert step["step_id"] == "s_summary"
    assert readiness.executable is True
    assert readiness.action.tool_name == "get_summary_stats"