from core.planning.execution_queue import find_next_executable_step


def test_running_plan_step_is_not_selected_again():
    plan = {
        "plan_id": "plan_test",
        "steps": [
            {
                "step_id": "s1",
                "tool_name": "clean_data",
                "status": "ready",
                "execution_ready": True,
                "execution_status": "running",
                "arguments": {},
            }
        ],
    }

    step, readiness = find_next_executable_step(plan)

    assert step is None
    assert readiness is not None
    assert readiness.executable is False
    assert readiness.status == "not_executable"


def test_failed_plan_step_is_not_selected_again():
    plan = {
        "plan_id": "plan_test",
        "steps": [
            {
                "step_id": "s1",
                "tool_name": "clean_data",
                "status": "ready",
                "execution_ready": True,
                "execution_status": "failed",
                "arguments": {},
            }
        ],
    }

    step, readiness = find_next_executable_step(plan)

    assert step is None
    assert readiness is not None
    assert readiness.executable is False
    assert readiness.status == "not_executable"

import pandas as pd

from core.analysis_tool_plugins import get_plugin
from core.dataset_intelligence.profiler import profile_dataframe
from core.dataset_intelligence.capability_map import build_capability_map


def test_clean_data_is_not_execution_ready_without_action_type_and_strategy():
    df = pd.DataFrame({
        "x": [1.0, None, 3.0],
        "y": [1.0, 2.0, None],
    })

    profile = profile_dataframe(df, data_version_id="raw_v1")
    plugin = get_plugin("clean_data")

    capability_map = build_capability_map(
        profile,
        plugins={"clean_data": plugin},
    )

    cap = capability_map.capabilities[0]

    assert cap.tool_name == "clean_data"
    assert cap.status == "needs_user_choice"
    assert "action_type" in cap.required_user_choices
    assert "strategy" in cap.required_user_choices