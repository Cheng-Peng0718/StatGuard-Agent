from types import SimpleNamespace

from core.audit.execution_state import audit_execution_state
from core.controller.backend_turn import _apply_updates
from core.graph import summarize_node


def make_action(action_id, tool_name):
    return SimpleNamespace(
        action_id=action_id,
        action_type="tool_call",
        tool_name=tool_name,
        arguments={},
    )


def summarize_success(state, *, action_id, tool_name, execution_id):
    state = dict(state)

    state["current_action"] = make_action(
        action_id=action_id,
        tool_name=tool_name,
    )

    state["current_execution"] = {
        "execution_id": execution_id,
        "status": "ok",
        "success": True,
        "error_code": None,
        "message": f"{tool_name} completed.",
        "artifacts": [],
        "payload": {
            "summary": f"{tool_name} completed.",
        },
    }

    updates = summarize_node(state)

    return _apply_updates(state, updates)


def test_controller_preserves_observations_for_all_analysis_runs():
    state = {
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "data_audit_log": [],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
        "pending_plan": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
    }

    state = summarize_success(
        state,
        action_id="act_summary",
        tool_name="get_summary_stats",
        execution_id="exec_summary",
    )

    state = summarize_success(
        state,
        action_id="act_missingness",
        tool_name="missingness_report",
        execution_id="exec_missingness",
    )

    state = summarize_success(
        state,
        action_id="act_corr",
        tool_name="get_correlation_matrix",
        execution_id="exec_corr",
    )

    observation_ids = {
        obs["observation_id"]
        for obs in state["observations"]
    }

    assert len(state["observations"]) == 3
    assert len(state["analysis_runs"]) == 3

    for run in state["analysis_runs"]:
        assert run["observation_id"] in observation_ids


def test_execution_audit_accepts_multi_step_observation_links():
    state = {
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "data_audit_log": [],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
        "pending_plan": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
    }

    state = summarize_success(
        state,
        action_id="act_summary",
        tool_name="get_summary_stats",
        execution_id="exec_summary",
    )

    state = summarize_success(
        state,
        action_id="act_missingness",
        tool_name="missingness_report",
        execution_id="exec_missingness",
    )

    audit = audit_execution_state(state)

    assert audit.status == "ok", audit.model_dump()