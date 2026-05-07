import pandas as pd

from core.audit.execution_state import audit_execution_state
from core.controller.backend_turn import run_backend_turn
from core.ui_adapter.dataset_upload import prepare_uploaded_dataset_state
from core.ui_adapter.events import (
    apply_ui_event_to_state,
    make_run_plan_event,
    make_user_message_event,
)


def apply_updates(state, updates):
    merged = dict(state)
    merged.update(updates)
    return merged


def run_turn_after_event(state, event):
    updates = apply_ui_event_to_state(state, event)
    state = apply_updates(state, updates)

    result = run_backend_turn(state)

    return result["state"], result


def completed_tools(state):
    plan = state.get("pending_plan") or {}

    return {
        step.get("tool_name")
        for step in (plan.get("steps") or [])
        if step.get("execution_status") == "completed"
    }


def test_app_v2_smoke_upload_plan_safe_eda_and_audit_ok(tmp_path):
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, None, 3.8, 4.0],
        "SATM": [600, None, 650, 680, 700],
        "Sex": ["F", "M", "F", "M", "F"],
    })

    state = prepare_uploaded_dataset_state(
        df=df,
        workspace_dir=str(tmp_path / "workspace"),
        filename="test_data.csv",
    )

    # Advisory uses the real uploaded dataset summary.
    state, advisory_result = run_turn_after_event(
        state,
        make_user_message_event(
            "I want to do analysis to this dataset, what can I do?"
        ),
    )

    assert advisory_result["status"] == "ok"

    advisory_content = advisory_result["ui_snapshot"]["assistant_response"]["content"]

    assert "Rows: 5" in advisory_content
    assert "Columns: 3" in advisory_content
    assert "Numeric columns: 2" in advisory_content

    # Plan-only creates a pending plan without executing tools.
    state, plan_result = run_turn_after_event(
        state,
        make_user_message_event("Could you make up a plan and tell me?"),
    )

    assert plan_result["status"] == "ok"
    assert state.get("pending_plan") is not None
    assert state.get("analysis_runs") == []
    assert state.get("observations") == []

    # Run plan repeatedly until safe EDA steps have completed or no progress.
    expected_safe_tools = {
        "get_summary_stats",
        "missingness_report",
        "get_correlation_matrix",
    }

    previous_completed = set()

    for _ in range(6):
        state, run_result = run_turn_after_event(
            state,
            make_run_plan_event(),
        )

        current_completed = completed_tools(state)

        # Execution/state audits should stay structurally valid after each turn.
        audit = audit_execution_state(state)
        assert audit.status == "ok", audit.model_dump()

        if expected_safe_tools.issubset(current_completed):
            break

        if current_completed == previous_completed and run_result["status"] in {
            "blocked",
            "needs_review",
        }:
            break

        previous_completed = current_completed

    assert expected_safe_tools.issubset(completed_tools(state))

    observation_ids = {
        obs["observation_id"]
        for obs in state.get("observations", [])
    }

    assert len(state.get("analysis_runs", [])) >= 3

    for run in state.get("analysis_runs", []):
        assert run["observation_id"] in observation_ids

    snapshot = run_result["ui_snapshot"]

    assert snapshot["audits"]["execution_audit"]["status"] == "ok"
    assert snapshot["audits"]["state_serialization_audit"]["status"] == "ok"