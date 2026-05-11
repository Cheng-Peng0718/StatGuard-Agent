import json
import pandas as pd
from core.workflow.nodes.interaction import (
    intent_router_node,
    advisory_answer_node,
)
from core.workflow.nodes.plan_execution import execute_pending_plan_node
from core.workflow.nodes.verification import verify_node
from core.workflow.nodes.human_review import human_review_node
from core.workflow.nodes.execution import execute_node
from core.workflow.nodes.summarization import summarize_node

from core.ui_adapter.events import (
    apply_ui_event_to_state,
    make_approve_human_review_event,
    make_run_plan_event,
    make_user_message_event,
)
from core.ui_adapter.snapshot import build_ui_snapshot


def get_field(value, field_name, default=None):
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(field_name, default)

    return getattr(value, field_name, default)


def as_dict(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def apply_updates(state, updates):
    merged = dict(state)
    merged.update(updates)
    return merged


def make_legacy_dataset_profile(*, with_missing=False):
    return {
        "n_rows": 5,
        "n_cols": 3,
        "columns": [
            {
                "name": "GPA",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "missing_count": 1 if with_missing else 0,
                "missing_rate": 0.2 if with_missing else 0.0,
                "n_unique": 4 if with_missing else 5,
            },
            {
                "name": "SATM",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "missing_count": 1 if with_missing else 0,
                "missing_rate": 0.2 if with_missing else 0.0,
                "n_unique": 4 if with_missing else 5,
            },
            {
                "name": "Sex",
                "dtype": "object",
                "semantic_type": "binary_categorical",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 2,
            },
        ],
    }


def make_workspace(tmp_path, *, with_missing=False):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    if with_missing:
        df = pd.DataFrame({
            "GPA": [3.0, 3.2, None, 3.8, 4.0],
            "SATM": [600, None, 650, 680, 700],
            "Sex": ["F", "M", "F", "M", "F"],
        })
    else:
        df = pd.DataFrame({
            "GPA": [3.0, 3.2, 3.5, 3.8, 4.0],
            "SATM": [600, 620, 650, 680, 700],
            "Sex": ["F", "M", "F", "M", "F"],
        })

    data_path = workspace_dir / "working_data.parquet"
    df.to_parquet(data_path)

    return workspace_dir, data_path


def make_base_state(tmp_path, *, with_missing=False):
    workspace_dir, data_path = make_workspace(
        tmp_path,
        with_missing=with_missing,
    )

    return {
        "user_request": "",
        "workspace_dir": str(workspace_dir),
        "current_step": 0,
        "max_steps": 5,

        "dataset_profile": make_legacy_dataset_profile(
            with_missing=with_missing,
        ),

        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": str(data_path),
                "parent_version_id": None,
                "n_rows": 5,
                "n_cols": 3,
            }
        ],
        "data_audit_log": [],
        "active_data_version_id": "raw_v1",

        "pending_plan": None,
        "plan_status": None,
        "plan_execution_status": None,
        "current_plan_step_id": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,

        "repair_attempts": [],
    }


def make_summary_pending_plan():
    return {
        "plan_id": "plan_ui_roundtrip_summary",
        "status": "partially_ready",
        "steps": [
            {
                "step_id": "s1",
                "title": "Compute summary statistics",
                "tool_name": "get_summary_stats",
                "status": "ready",
                "execution_ready": True,
                "execution_status": "not_started",
                "arguments": {},
                "reason": "Summarize the active dataset.",
            }
        ],
        "blocked_or_not_recommended": [],
    }


def make_clean_pending_plan():
    return {
        "plan_id": "plan_ui_roundtrip_clean",
        "status": "partially_ready",
        "steps": [
            {
                "step_id": "s1",
                "title": "Drop rows with missing GPA or SATM",
                "tool_name": "clean_data",
                "status": "ready",
                "execution_ready": True,
                "execution_status": "not_started",
                "arguments": {
                    "action_type": "drop",
                    "strategy": "rows",
                    "columns": ["GPA", "SATM"],
                },
                "reason": "Remove rows with missing values.",
            }
        ],
        "blocked_or_not_recommended": [],
    }


def test_ui_user_message_event_round_trips_to_advisory_snapshot(tmp_path):
    state = make_base_state(tmp_path)

    event = make_user_message_event(
        "I want to do analysis to this dataset, what can I do?"
    )

    updates = apply_ui_event_to_state(state, event)
    state = apply_updates(state, updates)

    assert state["user_request"] == (
        "I want to do analysis to this dataset, what can I do?"
    )

    updates = intent_router_node(state)
    state = apply_updates(state, updates)

    assert state["interaction_intent"] == "advisory"

    updates = advisory_answer_node(state)
    state = apply_updates(state, updates)

    snapshot = build_ui_snapshot(state)

    assert snapshot["assistant_response"]["response_type"] == "advisory"
    assert snapshot["assistant_response"]["content"]

    assert snapshot["runtime"]["has_current_action"] is False
    assert snapshot["analysis"]["analysis_runs"] == []

    json.dumps(snapshot)


def test_ui_run_plan_event_round_trips_to_completed_summary_snapshot(tmp_path):
    state = make_base_state(tmp_path)
    state["pending_plan"] = make_summary_pending_plan()
    state["plan_status"] = "partially_ready"

    event = make_run_plan_event()

    updates = apply_ui_event_to_state(state, event)
    state = apply_updates(state, updates)

    assert state["user_request"] == "run the plan"

    updates = execute_pending_plan_node(state)
    state = apply_updates(state, updates)

    assert state["plan_execution_status"] == "started_step"
    assert get_field(state["current_action"], "tool_name") == "get_summary_stats"

    updates = verify_node(state)
    state = apply_updates(state, updates)

    verification = as_dict(state["current_verification"])
    assert verification["status"] == "allowed"

    updates = execute_node(state)
    state = apply_updates(state, updates)

    execution = as_dict(state["current_execution"])
    assert execution["status"] in {"ok", "warning"}
    assert execution["success"] is True

    updates = summarize_node(state)
    state = apply_updates(state, updates)

    snapshot = build_ui_snapshot(state)

    assert snapshot["plan"]["plan_execution_status"] == "started_step"
    assert snapshot["plan"]["pending_plan"]["steps"][0]["execution_status"] == "completed"

    assert len(snapshot["analysis"]["analysis_runs"]) == 1
    assert snapshot["analysis"]["analysis_runs"][0]["tool_name"] == "get_summary_stats"
    assert snapshot["analysis"]["analysis_runs"][0]["success"] is True

    assert snapshot["runtime"]["has_current_action"] is False
    assert snapshot["audits"]["execution_audit"]["status"] == "ok"

    json.dumps(snapshot)


def test_ui_approve_human_review_event_round_trips_to_clean_data_snapshot(tmp_path):
    state = make_base_state(
        tmp_path,
        with_missing=True,
    )

    state["pending_plan"] = make_clean_pending_plan()
    state["plan_status"] = "partially_ready"

    updates = apply_ui_event_to_state(
        state,
        make_run_plan_event(),
    )
    state = apply_updates(state, updates)

    updates = execute_pending_plan_node(state)
    state = apply_updates(state, updates)

    assert get_field(state["current_action"], "tool_name") == "clean_data"

    updates = verify_node(state)
    state = apply_updates(state, updates)

    verification = as_dict(state["current_verification"])

    assert verification["status"] == "needs_review"

    review_snapshot = build_ui_snapshot(state)

    assert review_snapshot["human_review"]["required"] is True
    assert review_snapshot["human_review"]["action"]["tool_name"] == "clean_data"

    action_hash = review_snapshot["human_review"]["action_hash"]

    updates = apply_ui_event_to_state(
        state,
        make_approve_human_review_event(
            action_hash=action_hash,
        ),
    )
    state = apply_updates(state, updates)

    updates = human_review_node(state)
    state = apply_updates(state, updates)

    verification = as_dict(state["current_verification"])
    assert verification["status"] == "allowed"

    updates = execute_node(state)
    state = apply_updates(state, updates)

    execution = as_dict(state["current_execution"])

    assert execution["status"] in {"ok", "warning"}
    assert execution["success"] is True

    updates = summarize_node(state)
    state = apply_updates(state, updates)

    snapshot = build_ui_snapshot(state)

    assert snapshot["human_review"]["required"] is False
    assert snapshot["data"]["active_data_version_id"] != "raw_v1"

    assert len(snapshot["data"]["data_versions"]) == 2
    assert len(snapshot["analysis"]["analysis_runs"]) == 1

    run = snapshot["analysis"]["analysis_runs"][0]

    assert run["tool_name"] == "clean_data"
    assert run["success"] is True
    assert run["data_version_id"] == snapshot["data"]["active_data_version_id"]

    assert snapshot["runtime"]["has_current_action"] is False
    assert snapshot["audits"]["execution_audit"]["status"] == "ok"

    json.dumps(snapshot)