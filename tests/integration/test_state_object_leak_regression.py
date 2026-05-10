import json

import pandas as pd

from core.audit.state_serialization import audit_state_serialization

from core.workflow.nodes.execution import execute_node

from core.workflow.nodes.finalization import final_response_node

from core.workflow.nodes.summarization import summarize_node

from core.workflow.nodes.verification import verify_node

from core.workflow.nodes.plan_execution import execute_pending_plan_node

from core.workflow.nodes.finalization import deliverable_gate_node


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


def assert_json_safe_value(value):
    json.dumps(value)

    audit = audit_state_serialization({"value": value})

    assert audit.status == "ok", [
        issue.model_dump() if hasattr(issue, "model_dump") else issue
        for issue in audit.issues
    ]


def assert_state_fields_json_safe(state, fields):
    for field in fields:
        if field in state:
            assert_json_safe_value(state[field])


def make_state_dataset_profile():
    return {
        "n_rows": 5,
        "n_cols": 3,
        "columns": [
            {
                "name": "GPA",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 5,
            },
            {
                "name": "SATM",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 5,
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


def make_workspace_with_data(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    df = pd.DataFrame({
        "GPA": [3.0, 3.2, 3.5, 3.8, 4.0],
        "SATM": [600, 620, 650, 680, 700],
        "Sex": ["F", "M", "F", "M", "F"],
    })

    data_path = workspace_dir / "working_data.parquet"
    df.to_parquet(data_path)

    return workspace_dir, data_path


def make_success_pending_plan():
    return {
        "plan_id": "plan_state_leak_success",
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


def make_success_state(tmp_path):
    workspace_dir, data_path = make_workspace_with_data(tmp_path)

    return {
        "user_request": "run the plan",
        "workspace_dir": str(workspace_dir),
        "current_step": 0,
        "max_steps": 5,

        "dataset_profile": make_state_dataset_profile(),

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

        "pending_plan": make_success_pending_plan(),
        "plan_status": "partially_ready",
        "current_plan_step_id": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,

        "repair_attempts": [],

        "task_contract": {
            "required_tools": ["get_summary_stats"],
        },
    }


def test_success_flow_stable_state_fields_do_not_leak_objects(tmp_path):
    state = make_success_state(tmp_path)

    updates = execute_pending_plan_node(state)
    state = apply_updates(state, updates)

    # During runtime, current_action may be an object. That is allowed.
    assert state["current_action"] is not None

    updates = verify_node(state)
    state = apply_updates(state, updates)

    updates = execute_node(state)
    state = apply_updates(state, updates)

    updates = summarize_node(state)
    state = apply_updates(state, updates)

    updates = deliverable_gate_node(state)
    state = apply_updates(state, updates)

    state["final_answer"] = "Summary statistics completed successfully."

    updates = final_response_node(state)
    state = apply_updates(state, updates)

    assert state["current_action"] is None
    assert state["current_execution"] is None
    assert state["current_verification"] is None

    assert_state_fields_json_safe(
        state,
        fields=[
            "assistant_response",
            "pending_plan",
            "observations",
            "analysis_runs",
            "data_versions",
            "data_audit_log",
            "execution_audit",
            "state_serialization_audit",
            "deliverable_check",
            "repair_attempts",
        ],
    )


def make_failure_pending_plan():
    return {
        "plan_id": "plan_state_leak_failure",
        "status": "partially_ready",
        "steps": [
            {
                "step_id": "s1",
                "title": "Run regression",
                "tool_name": "run_multiple_regression",
                "status": "ready",
                "execution_ready": True,
                "execution_status": "not_started",
                "arguments": {
                    "target_col": "GPA",
                    "feature_cols": ["SATM"],
                },
                "reason": "Fit GPA on SATM.",
            }
        ],
        "blocked_or_not_recommended": [],
    }


def make_failure_state():
    return {
        "user_request": "run the plan",
        "workspace_dir": "workspaces/test",
        "current_step": 0,
        "max_steps": 5,

        "dataset_profile": {
            "n_rows": 5,
            "n_cols": 2,
            "columns": [
                {
                    "name": "GPA",
                    "dtype": "float64",
                    "semantic_type": "continuous_numeric",
                    "missing_count": 0,
                    "missing_rate": 0.0,
                    "n_unique": 5,
                },
                {
                    "name": "SATM",
                    "dtype": "float64",
                    "semantic_type": "continuous_numeric",
                    "missing_count": 0,
                    "missing_rate": 0.0,
                    "n_unique": 5,
                },
            ],
        },

        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "workspaces/test/data_versions/raw_v1.parquet",
                "parent_version_id": None,
            }
        ],
        "data_audit_log": [],
        "active_data_version_id": "raw_v1",

        "pending_plan": make_failure_pending_plan(),
        "plan_status": "partially_ready",
        "current_plan_step_id": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,

        "repair_attempts": [],

        "task_contract": {
            "required_tools": ["run_multiple_regression"],
        },
    }


def test_failure_flow_repair_state_fields_do_not_leak_objects():
    state = make_failure_state()

    updates = execute_pending_plan_node(state)
    state = apply_updates(state, updates)

    action = state["current_action"]
    assert get_field(action, "tool_name") == "run_multiple_regression"

    updates = verify_node(state)
    state = apply_updates(state, updates)

    verification = as_dict(state["current_verification"])
    assert verification["status"] == "allowed"

    # Synthetic failure to test repair state serialization without real execute_node.
    state["current_execution"] = {
        "execution_id": "exec_state_leak_failure",
        "status": "failed",
        "success": False,
        "error_code": "INTERNAL_PLUGIN_ERROR",
        "message": "Regression plugin crashed.",
        "artifacts": [],
        "payload": {},
    }

    updates = summarize_node(state)
    state = apply_updates(state, updates)

    updates = deliverable_gate_node(state)
    state = apply_updates(state, updates)

    assert state["current_action"] is None
    assert state["current_execution"] is None
    assert state["current_verification"] is None

    assert state["repair_decision"]["status"] == "terminal"
    assert state["repair_proposal"]["proposal_type"] == "no_op"

    assert_state_fields_json_safe(
        state,
        fields=[
            "pending_plan",
            "observations",
            "analysis_runs",
            "data_versions",
            "data_audit_log",
            "execution_audit",
            "state_serialization_audit",
            "deliverable_check",
            "repair_decision",
            "repair_proposal",
            "repair_attempts",
        ],
    )