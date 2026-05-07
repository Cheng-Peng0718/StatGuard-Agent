import json

import pandas as pd

from core.audit.state_serialization import (
    audit_state_serialization,
    make_checkpoint_safe_state,
)
from core.graph import (
    deliverable_gate_node,
    execute_node,
    execute_pending_plan_node,
    final_response_node,
    summarize_node,
    verify_node,
)


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


def make_legacy_dataset_profile():
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


def make_pending_plan():
    return {
        "plan_id": "plan_checkpoint_smoke_1",
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


def make_state(tmp_path):
    workspace_dir, data_path = make_workspace_with_data(tmp_path)

    return {
        "user_request": "run the plan",
        "workspace_dir": str(workspace_dir),
        "current_step": 0,
        "max_steps": 5,

        # verify_node still depends on this legacy profile field.
        "dataset_profile": make_legacy_dataset_profile(),

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

        "pending_plan": make_pending_plan(),
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


def test_full_backend_success_state_can_be_made_checkpoint_safe(tmp_path):
    state = make_state(tmp_path)

    # 1. pending plan -> action
    updates = execute_pending_plan_node(state)
    state = apply_updates(state, updates)

    assert get_field(state["current_action"], "tool_name") == "get_summary_stats"

    # 2. verify
    updates = verify_node(state)
    state = apply_updates(state, updates)

    verification = as_dict(state["current_verification"])
    assert verification["status"] == "allowed"

    # 3. real execute
    updates = execute_node(state)
    state = apply_updates(state, updates)

    execution = as_dict(state["current_execution"])
    assert execution["status"] in {"ok", "warning"}
    assert execution["success"] is True

    # 4. summarize
    updates = summarize_node(state)
    state = apply_updates(state, updates)

    assert len(state["observations"]) == 1
    assert len(state["analysis_runs"]) == 1
    assert "execution_audit" in state
    assert "state_serialization_audit" in state

    # 5. deliverable gate
    updates = deliverable_gate_node(state)
    state = apply_updates(state, updates)

    assert state["deliverable_check"]["status"] == "ok"

    # 6. final response
    state["final_answer"] = "Summary statistics completed successfully."

    updates = final_response_node(state)
    state = apply_updates(state, updates)

    assert state["assistant_response"]["response_type"] == "final_answer"

    # 7. checkpoint-safe boundary
    audit = audit_state_serialization(state)
    safe_state = make_checkpoint_safe_state(state)

    assert isinstance(safe_state, dict)

    # The safe state must be JSON serializable.
    json.dumps(safe_state)

    # It should preserve important backend outputs.
    assert safe_state["assistant_response"]["response_type"] == "final_answer"
    assert safe_state["assistant_response"]["content"] == "Summary statistics completed successfully."

    assert len(safe_state["observations"]) == 1
    assert len(safe_state["analysis_runs"]) == 1
    assert safe_state["execution_audit"]["status"] == "ok"
    assert safe_state["deliverable_check"]["status"] == "ok"

    # Runtime-only fields should already be cleared by summarize/final response.
    assert safe_state["current_action"] is None
    assert safe_state["current_execution"] is None
    assert safe_state["current_verification"] is None

    # Current backend may still produce warnings because some nodes use objects
    # internally before normalization. But checkpoint-safe state must serialize.
    assert audit.status in {"ok", "warning"}