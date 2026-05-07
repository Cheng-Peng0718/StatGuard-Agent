import pandas as pd

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
        "plan_id": "plan_final_smoke_1",
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

        # DeliverableGate contract for this smoke flow.
        "task_contract": {
            "required_tools": ["get_summary_stats"],
        },
    }


def test_real_execute_to_deliverable_gate_to_final_response_smoke(tmp_path):
    state = make_state(tmp_path)

    # 1. Pending plan -> current_action
    exec_plan_updates = execute_pending_plan_node(state)
    state = apply_updates(state, exec_plan_updates)

    assert state["plan_execution_status"] == "started_step"
    assert state["current_plan_step_id"] == "s1"
    assert state["action_origin"] == "pending_plan"

    action = state["current_action"]

    assert get_field(action, "action_type") == "tool_call"
    assert get_field(action, "tool_name") == "get_summary_stats"

    # 2. Verify
    verify_updates = verify_node(state)
    state = apply_updates(state, verify_updates)

    verification = as_dict(state["current_verification"])
    assert verification["status"] == "allowed"

    # 3. Real execute_node
    execute_updates = execute_node(state)
    state = apply_updates(state, execute_updates)

    execution = as_dict(state["current_execution"])

    assert execution["status"] in {"ok", "warning"}
    assert execution["success"] is True
    assert execution.get("error_code") in {None, ""}

    # 4. Summarize
    summarize_updates = summarize_node(state)
    state = apply_updates(state, summarize_updates)

    assert len(state["observations"]) == 1
    assert len(state["analysis_runs"]) == 1

    observation = state["observations"][0]
    analysis_run = state["analysis_runs"][0]

    assert observation["tool_name"] == "get_summary_stats"
    assert observation["status"] in {"ok", "warning"}
    assert observation["success"] is True

    assert analysis_run["observation_id"] == observation["observation_id"]
    assert analysis_run["tool_name"] == "get_summary_stats"
    assert analysis_run["status"] in {"ok", "warning"}
    assert analysis_run["success"] is True

    assert state["execution_audit"]["status"] == "ok"

    updated_plan = state["pending_plan"]
    step = updated_plan["steps"][0]

    assert step["execution_status"] == "completed"
    assert step["last_execution_id"]

    # 5. DeliverableGate
    gate_updates = deliverable_gate_node(state)
    state = apply_updates(state, gate_updates)

    assert state["deliverable_check"]["status"] == "ok"
    assert "tool:get_summary_stats" in state["deliverable_check"]["satisfied"]
    assert state["deliverable_check"]["missing"] == []
    assert state["deliverable_check"]["blocked"] == []

    # 6. Final response envelope
    state["final_answer"] = (
        "Summary statistics were computed successfully for the active dataset."
    )

    final_updates = final_response_node(state)

    assert "assistant_response" in final_updates
    assert final_updates["assistant_response"]["response_type"] == "final_answer"
    assert (
        final_updates["assistant_response"]["content"]
        == "Summary statistics were computed successfully for the active dataset."
    )

    assert final_updates["current_action"] is None
    assert final_updates["current_execution"] is None
    assert final_updates["current_verification"] is None