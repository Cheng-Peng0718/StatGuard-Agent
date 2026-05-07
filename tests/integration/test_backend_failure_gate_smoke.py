from core.graph import (
    deliverable_gate_node,
    execute_pending_plan_node,
    route_after_deliverable_gate,
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
    }


def make_pending_plan():
    return {
        "plan_id": "plan_failure_gate_smoke_1",
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
            },
        ],
        "blocked_or_not_recommended": [],
    }


def make_state():
    return {
        "user_request": "run the plan",
        "workspace_dir": "workspaces/test",
        "current_step": 0,
        "max_steps": 5,

        # verify_node still depends on this legacy profile field.
        "dataset_profile": make_legacy_dataset_profile(),

        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "workspaces/test/data_versions/raw_v1.parquet",
                "parent_version_id": None,
            }
        ],
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

        # This is the deliverable contract that must NOT be satisfied
        # when the required tool fails.
        "task_contract": {
            "required_tools": ["run_multiple_regression"],
        },
    }


def test_failed_required_tool_blocks_deliverable_gate_and_prevents_final_response_route():
    state = make_state()

    # 1. pending_plan -> current_action
    exec_plan_updates = execute_pending_plan_node(state)
    state = apply_updates(state, exec_plan_updates)

    assert state["plan_execution_status"] == "started_step"
    assert state["current_plan_step_id"] == "s1"
    assert state["action_origin"] == "pending_plan"

    action = state["current_action"]

    assert get_field(action, "action_type") == "tool_call"
    assert get_field(action, "tool_name") == "run_multiple_regression"
    assert get_field(action, "arguments") == {
        "target_col": "GPA",
        "feature_cols": ["SATM"],
    }

    # 2. verify
    verify_updates = verify_node(state)
    state = apply_updates(state, verify_updates)

    verification = as_dict(state["current_verification"])

    assert verification["status"] == "allowed"

    # 3. Synthetic failed execution.
    # S14H deliberately avoids real execute_node; this isolates failure-gate behavior.
    state["current_execution"] = {
        "execution_id": "exec_regression_failed_gate_smoke",
        "status": "failed",
        "success": False,
        "error_code": "INTERNAL_PLUGIN_ERROR",
        "message": "Regression plugin crashed during failure-gate smoke test.",
        "artifacts": [],
        "payload": {},
    }

    # 4. summarize failed execution
    summarize_updates = summarize_node(state)
    state = apply_updates(state, summarize_updates)

    assert len(state["observations"]) == 1
    assert len(state["analysis_runs"]) == 1

    observation = state["observations"][0]
    analysis_run = state["analysis_runs"][0]

    assert observation["tool_name"] == "run_multiple_regression"
    assert observation["status"] == "failed"
    assert observation["success"] is False
    assert observation["error_code"] == "INTERNAL_PLUGIN_ERROR"

    assert analysis_run["observation_id"] == observation["observation_id"]
    assert analysis_run["tool_name"] == "run_multiple_regression"
    assert analysis_run["status"] == "failed"
    assert analysis_run["success"] is False
    assert analysis_run["error_code"] == "INTERNAL_PLUGIN_ERROR"

    assert state["execution_audit"]["status"] == "ok"

    # Repair state should be recorded, but not executed.
    assert state["repair_decision"]["status"] == "terminal"
    assert state["repair_decision"]["tool_name"] == "run_multiple_regression"
    assert state["repair_proposal"]["proposal_type"] == "no_op"
    assert state.get("repair_attempts", []) == []

    # Pending plan step should be failed.
    step = state["pending_plan"]["steps"][0]

    assert step["execution_status"] == "failed"
    assert step["last_execution_id"] == "exec_regression_failed_gate_smoke"

    # 5. DeliverableGate should block final answer.
    gate_updates = deliverable_gate_node(state)
    state = apply_updates(state, gate_updates)

    deliverable_check = state["deliverable_check"]

    assert deliverable_check["status"] == "needs_more_work"
    assert "tool:run_multiple_regression" in deliverable_check["missing"]
    assert "tool_failed:run_multiple_regression" in deliverable_check["blocked"]

    # 6. Routing after DeliverableGate must not go to final_response.
    route = route_after_deliverable_gate(state)

    assert route != "final_response"
    assert route in {"build_context", "end"}