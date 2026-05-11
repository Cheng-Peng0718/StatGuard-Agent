from core.workflow.nodes.plan_execution import execute_pending_plan_node
from core.workflow.nodes.verification import verify_node
from core.workflow.nodes.summarization import summarize_node



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
        "n_rows": 226,
        "n_cols": 4,
        "columns": [
            {
                "name": "GPA",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 80,
            },
            {
                "name": "SATM",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 90,
            },
        ],
    }


def make_pending_plan():
    return {
        "plan_id": "plan_failed_smoke_1",
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

        # verify_node still depends on the legacy profile field.
        "dataset_profile": make_legacy_dataset_profile(),

        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "workspaces/test/data_versions/raw_v1.parquet",
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
    }


def test_pending_plan_ready_step_flows_to_failed_analysis_run_and_repair_state():
    state = make_state()

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

    verify_updates = verify_node(state)
    state = apply_updates(state, verify_updates)

    verification = as_dict(state["current_verification"])
    assert verification["status"] == "allowed"

    # S14D: do not call real execute_node yet.
    # Inject synthetic failed execution to test graph state flow only.
    state["current_execution"] = {
        "execution_id": "exec_reg_failed_smoke",
        "status": "failed",
        "success": False,
        "error_code": "INTERNAL_PLUGIN_ERROR",
        "message": "Regression plugin crashed during smoke test.",
        "artifacts": [],
        "payload": {},
    }

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

    assert "repair_decision" in state
    assert state["repair_decision"]["tool_name"] == "run_multiple_regression"
    assert state["repair_decision"]["error_code"] == "INTERNAL_PLUGIN_ERROR"
    assert state["repair_decision"]["status"] == "terminal"

    assert "repair_proposal" in state
    assert state["repair_proposal"]["proposal_type"] == "no_op"
    assert state["repair_proposal"]["source_tool_name"] == "run_multiple_regression"
    assert state["repair_proposal"]["source_action_id"] == get_field(action, "action_id")

    # Terminal failure should not create a repair attempt.
    assert state.get("repair_attempts", []) == []

    updated_plan = state["pending_plan"]
    step = updated_plan["steps"][0]

    assert step["step_id"] == "s1"
    assert step["execution_status"] == "failed"
    assert step["last_execution_id"] == "exec_reg_failed_smoke"
    assert step["last_execution_message"] == "Regression plugin crashed during smoke test."

    assert state["current_plan_step_id"] is None
    assert state["action_origin"] is None
    assert state["current_action"] is None
    assert state["current_execution"] is None
    assert state["current_verification"] is None