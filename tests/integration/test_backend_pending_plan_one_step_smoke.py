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
            {
                "name": "Sex",
                "dtype": "object",
                "semantic_type": "binary_categorical",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 2,
            },
            {
                "name": "Section",
                "dtype": "object",
                "semantic_type": "nominal_categorical",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 4,
            },
        ],
    }


def make_pending_plan():
    return {
        "plan_id": "plan_smoke_1",
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
            },
            {
                "step_id": "s2",
                "title": "Run regression later",
                "tool_name": "run_multiple_regression",
                "status": "needs_user_choice",
                "execution_ready": False,
                "execution_status": "not_started",
                "arguments": {},
                "reason": "Requires user-selected outcome and predictors.",
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


def test_pending_plan_ready_step_flows_to_action_verify_and_summarize_success():
    state = make_state()

    exec_plan_updates = execute_pending_plan_node(state)
    state = apply_updates(state, exec_plan_updates)

    assert state["plan_execution_status"] == "started_step"
    assert state["current_plan_step_id"] == "s1"
    assert state["action_origin"] == "pending_plan"

    action = state["current_action"]

    assert get_field(action, "action_type") == "tool_call"
    assert get_field(action, "tool_name") == "get_summary_stats"
    assert get_field(action, "arguments") == {}

    verify_updates = verify_node(state)
    state = apply_updates(state, verify_updates)

    verification = as_dict(state["current_verification"])

    assert verification["status"] == "allowed"

    # S14C: do not call real execute_node yet.
    # Inject a synthetic successful execution so we test graph state flow only.
    state["current_execution"] = {
        "execution_id": "exec_summary_smoke",
        "status": "ok",
        "success": True,
        "error_code": None,
        "message": "Summary statistics completed.",
        "artifacts": [],
        "payload": {
            "n_rows": 226,
            "n_cols": 4,
        },
    }

    summarize_updates = summarize_node(state)
    state = apply_updates(state, summarize_updates)

    assert len(state["observations"]) == 1
    assert len(state["analysis_runs"]) == 1

    observation = state["observations"][0]
    analysis_run = state["analysis_runs"][0]

    assert observation["tool_name"] == "get_summary_stats"
    assert observation["status"] == "ok"
    assert observation["success"] is True

    assert analysis_run["observation_id"] == observation["observation_id"]
    assert analysis_run["tool_name"] == "get_summary_stats"
    assert analysis_run["status"] == "ok"
    assert analysis_run["success"] is True

    assert state["execution_audit"]["status"] == "ok"

    updated_plan = state["pending_plan"]
    step = updated_plan["steps"][0]

    assert step["step_id"] == "s1"
    assert step["execution_status"] == "completed"
    assert step["last_execution_id"] == "exec_summary_smoke"
    assert step["last_execution_message"] == "Summary statistics completed."

    assert state["current_plan_step_id"] is None
    assert state["action_origin"] is None
    assert state["current_action"] is None
    assert state["current_execution"] is None
    assert state["current_verification"] is None