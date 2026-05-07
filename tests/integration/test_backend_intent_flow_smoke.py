from core.graph import (
    advisory_answer_node,
    execute_pending_plan_node,
    intent_router_node,
    plan_only_node,
)


def make_column(
    *,
    name,
    semantic_type,
    raw_dtype,
    measurement_scale,
    n_unique,
    n_missing=0,
    missing_rate=0.0,
    n_rows=226,
):
    return {
        "name": name,
        "semantic_type": semantic_type,

        # DatasetProfileV2 required fields
        "raw_dtype": raw_dtype,
        "measurement_scale": measurement_scale,
        "n_missing": n_missing,
        "unique_rate": n_unique / n_rows,

        # extra compatibility fields used by some older helpers/tests
        "dtype": raw_dtype,
        "missing_count": n_missing,
        "missing_rate": missing_rate,
        "n_unique": n_unique,
    }


def make_minimal_profile():
    return {
        "data_version_id": "raw_v1",
        "n_rows": 226,
        "n_cols": 4,
        "columns": {
            "GPA": make_column(
                name="GPA",
                semantic_type="continuous_numeric",
                raw_dtype="float64",
                measurement_scale="continuous",
                n_unique=80,
            ),
            "SATM": make_column(
                name="SATM",
                semantic_type="continuous_numeric",
                raw_dtype="float64",
                measurement_scale="continuous",
                n_unique=90,
            ),
            "Sex": make_column(
                name="Sex",
                semantic_type="binary_categorical",
                raw_dtype="object",
                measurement_scale="nominal",
                n_unique=2,
            ),
            "Section": make_column(
                name="Section",
                semantic_type="nominal_categorical",
                raw_dtype="object",
                measurement_scale="nominal",
                n_unique=4,
            ),
        },
    }


def make_minimal_capability_map():
    return {
        "data_version_id": "raw_v1",
        "capabilities": [
            {
                "tool_name": "get_summary_stats",
                "display_name": "Summary Statistics",
                "status": "ready",
                "method_family": "eda",
                "reason": "Can run without user-selected variables.",
                "required_roles": [],
                "optional_roles": [],
            },
            {
                "tool_name": "missingness_report",
                "display_name": "Missingness Report",
                "status": "ready",
                "method_family": "eda",
                "reason": "Can run without user-selected variables.",
                "required_roles": [],
                "optional_roles": ["columns"],
            },
            {
                "tool_name": "get_correlation_matrix",
                "display_name": "Correlation Matrix",
                "status": "ready",
                "method_family": "association_screening",
                "reason": "Can run using eligible numeric columns.",
                "required_roles": [],
                "optional_roles": ["columns"],
            },
            {
                "tool_name": "run_multiple_regression",
                "display_name": "Linear Model",
                "status": "needs_user_choice",
                "method_family": "regression",
                "reason": "Requires user-selected outcome and predictors.",
                "required_roles": ["target_col", "feature_cols"],
                "optional_roles": [],
            },
            {
                "tool_name": "run_anova",
                "display_name": "One-way ANOVA",
                "status": "needs_user_choice",
                "method_family": "group_comparison",
                "reason": "Requires numeric outcome and grouping column.",
                "required_roles": ["target_col", "group_col"],
                "optional_roles": [],
            },
        ],
    }


def make_state(user_request):
    return {
        "user_request": user_request,
        "workspace_dir": "workspaces/test",
        "current_step": 0,
        "max_steps": 5,
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "workspaces/test/data_versions/raw_v1.parquet",
            }
        ],
        "active_data_version_id": "raw_v1",
        "dataset_profile_v2": make_minimal_profile(),
        "dataset_profile": make_minimal_profile(),
        "capability_map": make_minimal_capability_map(),
        "pending_plan": None,
        "plan_status": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,
    }


def test_backend_intent_flow_advisory_does_not_create_action():
    state = make_state(
        "I want to do analysis to this dataset, what can I do?"
    )

    intent_updates = intent_router_node(state)
    state.update(intent_updates)

    assert state["interaction_intent"] == "advisory"

    response_updates = advisory_answer_node(state)

    assert "assistant_response" in response_updates
    assert response_updates["assistant_response"]["response_type"] == "advisory"
    assert response_updates["assistant_response"]["content"]

    assert response_updates["current_action"] is None
    assert response_updates["current_execution"] is None
    assert response_updates["current_verification"] is None


def test_backend_intent_flow_plan_only_creates_plan_but_no_action():
    state = make_state(
        "could you make up a plan and tell me?"
    )

    intent_updates = intent_router_node(state)
    state.update(intent_updates)

    assert state["interaction_intent"] == "plan_only"

    plan_updates = plan_only_node(state)

    assert "assistant_response" in plan_updates
    assert plan_updates["assistant_response"]["response_type"] == "plan"
    assert plan_updates["assistant_response"]["content"]

    assert plan_updates["pending_plan"] is not None
    assert plan_updates["plan_status"] in {
        "draft",
        "verified",
        "partially_ready",
        "blocked",
    }

    assert plan_updates["current_action"] is None
    assert plan_updates["current_execution"] is None
    assert plan_updates["current_verification"] is None


def test_backend_intent_flow_execute_plan_without_pending_plan_returns_status_response():
    state = make_state("run the plan")
    state["pending_plan"] = None

    intent_updates = intent_router_node(state)
    state.update(intent_updates)

    assert state["interaction_intent"] == "execute_plan"

    exec_updates = execute_pending_plan_node(state)

    assert "assistant_response" in exec_updates
    assert exec_updates["assistant_response"]["response_type"] == "plan_execution_status"
    assert exec_updates["assistant_response"]["content"]

    assert exec_updates["plan_execution_status"] == "no_pending_plan"
    assert exec_updates["current_action"] is None
    assert exec_updates["current_execution"] is None
    assert exec_updates["current_verification"] is None