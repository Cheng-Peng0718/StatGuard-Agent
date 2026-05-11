from core.workflow.nodes.plan_execution import execute_pending_plan_node


def make_state():
    return {
        "active_data_version_id": "raw_v1",
        "dataset_profile": {
            "n_rows": 5,
            "n_cols": 3,
            "columns": [
                {
                    "name": "GPA",
                    "semantic_type": "continuous_numeric",
                    "missing_count": 1,
                },
                {
                    "name": "SATM",
                    "semantic_type": "continuous_numeric",
                    "missing_count": 1,
                },
                {
                    "name": "Sex",
                    "semantic_type": "binary_categorical",
                    "missing_count": 0,
                },
            ],
        },
        "dataset_profile_v2": {
            "dataset_name": "uploaded_dataset",
            "data_version_id": "raw_v1",
            "n_rows": 5,
            "n_cols": 3,
            "columns": {
                "GPA": {
                    "name": "GPA",
                    "semantic_type": "continuous_numeric",
                    "raw_dtype": "float64",
                    "measurement_scale": "continuous",
                    "n_missing": 1,
                    "missing_rate": 0.2,
                    "n_unique": 4,
                    "unique_rate": 0.8,
                },
                "SATM": {
                    "name": "SATM",
                    "semantic_type": "continuous_numeric",
                    "raw_dtype": "float64",
                    "measurement_scale": "continuous",
                    "n_missing": 1,
                    "missing_rate": 0.2,
                    "n_unique": 4,
                    "unique_rate": 0.8,
                },
                "Sex": {
                    "name": "Sex",
                    "semantic_type": "binary_categorical",
                    "raw_dtype": "object",
                    "measurement_scale": "nominal",
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "n_unique": 2,
                    "unique_rate": 0.4,
                },
            },
        },
        "pending_plan": {
            "plan_id": "plan_block_modeling",
            "status": "partially_executed",
            "steps": [
                {
                    "step_id": "step_regression",
                    "tool_name": "run_multiple_regression",
                    "status": "ready",
                    "execution_ready": True,
                    "execution_status": "not_started",
                    "arguments": {
                        "target_col": "GPA",
                        "feature_cols": ["SATM"],
                    },
                    "variables": {
                        "target_col": "GPA",
                        "feature_cols": ["SATM"],
                    },
                    "required_user_choices": [],
                },
                {
                    "step_id": "step_clean",
                    "tool_name": "clean_data",
                    "status": "needs_user_choice",
                    "execution_ready": False,
                    "execution_status": "not_started",
                    "arguments": {},
                    "variables": {},
                    "required_user_choices": ["action_type", "strategy", "columns"],
                },
            ],
        },
        "plan_status": "partially_executed",
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
    }


def test_modeling_step_is_blocked_until_clean_data_completes():
    result = execute_pending_plan_node(make_state())

    assert result["plan_execution_status"] == "blocked_pending_data_cleaning"
    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None

    response = result["assistant_response"]

    assert response["response_type"] == "plan_execution_status"
    assert response["metadata"]["reason"] == "modeling_blocked_by_pending_cleaning"
    assert response["metadata"]["tool_name"] == "run_multiple_regression"