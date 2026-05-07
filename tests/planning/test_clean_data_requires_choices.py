from core.dataset_intelligence.schemas import DatasetProfileV2
from core.planning.schemas import PlanStep
from core.planning.verifier import verify_plan_step


def make_profile():
    return DatasetProfileV2.model_validate({
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
    })


def test_clean_data_without_arguments_needs_user_choices():
    step = PlanStep(
        step_id="step_clean",
        title="Clean Data",
        tool_name="clean_data",
        method_family="data_preparation",
        status="ready",
        execution_ready=True,
        purpose="Clean missing values.",
        rationale="Clean missing values.",
        variables={},
        arguments={},
        candidate_variables={},
        required_user_choices=[],
        warnings=[],
        requires_confirmation=True,
        mutates_data=True,
    )

    verified = verify_plan_step(step, make_profile())

    assert verified.status == "needs_user_choice"
    assert verified.execution_ready is False
    assert "action_type" in verified.required_user_choices
    assert "strategy" in verified.required_user_choices
    assert "columns" in verified.required_user_choices


def test_clean_data_with_arguments_is_ready_but_requires_confirmation():
    step = PlanStep(
        step_id="step_clean",
        title="Clean Data",
        tool_name="clean_data",
        method_family="data_preparation",
        status="needs_user_choice",
        execution_ready=False,
        purpose="Clean missing values.",
        rationale="Clean missing values.",
        variables={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA", "SATM"],
        },
        arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA", "SATM"],
        },
        candidate_variables={},
        required_user_choices=["action_type", "strategy", "columns"],
        warnings=[],
        requires_confirmation=True,
        mutates_data=True,
    )

    verified = verify_plan_step(step, make_profile())

    assert verified.status == "ready"
    assert verified.execution_ready is True
    assert verified.required_user_choices == []
    assert verified.requires_confirmation is True