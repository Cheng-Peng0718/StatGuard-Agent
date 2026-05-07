from core.dataset_intelligence.schemas import DatasetProfileV2
from core.planning.schemas import PlanProposal, PlanStep
from core.planning.verifier import verify_plan


def make_profile_with_missing():
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


def make_step(tool_name, *, status="ready", execution_ready=True):
    return PlanStep(
        step_id=f"step_{tool_name}",
        title=tool_name,
        tool_name=tool_name,
        method_family="test",
        status=status,
        execution_ready=execution_ready,
        purpose="test",
        rationale="test",
        variables={},
        arguments={},
        candidate_variables={},
        required_user_choices=[],
        warnings=[],
    )


def test_clean_data_is_reordered_before_modeling_when_missingness_exists():
    plan = PlanProposal(
        plan_id="plan_missing_order",
        user_request="make a plan",
        data_version_id="raw_v1",
        mode="plan_only",
        status="partially_ready",
        summary="test",
        assumptions=[],
        steps=[
            make_step("get_summary_stats"),
            make_step("run_multiple_regression", status="needs_user_choice", execution_ready=False),
            make_step("run_anova", status="needs_user_choice", execution_ready=False),
            make_step("clean_data", status="needs_user_choice", execution_ready=False),
        ],
        blocked_or_not_recommended=[],
    )

    verified = verify_plan(plan, make_profile_with_missing())

    tool_order = [
        step.tool_name
        for step in verified.steps
    ]

    assert tool_order.index("clean_data") < tool_order.index("run_multiple_regression")
    assert tool_order.index("clean_data") < tool_order.index("run_anova")