from core.dataset_intelligence.schemas import (
    AnalysisCapability,
    CapabilityMap,
)
from core.planning.planner import build_plan_from_capability_map


def test_build_plan_from_capability_map_does_not_execute():
    capability_map = CapabilityMap(
        data_version_id="raw_v1",
        capabilities=[
            AnalysisCapability(
                tool_name="generic_ready_tool",
                display_name="Generic Ready Tool",
                method_family="general",
                status="ready",
                reason="This tool is ready.",
            ),
            AnalysisCapability(
                tool_name="generic_model",
                display_name="Generic Model",
                method_family="modeling",
                status="needs_user_choice",
                reason="User must choose variables.",
                candidate_variables={
                    "outcome": ["y"],
                    "predictors": ["x"],
                },
                required_user_choices=["outcome", "predictors"],
            ),
        ],
    )

    plan = build_plan_from_capability_map(
        user_request="make a plan",
        capability_map=capability_map,
    )

    assert plan.mode == "plan_only"
    assert plan.data_version_id == "raw_v1"
    assert len(plan.steps) == 2

    ready_step = plan.steps[0]
    needs_choice_step = plan.steps[1]

    assert ready_step.execution_ready is True
    assert needs_choice_step.execution_ready is False
    assert needs_choice_step.required_user_choices == ["outcome", "predictors"]