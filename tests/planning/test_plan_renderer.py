from core.planning.schemas import PlanProposal, PlanStep
from core.planning.renderer import render_plan_for_user


def test_render_plan_states_no_tools_executed():
    plan = PlanProposal(
        plan_id="plan_test",
        user_request="make a plan",
        data_version_id="raw_v1",
        status="partially_ready",
        steps=[
            PlanStep(
                step_id="step_1",
                title="Generic Model",
                tool_name="generic_model",
                status="needs_user_choice",
                execution_ready=False,
                required_user_choices=["outcome", "predictors"],
                candidate_variables={
                    "outcome": ["GPA"],
                    "predictors": ["SATM", "SATV"],
                },
            )
        ],
    )

    text = render_plan_for_user(plan)

    assert "I have not run anything yet" in text
    assert "No tools have been executed" in text
    assert "Needs your choice" in text
    assert "outcome" in text
    assert "SATM" in text