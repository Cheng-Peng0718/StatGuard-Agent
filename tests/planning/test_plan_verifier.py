import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    VariableRoleSpec,
)
from core.dataset_intelligence.profiler import profile_dataframe
from core.planning.schemas import PlanStep, PlanProposal
from core.planning.verifier import verify_plan_step, verify_plan
from core.analysis_tool_plugins.registry import register_plugin, PLUGIN_REGISTRY


def dummy_execute(context):
    return {"status": "ok", "details": {}}


def test_verify_plan_step_needs_user_choice_when_required_variables_missing():
    plugin_name = "unit_test_generic_model_missing_vars"

    if plugin_name not in PLUGIN_REGISTRY:
        register_plugin(
            AnalysisToolPlugin(
                tool_name=plugin_name,
                display_name="Unit Test Generic Model Missing Vars",
                execute=dummy_execute,
                argument_schema=ArgumentSchema(),
                method_family="modeling",
                variable_roles=[
                    VariableRoleSpec(
                        role_name="outcome",
                        required=True,
                        user_must_select=True,
                        allowed_semantic_types=["continuous_numeric"],
                    )
                ],
            )
        )

    df = pd.DataFrame({"y": [1.2, 2.4, 3.1, 4.7]})
    profile = profile_dataframe(df, data_version_id="raw_v1")

    step = PlanStep(
        step_id="step_test",
        title="Generic Model",
        tool_name=plugin_name,
        status="ready",
        execution_ready=True,
    )

    verified = verify_plan_step(step, profile)

    assert verified.status == "needs_user_choice"
    assert verified.execution_ready is False
    assert "outcome" in verified.required_user_choices


def test_verify_plan_step_blocks_wrong_semantic_type():
    plugin_name = "unit_test_continuous_only_method"

    if plugin_name not in PLUGIN_REGISTRY:
        register_plugin(
            AnalysisToolPlugin(
                tool_name=plugin_name,
                display_name="Unit Test Continuous Only Method",
                execute=dummy_execute,
                argument_schema=ArgumentSchema(),
                method_family="modeling",
                variable_roles=[
                    VariableRoleSpec(
                        role_name="outcome",
                        required=True,
                        user_must_select=True,
                        allowed_semantic_types=["continuous_numeric"],
                    )
                ],
            )
        )

    df = pd.DataFrame({"group": ["A", "B", "A", "B"]})
    profile = profile_dataframe(df, data_version_id="raw_v1")

    step = PlanStep(
        step_id="step_test",
        title="Continuous Only Method",
        tool_name=plugin_name,
        variables={"outcome": "group"},
    )

    verified = verify_plan_step(step, profile)

    assert verified.status == "not_applicable"
    assert verified.execution_ready is False
    assert verified.warnings


def test_verify_plan_moves_not_applicable_steps_to_blocked_list():
    plugin_name = "unit_test_continuous_only_method_for_plan"

    if plugin_name not in PLUGIN_REGISTRY:
        register_plugin(
            AnalysisToolPlugin(
                tool_name=plugin_name,
                display_name="Unit Test Continuous Only Method For Plan",
                execute=dummy_execute,
                argument_schema=ArgumentSchema(),
                method_family="modeling",
                variable_roles=[
                    VariableRoleSpec(
                        role_name="outcome",
                        required=True,
                        user_must_select=True,
                        allowed_semantic_types=["continuous_numeric"],
                    )
                ],
            )
        )

    df = pd.DataFrame({"group": ["A", "B", "A", "B"]})
    profile = profile_dataframe(df, data_version_id="raw_v1")

    plan = PlanProposal(
        plan_id="plan_test",
        user_request="make a plan",
        data_version_id="raw_v1",
        steps=[
            PlanStep(
                step_id="step_bad",
                title="Bad Step",
                tool_name=plugin_name,
                variables={"outcome": "group"},
            )
        ],
    )

    verified_plan = verify_plan(plan, profile)

    assert len(verified_plan.steps) == 0
    assert len(verified_plan.blocked_or_not_recommended) == 1
    assert verified_plan.blocked_or_not_recommended[0].status == "not_applicable"