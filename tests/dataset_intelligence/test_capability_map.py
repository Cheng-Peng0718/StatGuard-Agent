import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    VariableRoleSpec,
)
from core.dataset_intelligence.capability_map import build_capability_map
from core.dataset_intelligence.profiler import profile_dataframe


def dummy_execute(context):
    return {"status": "ok", "details": {}}


def test_capability_map_marks_required_roles_as_needs_user_choice():
    df = pd.DataFrame({
        "y": [1.2, 2.4, 3.1, 4.7],
        "x": [10.5, 20.2, 30.8, 40.1],
        "group": ["A", "B", "A", "B"],
    })

    profile = profile_dataframe(df, data_version_id="raw_v1")

    plugin = AnalysisToolPlugin(
        tool_name="generic_model",
        display_name="Generic Model",
        execute=dummy_execute,
        argument_schema=ArgumentSchema(),
        method_family="modeling",
        variable_roles=[
            VariableRoleSpec(
                role_name="outcome",
                required=True,
                user_must_select=True,
                allowed_semantic_types=["continuous_numeric"],
            ),
            VariableRoleSpec(
                role_name="predictors",
                required=True,
                user_must_select=True,
                allowed_semantic_types=[
                    "continuous_numeric",
                    "binary_categorical",
                    "nominal_categorical",
                ],
                max_variables=None,
            ),
        ],
    )

    capability_map = build_capability_map(
        profile,
        plugins={"generic_model": plugin},
    )

    cap = capability_map.capabilities[0]

    assert cap.tool_name == "generic_model"
    assert cap.status == "needs_user_choice"
    assert "outcome" in cap.required_user_choices
    assert "predictors" in cap.required_user_choices
    assert "y" in cap.candidate_variables["outcome"]
    assert "x" in cap.candidate_variables["predictors"]
    assert "group" in cap.candidate_variables["predictors"]


def test_capability_map_blocks_when_no_compatible_columns():
    df = pd.DataFrame({
        "text": ["hello", "world", "foo", "bar"],
    })

    profile = profile_dataframe(df, data_version_id="raw_v1")

    plugin = AnalysisToolPlugin(
        tool_name="numeric_only_method",
        display_name="Numeric Only Method",
        execute=dummy_execute,
        argument_schema=ArgumentSchema(),
        method_family="numeric",
        variable_roles=[
            VariableRoleSpec(
                role_name="columns",
                required=True,
                user_must_select=True,
                allowed_semantic_types=["continuous_numeric"],
                min_variables=1,
                max_variables=None,
            ),
        ],
    )

    capability_map = build_capability_map(
        profile,
        plugins={"numeric_only_method": plugin},
    )

    cap = capability_map.capabilities[0]

    assert cap.status == "not_applicable"
    assert cap.warnings

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    PlanningPolicy,
)
from core.dataset_intelligence.capability_map import build_capability_map
from core.dataset_intelligence.profiler import profile_dataframe
import pandas as pd


def dummy_execute(context):
    return {"status": "ok", "details": {}}


def test_capability_map_marks_no_variable_plugin_as_ready_when_policy_allows():
    df = pd.DataFrame({
        "y": [1.2, 2.4, 3.1, 4.7],
        "x": [10.5, 20.2, 30.8, 40.1],
    })

    profile = profile_dataframe(df, data_version_id="raw_v1")

    plugin = AnalysisToolPlugin(
        tool_name="generic_overview",
        display_name="Generic Overview",
        execute=dummy_execute,
        argument_schema=ArgumentSchema(),
        method_family="eda",
        variable_roles=[],
        planning_policy=PlanningPolicy(
            include_in_capability_map=True,
            ready_without_user_variables=True,
            allow_default_arguments=True,
            requires_variable_contract=False,
            planning_description="Can summarize the dataset without user-selected variables.",
        ),
    )

    capability_map = build_capability_map(
        profile,
        plugins={"generic_overview": plugin},
    )

    cap = capability_map.capabilities[0]

    assert cap.tool_name == "generic_overview"
    assert cap.status == "ready"
    assert cap.required_user_choices == []
    assert cap.candidate_variables == {}

def test_capability_map_keeps_no_contract_plugin_conservative():
    df = pd.DataFrame({
        "y": [1.2, 2.4, 3.1, 4.7],
    })

    profile = profile_dataframe(df, data_version_id="raw_v1")

    plugin = AnalysisToolPlugin(
        tool_name="no_contract_tool",
        display_name="No Contract Tool",
        execute=dummy_execute,
        argument_schema=ArgumentSchema(),
        method_family="unknown",
        variable_roles=[],
    )

    capability_map = build_capability_map(
        profile,
        plugins={"no_contract_tool": plugin},
    )

    cap = capability_map.capabilities[0]

    assert cap.status == "needs_user_choice"