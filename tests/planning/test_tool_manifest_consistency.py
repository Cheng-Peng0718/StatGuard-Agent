import pandas as pd

from core.analysis_tool_plugins import PLUGIN_REGISTRY, ensure_plugins_loaded
from core.analysis_tool_plugins.manifest import ToolManifest, build_tool_manifests
from core.analysis_tool_plugins.planning_metadata import TOOL_PLANNING_METADATA
from core.data.context_refresh import refresh_dataset_context_from_df
from core.deliverables.gate import evaluate_deliverable_gate_state
from core.domain.task import TaskSpec
from core.services import intelligent_planner as planner


def _manifests():
    ensure_plugins_loaded()
    return build_tool_manifests(dict(PLUGIN_REGISTRY))


def _context(df):
    refreshed = refresh_dataset_context_from_df(
        df,
        dataset_name="student_data",
        data_version_id="raw_v1",
    )
    return refreshed.dataset_profile_v2, refreshed.capability_map


def test_hardcoded_goal_lists_are_covered_by_manifest_goal_metadata():
    manifests = _manifests()

    for tool_name in planner.DATASET_OVERVIEW_TOOLS:
        manifest = manifests[tool_name]
        assert (
            "dataset_overview" in manifest.supported_goal_types
            or "eda" in manifest.supported_goal_types
            or "eda" in manifest.planning_tags
        ), tool_name

    for tool_name in planner.REGRESSION_TOOLS:
        manifest = manifests[tool_name]
        assert (
            "regression_modeling" in manifest.supported_goal_types
            or "regression" in manifest.planning_tags
            or "diagnostics" in manifest.planning_tags
        ), tool_name

    for tool_name in planner.ASSOCIATION_TOOLS:
        manifest = manifests[tool_name]
        assert (
            "association_analysis" in manifest.supported_goal_types
            or "association" in manifest.planning_tags
        ), tool_name

    for tool_name in planner.VISUALIZATION_TOOLS:
        manifest = manifests[tool_name]
        assert (
            "visualization" in manifest.supported_goal_types
            or "visualization" in manifest.planning_tags
        ), tool_name

    for tool_name in planner.DATA_CLEANING_TOOLS:
        manifest = manifests[tool_name]
        assert "data_cleaning" in manifest.supported_goal_types, tool_name


def test_manifest_plan_order_preserves_current_hardcoded_order():
    manifests = _manifests()

    for tool_names in [
        planner.DATASET_OVERVIEW_TOOLS,
        planner.REGRESSION_TOOLS,
        planner.ASSOCIATION_TOOLS,
        planner.VISUALIZATION_TOOLS,
        planner.DATA_CLEANING_TOOLS,
    ]:
        orders = [
            manifests[tool_name].plan_order
            for tool_name in tool_names
        ]

        assert orders == sorted(orders), tool_names

    assert (
        manifests["run_multiple_regression"].plan_order
        < manifests["regression_diagnostics"].plan_order
        < manifests["generate_residual_histogram"].plan_order
    )


def test_overview_disallowed_tools_match_manifest_exclusion_metadata():
    manifests = _manifests()

    for tool_name in planner.DISALLOWED_FOR_OVERVIEW:
        assert (
            "dataset_overview"
            in manifests[tool_name].not_recommended_for_goal_types
        ), tool_name


def test_manifest_purpose_text_matches_current_planner_purpose_map():
    manifests = _manifests()
    purpose_tools = [
        "inspect_dataset",
        "missingness_report",
        "get_summary_stats",
        "summarize_columns",
        "get_correlation_matrix",
        "run_multiple_regression",
        "regression_diagnostics",
        "generate_residual_histogram",
        "run_correlation_test",
        "generate_scatterplot",
        "clean_data",
    ]

    for tool_name in purpose_tools:
        manifest_purpose = manifests[tool_name].default_plan_purpose
        planner_purpose = planner._purpose_for_tool(
            tool_name,
            manifests[tool_name].supported_goal_types[0],
        )

        assert manifest_purpose == planner_purpose, tool_name


def test_registered_plugin_manifests_have_expected_deliverables_metadata():
    manifests = _manifests()

    for tool_name in TOOL_PLANNING_METADATA:
        manifest = manifests[tool_name]
        assert isinstance(manifest.expected_deliverables, list), tool_name
        assert manifest.expected_deliverables, tool_name


def test_known_tool_expected_deliverables_match_manifest_metadata():
    manifests = _manifests()

    expected_by_tool = {
        "inspect_dataset": ["dataset_overview"],
        "missingness_report": ["missingness_assessment"],
        "get_summary_stats": ["descriptive_statistics"],
        "get_correlation_matrix": ["correlation_screening"],
        "run_multiple_regression": ["regression_model"],
        "regression_diagnostics": ["regression_diagnostics"],
        "generate_residual_histogram": ["residual_distribution"],
        "clean_data": ["cleaned_dataset_version"],
        "run_anova": ["group_comparison_test"],
        "run_chi_square": ["categorical_association_test"],
        "run_independent_t_test": ["group_comparison_test"],
    }

    for tool_name, expected_deliverables in expected_by_tool.items():
        assert manifests[tool_name].expected_deliverables == expected_deliverables


def test_manifest_exposes_task_argument_bindings_for_planner():
    manifests = _manifests()

    expected_regression_bindings = [
        {
            "task_field": "target_variables",
            "index": 0,
            "argument": "target_col",
            "required_choice": "target_col",
        },
        {
            "task_field": "predictor_variables",
            "argument": "feature_cols",
            "required_choice": "feature_cols",
        },
    ]

    assert (
        manifests["run_multiple_regression"].task_argument_bindings
        == expected_regression_bindings
    )
    assert (
        manifests["regression_diagnostics"].task_argument_bindings
        == expected_regression_bindings
    )
    assert (
        manifests["generate_residual_histogram"].task_argument_bindings
        == expected_regression_bindings
    )

    assert manifests["run_correlation_test"].task_argument_bindings == [
        {
            "task_field": "predictor_variables",
            "index": 0,
            "argument": "x_col",
            "required_choice": "x_col",
        },
        {
            "task_field": "predictor_variables",
            "index": 1,
            "argument": "y_col",
            "required_choice": "y_col",
        },
    ]

    assert manifests["clean_data"].task_argument_bindings == [
        {
            "task_field": "target_variables",
            "argument": "columns",
            "required_choice": "columns",
        }
    ]
    assert manifests["clean_data"].required_planning_choices == [
        "action_type",
        "strategy",
    ]


def test_planner_can_build_arguments_from_manifest_task_bindings(monkeypatch):
    manifest = ToolManifest(
        tool_name="fake_manifest_tool",
        display_name="Fake Manifest Tool",
        task_argument_bindings=[
            {
                "task_field": "target_variables",
                "index": 0,
                "argument": "target_col",
                "required_choice": "target_col",
            },
            {
                "task_field": "predictor_variables",
                "argument": "feature_cols",
                "required_choice": "feature_cols",
            },
        ],
    )

    monkeypatch.setattr(
        planner,
        "_manifest_for_tool",
        lambda tool_name: (
            manifest
            if tool_name == "fake_manifest_tool"
            else None
        ),
    )

    arguments = planner._step_arguments_for_task(
        "fake_manifest_tool",
        TaskSpec(
            goal_type="fake_goal",
            user_goal="Test manifest bindings.",
            target_variables=["GPA"],
            predictor_variables=["SATM", "ACT"],
        ),
    )

    assert arguments == {
        "target_col": "GPA",
        "feature_cols": ["SATM", "ACT"],
    }

def test_correlation_test_arguments_are_built_from_manifest_bindings():
    arguments = planner._step_arguments_for_task(
        "run_correlation_test",
        TaskSpec(
            goal_type="association_analysis",
            user_goal="Test association.",
            predictor_variables=["GPA", "SATM"],
        ),
    )

    assert arguments == {
        "x_col": "GPA",
        "y_col": "SATM",
    }


def test_correlation_test_manifest_bindings_report_missing_choices():
    from core.dataset_intelligence.schemas import AnalysisCapability

    capability = AnalysisCapability(
        tool_name="run_correlation_test",
        display_name="Correlation Test",
        method_family="association_test",
        status="needs_user_choice",
        reason="Needs selected variables.",
        required_user_choices=[],
    )

    missing = planner._required_choices_for_task(
        capability,
        TaskSpec(
            goal_type="association_analysis",
            user_goal="Test association.",
            predictor_variables=["GPA"],
        ),
    )

    assert missing == ["y_col"]

def test_current_planner_does_not_populate_manifest_expected_deliverables():
    profile, capability_map = _context(pd.DataFrame({
        "GPA": [3.0, 3.2, 3.8, 4.0],
        "SATM": [600, 620, 650, 700],
        "Sex": ["F", "M", "F", "M"],
    }))

    plan = planner.create_plan(
        user_request="run linear regression of GPA on SATM",
        task_spec=TaskSpec(
            goal_type="regression_modeling",
            user_goal="Fit a regression model.",
            source_user_request="run linear regression of GPA on SATM",
            target_variables=["GPA"],
            predictor_variables=["SATM"],
            requested_methods=["linear_regression"],
        ),
        dataset_profile=profile,
        capability_map=capability_map,
    )

    assert plan.steps
    assert all(step.expected_deliverables == [] for step in plan.steps)


def test_manifest_expected_deliverables_do_not_satisfy_task_contract_evidence():
    result = evaluate_deliverable_gate_state({
        "pending_plan": {
            "plan_id": "plan_regression",
            "steps": [
                {
                    "tool_name": "run_multiple_regression",
                    "expected_deliverables": ["regression_model"],
                }
            ],
        },
        "task_contract": {
            "contract_id": "contract_regression",
            "user_goal": "Fit a regression model.",
            "required_deliverables": [
                {
                    "deliverable_id": "regression_model",
                    "satisfied_by": ["run_multiple_regression"],
                    "required_evidence": ["status_ok"],
                }
            ],
        },
        "analysis_runs": [],
    })

    assert result.status == "needs_more_work"
    assert "deliverable:regression_model" in result.missing
    assert "deliverable:regression_model" not in result.satisfied


def _plan_for_goal(goal_type, *, target_variables=None, predictor_variables=None):
    profile, capability_map = _context(pd.DataFrame({
        "GPA": [3.0, 3.2, 3.8, 4.0],
        "SATM": [600, 620, 650, 700],
        "ACT": [24, 25, 27, 30],
        "Sex": ["F", "M", "F", "M"],
    }))

    return planner.create_plan(
        user_request=f"make a {goal_type} plan",
        task_spec=TaskSpec(
            goal_type=goal_type,
            user_goal=f"Make a {goal_type} plan.",
            source_user_request=f"make a {goal_type} plan",
            target_variables=target_variables or [],
            predictor_variables=predictor_variables or [],
        ),
        dataset_profile=profile,
        capability_map=capability_map,
    )


def _profile_for_df(df):
    profile, _ = _context(df)
    return profile


def _tool_names(plan):
    return [
        step.tool_name
        for step in plan.steps
    ]


def _tools_from_manifest_goal(manifests, goal_type):
    selected = [
        manifest
        for manifest in manifests.values()
        if goal_type in manifest.supported_goal_types
    ]
    selected = sorted(
        selected,
        key=lambda manifest: manifest.plan_order,
    )

    return [
        manifest.tool_name
        for manifest in selected
    ]


def _apply_overview_numeric_count_filter(tool_names, *, numeric_column_count):
    filtered = list(tool_names)

    if numeric_column_count < 2 and "get_correlation_matrix" in filtered:
        filtered.remove("get_correlation_matrix")

    return filtered


def test_manifest_goal_selection_matches_zero_diff_hardcoded_candidates():
    manifests = _manifests()

    assert (
        _tools_from_manifest_goal(manifests, "data_cleaning")
        == planner.DATA_CLEANING_TOOLS
    )
    assert (
        _tools_from_manifest_goal(manifests, "visualization")
        == planner.VISUALIZATION_TOOLS
    )
    assert (
        _tools_from_manifest_goal(manifests, "regression_modeling")
        == planner.REGRESSION_TOOLS
    )
    assert planner.REGRESSION_TOOLS == [
        "run_multiple_regression",
        "regression_diagnostics",
        "generate_residual_histogram",
    ]


def test_planner_data_cleaning_goal_selection_can_use_manifest_metadata():
    assert (
        planner._tools_for_goal_from_manifest("data_cleaning")
        == planner.DATA_CLEANING_TOOLS
    )


def test_planner_visualization_goal_selection_can_use_manifest_metadata():
    assert (
        planner._tools_for_goal_from_manifest("visualization")
        == planner.VISUALIZATION_TOOLS
    )


def test_planner_regression_goal_selection_can_use_manifest_metadata():
    assert (
        planner._tools_for_goal_from_manifest("regression_modeling")
        == planner.REGRESSION_TOOLS
    )


def test_planner_overview_style_goal_selection_can_use_manifest_metadata():
    assert (
        planner._tools_for_goal_from_manifest("dataset_overview")
        == planner.DATASET_OVERVIEW_TOOLS
    )
    assert (
        planner._tools_for_goal_from_manifest("analysis_planning")
        == planner.DATASET_OVERVIEW_TOOLS
    )
    assert (
        planner._tools_for_goal_from_manifest("analysis_recommendation")
        == planner.DATASET_OVERVIEW_TOOLS
    )


def test_manifest_overview_selection_matches_after_numeric_count_filter():
    manifests = _manifests()
    overview_tools = _tools_from_manifest_goal(manifests, "dataset_overview")

    assert (
        _apply_overview_numeric_count_filter(
            overview_tools,
            numeric_column_count=2,
        )
        == planner.DATASET_OVERVIEW_TOOLS
    )
    assert (
        _apply_overview_numeric_count_filter(
            overview_tools,
            numeric_column_count=1,
        )
        == [
            tool_name
            for tool_name in planner.DATASET_OVERVIEW_TOOLS
            if tool_name != "get_correlation_matrix"
        ]
    )


def test_manifest_association_selection_is_not_zero_diff_yet():
    manifests = _manifests()

    manifest_tools = _tools_from_manifest_goal(
        manifests,
        "association_analysis",
    )

    assert manifest_tools != planner.ASSOCIATION_TOOLS
    assert set(planner.ASSOCIATION_TOOLS).issubset(set(manifest_tools))
    assert {
        "generate_scatterplot",
        "run_chi_square",
    }.issubset(set(manifest_tools) - set(planner.ASSOCIATION_TOOLS))


def test_manifest_eda_selection_is_not_current_planner_fallback():
    manifests = _manifests()

    manifest_tools = _tools_from_manifest_goal(manifests, "eda")
    current_fallback = planner.DATASET_OVERVIEW_TOOLS[:3]

    assert manifest_tools != current_fallback
    assert current_fallback == [
        "inspect_dataset",
        "missingness_report",
        "get_summary_stats",
    ]
    assert {
        "summarize_columns",
        "get_correlation_matrix",
    }.issubset(set(manifest_tools) - set(current_fallback))


def test_planner_output_order_matches_current_hardcoded_selection_order():
    overview_plan = _plan_for_goal("dataset_overview")
    regression_plan = _plan_for_goal(
        "regression_modeling",
        target_variables=["GPA"],
        predictor_variables=["SATM"],
    )
    association_plan = _plan_for_goal(
        "association_analysis",
        predictor_variables=["GPA", "SATM"],
    )
    visualization_plan = _plan_for_goal(
        "visualization",
        target_variables=["GPA"],
        predictor_variables=["SATM"],
    )
    cleaning_plan = _plan_for_goal(
        "data_cleaning",
        target_variables=["GPA"],
    )
    eda_plan = _plan_for_goal("eda")

    assert _tool_names(overview_plan) == planner.DATASET_OVERVIEW_TOOLS
    assert _tool_names(regression_plan) == planner.REGRESSION_TOOLS
    assert _tool_names(association_plan) == planner.ASSOCIATION_TOOLS
    assert _tool_names(visualization_plan) == planner.VISUALIZATION_TOOLS
    assert _tool_names(cleaning_plan) == planner.DATA_CLEANING_TOOLS
    assert _tool_names(eda_plan) == planner.DATASET_OVERVIEW_TOOLS[:3]


def test_data_cleaning_manifest_selection_falls_back_to_hardcoded_tools(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_tools_for_goal_from_manifest",
        lambda goal_type: [],
    )

    cleaning_plan = _plan_for_goal(
        "data_cleaning",
        target_variables=["GPA"],
    )

    assert _tool_names(cleaning_plan) == planner.DATA_CLEANING_TOOLS

    clean_data_step = cleaning_plan.steps[0]

    assert clean_data_step.tool_name == "clean_data"
    assert clean_data_step.requires_confirmation is True
    assert clean_data_step.mutates_data is True
    assert clean_data_step.execution_ready is False


def test_visualization_manifest_selection_falls_back_to_hardcoded_tools(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_tools_for_goal_from_manifest",
        lambda goal_type: [],
    )

    visualization_plan = _plan_for_goal(
        "visualization",
        target_variables=["GPA"],
        predictor_variables=["SATM"],
    )

    assert _tool_names(visualization_plan) == planner.VISUALIZATION_TOOLS


def test_regression_manifest_selection_falls_back_when_empty(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_tools_for_goal_from_manifest",
        lambda goal_type: [],
    )

    regression_plan = _plan_for_goal(
        "regression_modeling",
        target_variables=["GPA"],
        predictor_variables=["SATM"],
    )

    assert _tool_names(regression_plan) == planner.REGRESSION_TOOLS


def test_regression_manifest_selection_falls_back_when_partial(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_tools_for_goal_from_manifest",
        lambda goal_type: ["run_multiple_regression"],
    )

    regression_plan = _plan_for_goal(
        "regression_modeling",
        target_variables=["GPA"],
        predictor_variables=["SATM"],
    )

    assert _tool_names(regression_plan) == planner.REGRESSION_TOOLS


def test_regression_manifest_selection_preserves_arguments_and_choices():
    regression_plan = _plan_for_goal(
        "regression_modeling",
        target_variables=["GPA"],
        predictor_variables=["SATM"],
    )

    regression_step = next(
        step
        for step in regression_plan.steps
        if step.tool_name == "run_multiple_regression"
    )

    assert regression_step.arguments["target_col"] == "GPA"
    assert regression_step.arguments["feature_cols"] == ["SATM"]
    assert regression_step.required_user_choices == []

    missing_choices_plan = _plan_for_goal("regression_modeling")
    missing_choices_step = next(
        step
        for step in missing_choices_plan.steps
        if step.tool_name == "run_multiple_regression"
    )

    assert "target_col" in missing_choices_step.required_user_choices
    assert "feature_cols" in missing_choices_step.required_user_choices


def test_overview_manifest_selection_preserves_numeric_count_filter():
    numeric_profile = _profile_for_df(pd.DataFrame({
        "GPA": [3.0, 3.2, 3.8, 4.0],
        "SATM": [600, 620, 650, 700],
        "Sex": ["F", "M", "F", "M"],
    }))
    low_numeric_profile = _profile_for_df(pd.DataFrame({
        "GPA": [3.0, 3.2, 3.8, 4.0],
        "Sex": ["F", "M", "F", "M"],
        "Major": ["A", "B", "A", "C"],
    }))

    numeric_tools = planner._tools_for_goal(
        TaskSpec(
            goal_type="dataset_overview",
            user_goal="Understand the dataset.",
        ),
        numeric_profile,
    )
    low_numeric_tools = planner._tools_for_goal(
        TaskSpec(
            goal_type="dataset_overview",
            user_goal="Understand the dataset.",
        ),
        low_numeric_profile,
    )

    assert numeric_tools == planner.DATASET_OVERVIEW_TOOLS
    assert "get_correlation_matrix" in numeric_tools
    assert low_numeric_tools == [
        tool_name
        for tool_name in planner.DATASET_OVERVIEW_TOOLS
        if tool_name != "get_correlation_matrix"
    ]
    assert "get_correlation_matrix" not in low_numeric_tools


def test_overview_manifest_selection_falls_back_when_empty(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_tools_for_goal_from_manifest",
        lambda goal_type: [],
    )

    profile = _profile_for_df(pd.DataFrame({
        "GPA": [3.0, 3.2, 3.8, 4.0],
        "SATM": [600, 620, 650, 700],
        "Sex": ["F", "M", "F", "M"],
    }))

    tools = planner._tools_for_goal(
        TaskSpec(
            goal_type="dataset_overview",
            user_goal="Understand the dataset.",
        ),
        profile,
    )

    assert tools == planner.DATASET_OVERVIEW_TOOLS


def test_overview_manifest_selection_falls_back_when_partial(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_tools_for_goal_from_manifest",
        lambda goal_type: ["inspect_dataset"],
    )

    profile = _profile_for_df(pd.DataFrame({
        "GPA": [3.0, 3.2, 3.8, 4.0],
        "Sex": ["F", "M", "F", "M"],
        "Major": ["A", "B", "A", "C"],
    }))

    tools = planner._tools_for_goal(
        TaskSpec(
            goal_type="analysis_recommendation",
            user_goal="Recommend an analysis plan.",
        ),
        profile,
    )

    assert tools == [
        tool_name
        for tool_name in planner.DATASET_OVERVIEW_TOOLS
        if tool_name != "get_correlation_matrix"
    ]


def test_analysis_planning_and_recommendation_match_overview_behavior():
    profile = _profile_for_df(pd.DataFrame({
        "GPA": [3.0, 3.2, 3.8, 4.0],
        "SATM": [600, 620, 650, 700],
        "Sex": ["F", "M", "F", "M"],
    }))

    planning_tools = planner._tools_for_goal(
        TaskSpec(
            goal_type="analysis_planning",
            user_goal="Plan an analysis.",
        ),
        profile,
    )
    recommendation_tools = planner._tools_for_goal(
        TaskSpec(
            goal_type="analysis_recommendation",
            user_goal="Recommend an analysis.",
        ),
        profile,
    )

    assert planning_tools == planner.DATASET_OVERVIEW_TOOLS
    assert recommendation_tools == planner.DATASET_OVERVIEW_TOOLS


def test_manifest_order_fallback_preserves_hardcoded_selection_order(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_manifest_for_tool",
        lambda tool_name: None,
    )

    regression_plan = _plan_for_goal(
        "regression_modeling",
        target_variables=["GPA"],
        predictor_variables=["SATM"],
    )

    assert _tool_names(regression_plan) == planner.REGRESSION_TOOLS


def test_planner_purpose_falls_back_when_manifest_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_manifest_for_tool",
        lambda tool_name: None,
    )

    assert (
        planner._plan_purpose_for_tool("unknown_tool", "custom_goal")
        == "Support the custom_goal goal."
    )


def test_clean_data_safety_metadata_matches_current_planner_output():
    manifests = _manifests()
    clean_data_manifest = manifests["clean_data"]

    assert clean_data_manifest.mutates_data is True
    assert clean_data_manifest.requires_confirmation is True

    profile, capability_map = _context(pd.DataFrame({
        "GPA": [3.0, None, 3.8, 4.0],
        "SATM": [600, 620, 650, 700],
    }))

    plan = planner.create_plan(
        user_request="drop rows with missing GPA",
        task_spec=TaskSpec(
            goal_type="data_cleaning",
            user_goal="Clean missing GPA rows.",
            source_user_request="drop rows with missing GPA",
            target_variables=["GPA"],
            requested_methods=["data_cleaning"],
        ),
        dataset_profile=profile,
        capability_map=capability_map,
    )

    assert [step.tool_name for step in plan.steps] == ["clean_data"]

    clean_data_step = plan.steps[0]

    assert clean_data_step.mutates_data is True
    assert clean_data_step.requires_confirmation is True
    assert clean_data_step.execution_ready is False


def test_manifest_safety_metadata_matches_capability_map_safety():
    manifests = _manifests()
    profile, capability_map = _context(pd.DataFrame({
        "GPA": [3.0, None, 3.8, 4.0],
        "SATM": [600, 620, 650, 700],
    }))

    del profile

    capability_requires_confirmation = sorted(
        capability.tool_name
        for capability in capability_map.capabilities
        if capability.requires_confirmation
    )
    manifest_requires_confirmation = sorted(
        tool_name
        for tool_name, manifest in manifests.items()
        if manifest.requires_confirmation
    )
    capability_mutates_data = sorted(
        capability.tool_name
        for capability in capability_map.capabilities
        if capability.mutates_data
    )
    manifest_mutates_data = sorted(
        tool_name
        for tool_name, manifest in manifests.items()
        if manifest.mutates_data
    )

    assert capability_requires_confirmation == ["clean_data"]
    assert manifest_requires_confirmation == ["clean_data"]
    assert capability_mutates_data == ["clean_data"]
    assert manifest_mutates_data == ["clean_data"]

    manifest_by_tool = manifests

    for capability in capability_map.capabilities:
        manifest = manifest_by_tool[capability.tool_name]

        assert manifest.requires_confirmation == capability.requires_confirmation
        assert manifest.mutates_data == capability.mutates_data


def test_planner_safety_falls_back_to_capability_when_manifest_unavailable(monkeypatch):
    monkeypatch.setattr(
        planner,
        "_manifest_for_tool",
        lambda tool_name: None,
    )

    profile, capability_map = _context(pd.DataFrame({
        "GPA": [3.0, None, 3.8, 4.0],
        "SATM": [600, 620, 650, 700],
    }))

    plan = planner.create_plan(
        user_request="drop rows with missing GPA",
        task_spec=TaskSpec(
            goal_type="data_cleaning",
            user_goal="Clean missing GPA rows.",
            source_user_request="drop rows with missing GPA",
            target_variables=["GPA"],
            requested_methods=["data_cleaning"],
        ),
        dataset_profile=profile,
        capability_map=capability_map,
    )

    assert [step.tool_name for step in plan.steps] == ["clean_data"]

    clean_data_step = plan.steps[0]

    assert clean_data_step.requires_confirmation is True
    assert clean_data_step.mutates_data is True
    assert clean_data_step.execution_ready is False
