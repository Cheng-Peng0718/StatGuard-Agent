from core.analysis_tool_plugins import get_plugin
from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.manifest import build_tool_manifest
from core.analysis_tool_plugins.planning_contracts import PlanningMetadata


def test_analysis_tool_plugin_accepts_local_planning_metadata():
    plugin = AnalysisToolPlugin(
        tool_name="fake_local_planning_tool",
        display_name="Fake Local Planning Tool",
        planning_metadata=PlanningMetadata(
            supported_goal_types=["fake_goal"],
            planning_tags=["fake", "local"],
            default_plan_purpose="Use local planning metadata.",
            expected_deliverables=["fake_deliverable"],
            plan_order=7,
        ),
    )

    manifest = build_tool_manifest(plugin)

    assert manifest.supported_goal_types == ["fake_goal"]
    assert manifest.planning_tags == ["fake", "local"]
    assert manifest.default_plan_purpose == "Use local planning metadata."
    assert manifest.expected_deliverables == ["fake_deliverable"]
    assert manifest.plan_order == 7


def test_run_multiple_regression_declares_local_planning_metadata():
    plugin = get_plugin("run_multiple_regression")

    assert plugin is not None
    assert plugin.planning_metadata.supported_goal_types == [
        "regression_modeling",
    ]
    assert plugin.planning_metadata.expected_deliverables == [
        "regression_model",
    ]
    assert plugin.planning_metadata.task_argument_bindings == [
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


def test_run_multiple_regression_manifest_uses_local_planning_metadata():
    plugin = get_plugin("run_multiple_regression")
    manifest = build_tool_manifest(plugin)

    assert manifest.supported_goal_types == [
        "regression_modeling",
    ]
    assert manifest.expected_deliverables == [
        "regression_model",
    ]
    assert manifest.task_argument_bindings == [
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
    assert manifest.plan_order == 10

def test_regression_diagnostics_declares_local_planning_metadata():
    plugin = get_plugin("regression_diagnostics")

    assert plugin is not None
    assert plugin.planning_metadata.supported_goal_types == [
        "regression_modeling",
    ]
    assert plugin.planning_metadata.expected_deliverables == [
        "regression_diagnostics",
    ]
    assert plugin.planning_metadata.plan_order == 20


def test_residual_histogram_declares_local_planning_metadata():
    plugin = get_plugin("generate_residual_histogram")

    assert plugin is not None
    assert plugin.planning_metadata.supported_goal_types == [
        "regression_modeling",
        "visualization",
    ]
    assert plugin.planning_metadata.expected_deliverables == [
        "residual_distribution",
    ]
    assert plugin.planning_metadata.plan_order == 30


def test_clean_data_declares_local_planning_metadata():
    plugin = get_plugin("clean_data")

    assert plugin is not None
    assert plugin.planning_metadata.supported_goal_types == [
        "data_cleaning",
    ]
    assert plugin.planning_metadata.expected_deliverables == [
        "cleaned_dataset_version",
    ]
    assert plugin.planning_metadata.required_planning_choices == [
        "action_type",
        "strategy",
    ]
    assert plugin.planning_metadata.task_argument_bindings == [
        {
            "task_field": "target_variables",
            "argument": "columns",
            "required_choice": "columns",
        },
    ]


def test_migrated_plugin_manifests_use_local_planning_metadata():
    expected = {
        "regression_diagnostics": {
            "expected_deliverables": ["regression_diagnostics"],
            "plan_order": 20,
        },
        "generate_residual_histogram": {
            "expected_deliverables": ["residual_distribution"],
            "plan_order": 30,
        },
        "clean_data": {
            "expected_deliverables": ["cleaned_dataset_version"],
            "plan_order": 10,
        },
    }

    for tool_name, values in expected.items():
        plugin = get_plugin(tool_name)
        manifest = build_tool_manifest(plugin)

        assert manifest.expected_deliverables == values["expected_deliverables"]
        assert manifest.plan_order == values["plan_order"]
        assert manifest.supported_goal_types