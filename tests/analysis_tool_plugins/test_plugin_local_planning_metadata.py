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

def test_correlation_matrix_declares_local_planning_metadata():
    plugin = get_plugin("get_correlation_matrix")

    assert plugin is not None
    assert plugin.planning_metadata.supported_goal_types == [
        "dataset_overview",
        "analysis_recommendation",
        "analysis_planning",
        "association_analysis",
        "eda",
    ]
    assert plugin.planning_metadata.expected_deliverables == [
        "correlation_screening",
    ]
    assert plugin.planning_metadata.plan_order == 50


def test_correlation_test_declares_local_planning_metadata():
    plugin = get_plugin("run_correlation_test")

    assert plugin is not None
    assert plugin.planning_metadata.supported_goal_types == [
        "association_analysis",
    ]
    assert plugin.planning_metadata.expected_deliverables == [
        "association_test",
    ]
    assert plugin.planning_metadata.task_argument_bindings == [
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
    assert plugin.planning_metadata.plan_order == 10


def test_scatterplot_declares_local_planning_metadata():
    plugin = get_plugin("generate_scatterplot")

    assert plugin is not None
    assert plugin.planning_metadata.supported_goal_types == [
        "association_analysis",
        "visualization",
    ]
    assert plugin.planning_metadata.expected_deliverables == [
        "scatterplot",
    ]
    assert plugin.planning_metadata.task_argument_bindings == [
        {
            "task_field": "predictor_variables",
            "index": 0,
            "argument": "x_column",
            "required_choice": "x_column",
        },
        {
            "task_field": "predictor_variables",
            "index": 1,
            "argument": "y_column",
            "required_choice": "y_column",
        },
    ]
    assert plugin.planning_metadata.plan_order == 10


def test_correlation_and_visualization_manifests_use_local_planning_metadata():
    expected = {
        "get_correlation_matrix": {
            "expected_deliverables": ["correlation_screening"],
            "plan_order": 50,
        },
        "run_correlation_test": {
            "expected_deliverables": ["association_test"],
            "plan_order": 10,
        },
        "generate_scatterplot": {
            "expected_deliverables": ["scatterplot"],
            "plan_order": 10,
        },
    }

    for tool_name, values in expected.items():
        plugin = get_plugin(tool_name)
        manifest = build_tool_manifest(plugin)

        assert manifest.expected_deliverables == values["expected_deliverables"]
        assert manifest.plan_order == values["plan_order"]
        assert manifest.supported_goal_types

def test_overview_plugins_declare_local_planning_metadata():
    expected = {
        "inspect_dataset": {
            "supported_goal_types": [
                "dataset_overview",
                "analysis_recommendation",
                "analysis_planning",
                "eda",
            ],
            "expected_deliverables": ["dataset_overview"],
            "plan_order": 10,
        },
        "missingness_report": {
            "supported_goal_types": [
                "dataset_overview",
                "analysis_recommendation",
                "analysis_planning",
                "eda",
            ],
            "expected_deliverables": ["missingness_assessment"],
            "plan_order": 20,
        },
        "get_summary_stats": {
            "supported_goal_types": [
                "dataset_overview",
                "analysis_recommendation",
                "analysis_planning",
                "eda",
            ],
            "expected_deliverables": ["descriptive_statistics"],
            "plan_order": 30,
        },
        "summarize_columns": {
            "supported_goal_types": [
                "dataset_overview",
                "analysis_recommendation",
                "analysis_planning",
                "eda",
            ],
            "expected_deliverables": ["column_summary"],
            "plan_order": 40,
        },
    }

    for tool_name, values in expected.items():
        plugin = get_plugin(tool_name)

        assert plugin is not None
        assert plugin.planning_metadata.supported_goal_types == values["supported_goal_types"]
        assert plugin.planning_metadata.expected_deliverables == values["expected_deliverables"]
        assert plugin.planning_metadata.plan_order == values["plan_order"]


def test_overview_plugin_manifests_use_local_planning_metadata():
    expected = {
        "inspect_dataset": {
            "expected_deliverables": ["dataset_overview"],
            "plan_order": 10,
        },
        "missingness_report": {
            "expected_deliverables": ["missingness_assessment"],
            "plan_order": 20,
        },
        "get_summary_stats": {
            "expected_deliverables": ["descriptive_statistics"],
            "plan_order": 30,
        },
        "summarize_columns": {
            "expected_deliverables": ["column_summary"],
            "plan_order": 40,
        },
    }

    for tool_name, values in expected.items():
        plugin = get_plugin(tool_name)
        manifest = build_tool_manifest(plugin)

        assert manifest.expected_deliverables == values["expected_deliverables"]
        assert manifest.plan_order == values["plan_order"]
        assert manifest.supported_goal_types == [
            "dataset_overview",
            "analysis_recommendation",
            "analysis_planning",
            "eda",
        ]

def test_inferential_plugins_declare_local_planning_metadata():
    expected = {
        "run_anova": {
            "supported_goal_types": [
                "group_comparison",
            ],
            "expected_deliverables": ["group_comparison_test"],
            "plan_order": 10,
        },
        "run_chi_square": {
            "supported_goal_types": [
                "categorical_association",
                "association_analysis",
            ],
            "expected_deliverables": ["categorical_association_test"],
            "plan_order": 10,
        },
        "run_independent_t_test": {
            "supported_goal_types": [
                "group_comparison",
            ],
            "expected_deliverables": ["group_comparison_test"],
            "plan_order": 10,
        },
    }

    for tool_name, values in expected.items():
        plugin = get_plugin(tool_name)

        assert plugin is not None
        assert plugin.planning_metadata.supported_goal_types == values["supported_goal_types"]
        assert plugin.planning_metadata.expected_deliverables == values["expected_deliverables"]
        assert plugin.planning_metadata.plan_order == values["plan_order"]


def test_inferential_plugin_manifests_use_local_planning_metadata():
    expected = {
        "run_anova": {
            "expected_deliverables": ["group_comparison_test"],
            "plan_order": 10,
        },
        "run_chi_square": {
            "expected_deliverables": ["categorical_association_test"],
            "plan_order": 10,
        },
        "run_independent_t_test": {
            "expected_deliverables": ["group_comparison_test"],
            "plan_order": 10,
        },
    }

    for tool_name, values in expected.items():
        plugin = get_plugin(tool_name)
        manifest = build_tool_manifest(plugin)

        assert manifest.expected_deliverables == values["expected_deliverables"]
        assert manifest.plan_order == values["plan_order"]
        assert manifest.supported_goal_types

def test_all_registered_plugins_declare_local_planning_metadata():
    from core.analysis_tool_plugins import PLUGIN_REGISTRY, ensure_plugins_loaded

    ensure_plugins_loaded()

    missing = []

    for tool_name, plugin in PLUGIN_REGISTRY.items():
        if tool_name.startswith("unit_test_") or tool_name.startswith("fake_"):
            continue

        metadata = plugin.planning_metadata

        if not metadata.supported_goal_types:
            missing.append((tool_name, "supported_goal_types"))

        if not metadata.expected_deliverables:
            missing.append((tool_name, "expected_deliverables"))

        if metadata.plan_order is None:
            missing.append((tool_name, "plan_order"))

    assert missing == []


def test_manifest_builder_no_longer_depends_on_central_planning_metadata_file():
    from pathlib import Path

    manifest_text = Path("core/analysis_tool_plugins/manifest.py").read_text(
        encoding="utf-8"
    )

    assert "TOOL_PLANNING_METADATA" not in manifest_text
    assert "planning_metadata import" not in manifest_text

def test_manifest_defaults_missing_local_plan_order_to_100():
    plugin = AnalysisToolPlugin(
        tool_name="fake_missing_plan_order_tool",
        display_name="Fake Missing Plan Order Tool",
        planning_metadata=PlanningMetadata(
            supported_goal_types=["fake_goal"],
            expected_deliverables=["fake_output"],
            plan_order=None,
        ),
    )

    manifest = build_tool_manifest(plugin)

    assert manifest.plan_order == 100