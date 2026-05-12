from core.analysis_tool_plugins.base import AnalysisToolPlugin, ArgumentSchema
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.validation import validate_tool_call_schema


def test_unified_plugin_schema_is_used_by_legacy_validator():
    plugin = AnalysisToolPlugin(
        tool_name="unit_test_schema_adapter_tool",
        display_name="Unit Test Schema Adapter Tool",
        argument_schema=ArgumentSchema(
            required={
                "x_col": str,
                "y_col": str,
            },
            optional={
                "method": str,
            },
            column_args=[
                "x_col",
                "y_col",
            ],
            column_list_args=[],
            allow_all_columns=False,
        ),
    )

    register_plugin(plugin)

    result = validate_tool_call_schema(
        "unit_test_schema_adapter_tool",
        {
            "x_col": "A",
            "y_col": "B",
            "method": "pearson",
        },
        profile=None,
    )

    assert result["status"] == "ok"


def test_unified_plugin_schema_blocks_missing_required_args():
    plugin = AnalysisToolPlugin(
        tool_name="unit_test_schema_missing_required_tool",
        display_name="Unit Test Schema Missing Required Tool",
        argument_schema=ArgumentSchema(
            required={
                "x_col": str,
                "y_col": str,
            },
            optional={},
            column_args=[
                "x_col",
                "y_col",
            ],
            column_list_args=[],
            allow_all_columns=False,
        ),
    )

    register_plugin(plugin)

    result = validate_tool_call_schema(
        "unit_test_schema_missing_required_tool",
        {
            "x_col": "A",
        },
        profile=None,
    )

    assert result["status"] == "blocked"
    assert result["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert "y_col" in result["details"]["missing_required_arguments"]


def test_existing_plugin_schema_works_for_existing_tools():
    result = validate_tool_call_schema(
        "run_multiple_regression",
        {
            "target_col": "GPA",
            "feature_cols": ["SATM"],
        },
        profile=None,
    )

    assert result["status"] == "ok"