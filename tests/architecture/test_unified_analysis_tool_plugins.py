from core.analysis_tool_plugins.base import AnalysisToolPlugin, ArgumentSchema
from core.analysis_tool_plugins.registry import register_plugin, get_plugin


def test_analysis_tool_plugin_can_be_constructed():
    plugin = AnalysisToolPlugin(
        tool_name="unit_test_tool_plugin",
        display_name="Unit Test Tool Plugin",
        argument_schema=ArgumentSchema(
            required={"x_col": str},
            optional={"method": str},
            column_args=["x_col"],
        ),
    )

    assert plugin.tool_name == "unit_test_tool_plugin"
    assert plugin.display_name == "Unit Test Tool Plugin"
    assert plugin.argument_schema.required["x_col"] is str


def test_analysis_tool_plugin_can_build_fallback_analysis_run():
    plugin = AnalysisToolPlugin(
        tool_name="unit_test_tool_plugin_fallback",
        display_name="Unit Test Tool Plugin Fallback",
    )

    run = plugin.build_analysis_run(
        action_id="act_test",
        arguments={},
        data_version_id="raw_v1",
        status="ok",
        success=True,
        message="Finished.",
        payload={"hello": "world"},
        artifacts=[],
        observation_id="obs_test",
    )

    assert run["tool_name"] == "unit_test_tool_plugin_fallback"
    assert run["title"] == "Unit Test Tool Plugin Fallback"
    assert run["report_blocks"]


def test_unified_plugin_registry_registers_plugin():
    plugin = AnalysisToolPlugin(
        tool_name="unit_test_registered_plugin",
        display_name="Unit Test Registered Plugin",
    )

    register_plugin(plugin)

    assert get_plugin("unit_test_registered_plugin") is plugin