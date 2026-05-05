from core.analysis_tool_plugins.base import AnalysisToolPlugin, ArgumentSchema
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.execution import execute_analysis_tool


class DummyAction:
    action_id = "act_test"
    tool_name = "unit_test_unified_execution_tool"
    arguments = {"x_col": "A"}


class DummyContextPackage:
    workspace_dir = "./"
    data_versions = []
    active_data_version_id = "raw_v1"
    data_audit_log = []


def test_execute_analysis_tool_uses_unified_plugin_execute():
    def execute_demo(context):
        return {
            "status": "ok",
            "message": "Unified plugin executed.",
            "recoverable": False,
            "details": {
                "value": 123,
                "arg_value": context.get_arg("x_col"),
            },
            "artifacts": [],
        }

    plugin = AnalysisToolPlugin(
        tool_name="unit_test_unified_execution_tool",
        display_name="Unit Test Unified Execution Tool",
        argument_schema=ArgumentSchema(
            required={"x_col": str},
            column_args=["x_col"],
        ),
        execute=execute_demo,
    )

    register_plugin(plugin)

    result = execute_analysis_tool(DummyAction(), DummyContextPackage())

    assert result.tool_name == "unit_test_unified_execution_tool"
    assert result.status == "ok"
    assert result.success is True
    assert result.payload["value"] == 123
    assert result.payload["arg_value"] == "A"