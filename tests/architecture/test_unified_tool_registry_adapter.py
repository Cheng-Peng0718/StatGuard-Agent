from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.registry import register_plugin
from tools.registry import ToolRegistry


def test_tool_registry_can_register_unified_plugin_as_tool():
    def execute_demo(context):
        return {
            "status": "ok",
            "message": "Demo plugin executed.",
            "recoverable": False,
            "details": {"value": 123},
            "artifacts": [],
        }

    plugin = AnalysisToolPlugin(
        tool_name="unit_test_unified_tool",
        display_name="Unit Test Unified Tool",
        execute=execute_demo,
    )

    registry = ToolRegistry()
    registry.register_analysis_tool_plugin(plugin)

    assert "unit_test_unified_tool" in registry.tools

    spec = registry.tools["unit_test_unified_tool"]
    result = spec.func(context=None)

    assert result["status"] == "ok"
    assert result["details"]["value"] == 123


def test_tool_registry_loads_registered_unified_plugins():
    def execute_demo(context):
        return {
            "status": "ok",
            "message": "Registered plugin executed.",
            "recoverable": False,
            "details": {"value": 456},
            "artifacts": [],
        }

    plugin = AnalysisToolPlugin(
        tool_name="unit_test_registered_unified_tool",
        display_name="Unit Test Registered Unified Tool",
        execute=execute_demo,
    )

    register_plugin(plugin)

    registry = ToolRegistry()
    registry.load_all_tools()

    assert "unit_test_registered_unified_tool" in registry.tools