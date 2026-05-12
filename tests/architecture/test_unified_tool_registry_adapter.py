from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.registry import (
    get_plugin,
    get_tool_specs_for_llm,
)


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

    register_plugin(plugin)

    registered = get_plugin("unit_test_unified_tool")

    assert registered is plugin

    result = registered.run(context=None)

    assert result["status"] == "ok"
    assert result["details"]["value"] == 123


def test_get_tool_specs_for_llm_lists_registered_plugins():
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

    specs = get_tool_specs_for_llm()

    assert "unit_test_registered_unified_tool" in specs
    assert specs["unit_test_registered_unified_tool"]["display_name"] == "Unit Test Registered Unified Tool"