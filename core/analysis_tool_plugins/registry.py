from typing import Dict

from core.analysis_tool_plugins.base import AnalysisToolPlugin


PLUGIN_REGISTRY: Dict[str, AnalysisToolPlugin] = {}


def register_plugin(plugin: AnalysisToolPlugin) -> AnalysisToolPlugin:
    if plugin.tool_name in PLUGIN_REGISTRY:
        raise ValueError(f"Duplicate analysis tool plugin registered: {plugin.tool_name}")

    PLUGIN_REGISTRY[plugin.tool_name] = plugin
    return plugin


def get_plugin(tool_name: str) -> AnalysisToolPlugin | None:
    return PLUGIN_REGISTRY.get(tool_name)


def has_plugin(tool_name: str) -> bool:
    return tool_name in PLUGIN_REGISTRY