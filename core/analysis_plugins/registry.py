from typing import Dict

from core.analysis_plugins.base import AnalysisPlugin


PLUGIN_REGISTRY: Dict[str, AnalysisPlugin] = {}


def register_plugin(plugin: AnalysisPlugin) -> AnalysisPlugin:
    if plugin.tool_name in PLUGIN_REGISTRY:
        raise ValueError(f"Duplicate analysis plugin registered: {plugin.tool_name}")

    PLUGIN_REGISTRY[plugin.tool_name] = plugin
    return plugin


def get_plugin(tool_name: str) -> AnalysisPlugin:
    plugin = PLUGIN_REGISTRY.get(tool_name)

    if plugin is not None:
        return plugin

    return AnalysisPlugin(
        tool_name=tool_name,
        display_name=tool_name.replace("_", " ").title(),
    )