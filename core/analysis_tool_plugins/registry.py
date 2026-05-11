from typing import Any, Dict, List

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


def get_tool_specs_for_llm() -> List[Dict[str, Any]]:
    """
    Tool listing for Supervisor prompt.

    This replaces tools.registry.get_tool_specs_for_llm().
    """
    specs = []

    for tool_name, plugin in sorted(PLUGIN_REGISTRY.items()):
        if plugin.execute is None:
            continue
        schema = plugin.argument_schema

        required = {
            name: getattr(tp, "__name__", str(tp))
            for name, tp in (schema.required or {}).items()
        }

        optional = {
            name: getattr(tp, "__name__", str(tp))
            for name, tp in (schema.optional or {}).items()
        }

        specs.append({
            "name": plugin.tool_name,
            "display_name": plugin.display_name,
            "description": (
                plugin.execute.__doc__.strip()
                if plugin.execute and plugin.execute.__doc__
                else plugin.display_name
            ),
            "requires_confirmation": plugin.requires_confirmation,
            "arguments": {
                "required": required,
                "optional": optional,
                "column_args": schema.column_args,
                "column_list_args": schema.column_list_args,
                "allow_all_columns": schema.allow_all_columns,
                "allowed_values": schema.allowed_values,
                "conditional_allowed_values": schema.conditional_allowed_values,
                "value_aliases": schema.value_aliases,
            },
        })

    return specs