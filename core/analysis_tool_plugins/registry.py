from typing import Any, Dict

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


def list_plugins(*, executable_only: bool = False) -> Dict[str, AnalysisToolPlugin]:
    if not executable_only:
        return dict(PLUGIN_REGISTRY)

    return {
        name: plugin
        for name, plugin in PLUGIN_REGISTRY.items()
        if plugin.execute is not None
    }


def _type_name(t: Any) -> str:
    if t is object:
        return "any"
    return getattr(t, "__name__", str(t))


def _format_argument_schema(plugin: AnalysisToolPlugin) -> Dict[str, Any]:
    schema = plugin.argument_schema

    return {
        "required": {
            name: _type_name(tp)
            for name, tp in (schema.required or {}).items()
        },
        "optional": {
            name: _type_name(tp)
            for name, tp in (schema.optional or {}).items()
        },
        "column_args": list(schema.column_args or []),
        "column_list_args": list(schema.column_list_args or []),
        "allow_all_columns": bool(schema.allow_all_columns),
    }


def get_tool_specs_for_llm() -> Dict[str, Dict[str, Any]]:
    """
    Return tool descriptions for the Supervisor prompt.

    This replaces the legacy tools.registry.ToolRegistry pathway.
    The Supervisor should only see tools backed by AnalysisToolPlugin.
    """
    specs: Dict[str, Dict[str, Any]] = {}

    for name, plugin in list_plugins(executable_only=True).items():
        specs[name] = {
            "name": plugin.tool_name,
            "display_name": plugin.display_name,
            "description": plugin.description or plugin.display_name,
            "evidence_categories": list(plugin.evidence_categories or []),
            "usage_guidance": plugin.usage_guidance,
            "use_when": list(plugin.use_when or []),
            "do_not_use_when": list(plugin.do_not_use_when or []),
            "requires_data_source": plugin.requires_data_source,
            "produces_active_dataset": bool(plugin.produces_active_dataset),
            "requires_confirmation": plugin.requires_confirmation,
            "argument_schema": _format_argument_schema(plugin),
            "examples": list(plugin.examples or []),
        }

    return specs