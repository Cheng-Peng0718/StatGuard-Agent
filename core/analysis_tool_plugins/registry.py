from __future__ import annotations

import importlib
import pkgutil
from typing import Any, Dict, List

from core.analysis_tool_plugins.base import AnalysisToolPlugin


_PLUGINS_LOADED = False
_PLUGINS_LOADING = False


class LazyPluginRegistry(dict):
    """
    Backward-compatible lazy plugin registry.

    Importing core.analysis_tool_plugins should not import concrete plugin
    modules. Accessing the registry through public read operations discovers
    plugins on demand. This lets older code that reads PLUGIN_REGISTRY.items()
    keep working while removing import-time side effects.
    """

    def _ensure_loaded(self) -> None:
        if not _PLUGINS_LOADING:
            ensure_plugins_loaded()

    def __contains__(self, key: object) -> bool:
        self._ensure_loaded()
        return dict.__contains__(self, key)

    def __getitem__(self, key: str) -> AnalysisToolPlugin:
        self._ensure_loaded()
        return dict.__getitem__(self, key)

    def get(self, key: str, default: Any = None) -> AnalysisToolPlugin | None:
        self._ensure_loaded()
        return dict.get(self, key, default)

    def keys(self):
        self._ensure_loaded()
        return dict.keys(self)

    def values(self):
        self._ensure_loaded()
        return dict.values(self)

    def items(self):
        self._ensure_loaded()
        return dict.items(self)

    def __iter__(self):
        self._ensure_loaded()
        return dict.__iter__(self)

    def __len__(self) -> int:
        self._ensure_loaded()
        return dict.__len__(self)


PLUGIN_REGISTRY: Dict[str, AnalysisToolPlugin] = LazyPluginRegistry()


def register_plugin(plugin: AnalysisToolPlugin) -> AnalysisToolPlugin:
    existing = dict.get(PLUGIN_REGISTRY, plugin.tool_name)

    if existing is not None:
        if existing is plugin:
            return plugin
        raise ValueError(f"Duplicate analysis tool plugin registered: {plugin.tool_name}")

    dict.__setitem__(PLUGIN_REGISTRY, plugin.tool_name, plugin)
    return plugin


def load_plugins() -> None:
    """
    Explicitly discover unified analysis tool plugins.

    This function is intentionally not called at package import time.
    Public read APIs call it lazily when registered tools are actually needed.
    """
    global _PLUGINS_LOADED, _PLUGINS_LOADING

    if _PLUGINS_LOADED or _PLUGINS_LOADING:
        return

    _PLUGINS_LOADING = True
    try:
        import core.analysis_tool_plugins.plugins as plugins_pkg

        for module_info in pkgutil.iter_modules(plugins_pkg.__path__):
            module_name = module_info.name

            if module_name.startswith("_"):
                continue

            importlib.import_module(f"{plugins_pkg.__name__}.{module_name}")

        _PLUGINS_LOADED = True
    finally:
        _PLUGINS_LOADING = False


def ensure_plugins_loaded() -> None:
    load_plugins()


def get_plugin(tool_name: str) -> AnalysisToolPlugin | None:
    ensure_plugins_loaded()
    return dict.get(PLUGIN_REGISTRY, tool_name)


def has_plugin(tool_name: str) -> bool:
    ensure_plugins_loaded()
    return dict.__contains__(PLUGIN_REGISTRY, tool_name)


def get_tool_specs_for_llm() -> List[Dict[str, Any]]:
    """
    Tool listing for Supervisor prompt.

    This replaces tools.registry.get_tool_specs_for_llm().
    """
    ensure_plugins_loaded()
    specs = []

    for tool_name, plugin in sorted(dict.items(PLUGIN_REGISTRY)):
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