from core.schema import ToolSpec


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, requires_confirmation=False):
        def decorator(func):
            spec = ToolSpec.from_function(func, requires_confirmation)
            self.tools[func.__name__] = spec
            return func
        return decorator

    def register_analysis_tool_plugin(self, plugin):
        """
        Register a unified AnalysisToolPlugin as an executable tool.

        The wrapper lets the existing execution layer continue to call:
            tool_spec.func(context)
        """
        if plugin.execute is None:
            return

        def _plugin_tool(context):
            return plugin.run(context)

        _plugin_tool.__name__ = plugin.tool_name
        _plugin_tool.__doc__ = plugin.display_name

        spec = ToolSpec.from_function(
            _plugin_tool,
            plugin.requires_confirmation,
        )

        self.tools[plugin.tool_name] = spec

    def load_all_tools(self):
        """
        Load unified plugins first, then legacy tools.

        Unified plugin tools have priority if names collide.
        """
        # 1. Load unified analysis tool plugins.
        try:
            from core.analysis_tool_plugins import PLUGIN_REGISTRY

            for plugin in PLUGIN_REGISTRY.values():
                self.register_analysis_tool_plugin(plugin)
        except Exception as e:
            print(f"[ToolRegistry] Failed to load unified analysis tool plugins: {e}")

        # 2. Load legacy tool implementations.
        # They use @registry.register(), which mutates self.tools.
        try:
            import tools.methods  # noqa: F401
        except Exception as e:
            print(f"[ToolRegistry] Failed to load legacy tools.methods: {e}")

        # 3. Re-register unified plugins after legacy load so unified wins on collisions.
        try:
            from core.analysis_tool_plugins import PLUGIN_REGISTRY

            for plugin in PLUGIN_REGISTRY.values():
                self.register_analysis_tool_plugin(plugin)
        except Exception as e:
            print(f"[ToolRegistry] Failed to reload unified analysis tool plugins: {e}")

    def get_tool_specs_for_llm(self):
        if not self.tools:
            self.load_all_tools()

        return {name: t.description for name, t in self.tools.items()}


registry = ToolRegistry()