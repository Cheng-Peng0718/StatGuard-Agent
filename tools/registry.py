from core.schema import ToolSpec


class ToolRegistry:
    def __init__(self):
        self.tools = {}

    def register(self, requires_confirmation=False):
        def decorator(func):
            from core.schema import ToolSpec
            spec = ToolSpec.from_function(func, requires_confirmation)
            self.tools[func.__name__] = spec
            return func
        return decorator

    def get_tool_specs_for_llm(self):
        # Lazy-load tool implementations if registry is empty
        if not self.tools:
            import tools.methods
        return {name: t.description for name, t in self.tools.items()}

registry = ToolRegistry()