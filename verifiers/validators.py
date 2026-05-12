from core.analysis_tool_plugins.registry import get_plugin
from core.analysis_tool_plugins.validation import validate_tool_call_schema


def verify(action, profile):
    """
    Verify whether the proposed action is valid and needs human review.

    Stabilized plugin-only mode:
    1. plugin exists
    2. schema validation
    3. requires_confirmation
    """
    tool_name = action.tool_name

    SQL_TOOLS = {"inspect_sql_schema", "run_sql_query"}

    if profile is None and tool_name not in SQL_TOOLS:
        return "rejected_recoverable", (
            "No in-memory CSV/DataFrame dataset is loaded. "
            "Use SQL tools if the user provided a database path, "
            "or ask the user to upload a dataset before using DataFrame-specific tools."
        )

    if not tool_name:
        return "rejected_recoverable", "Error: tool_name is missing."

    plugin = get_plugin(tool_name)

    if plugin is None:
        return "rejected_terminal", (
            f"Error: tool '{tool_name}' is not registered as an analysis tool plugin."
        )

    schema_result = validate_tool_call_schema(
        tool_name=tool_name,
        arguments=getattr(action, "arguments", {}) or {},
        profile=profile,
    )

    if schema_result["status"] == "blocked":
        feedback = (
            f"Tool schema validation failed.\n"
            f"error_code={schema_result.get('error_code')}\n"
            f"message={schema_result.get('message')}\n"
            f"details={schema_result.get('details')}"
        )
        return "rejected_recoverable", feedback

    if bool(getattr(plugin, "requires_confirmation", False)):
        print(f"[VERIFIER DECISION] {tool_name} -> needs_review")

        return "needs_review", (
            f"Action '{tool_name}' mutates data or is high-risk; user confirmation is required before execution."
        )

    print(f"[VERIFIER DECISION] {tool_name} -> allowed")

    return "allowed", "Validation passed; executing..."