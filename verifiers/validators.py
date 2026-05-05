from core.analysis_tool_plugins import get_plugin
from core.analysis_tool_plugins.validation import validate_tool_call_schema


def _get_action_field(action, name, default=None):
    if isinstance(action, dict):
        return action.get(name, default)
    return getattr(action, name, default)


def verify(action, dataset_profile=None):
    """
    Verify whether a proposed action can run.

    Valid VerificationResult.status values:
    - allowed
    - rejected_recoverable
    - rejected_terminal
    - needs_review
    """
    tool_name = _get_action_field(action, "tool_name")
    arguments = _get_action_field(action, "arguments", {}) or {}

    if not tool_name:
        return (
            "rejected_recoverable",
            "No tool_name was provided for the proposed action.",
        )

    plugin = get_plugin(tool_name)

    if plugin is None:
        return (
            "rejected_recoverable",
            f"Unknown tool: {tool_name}",
        )

    schema_result = validate_tool_call_schema(
        tool_name,
        arguments,
        profile=dataset_profile,
    )

    schema_status = schema_result.get("status")

    if schema_status == "blocked":
        return (
            "rejected_recoverable",
            schema_result.get("message", "Tool argument validation failed."),
        )

    if plugin.requires_confirmation:
        return (
            "needs_review",
            (
                f"Tool `{tool_name}` requires user confirmation before execution "
                "because it may modify data or perform a high-risk operation."
            ),
        )

    # Schema warning should not block execution.
    if schema_status == "warning":
        return (
            "allowed",
            schema_result.get("message", "Tool schema validation produced a warning."),
        )

    return (
        "allowed",
        "Tool verification passed.",
    )