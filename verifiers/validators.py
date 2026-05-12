from core.analysis_tool_plugins.registry import get_plugin
from core.analysis_tool_plugins.validation import validate_tool_call_schema
from pathlib import Path

def _has_active_dataframe_dataset(profile=None, state=None) -> bool:
    """
    A DataFrame dataset is available if either:
    1. dataset_profile exists, or
    2. active_data_version_id points to a data version with an existing file path.
    """
    if profile is not None:
        return True

    if not isinstance(state, dict):
        return False

    active_id = state.get("active_data_version_id")
    data_versions = state.get("data_versions", []) or []

    if not active_id:
        return False

    for version in data_versions:
        if not isinstance(version, dict):
            continue

        if version.get("version_id") != active_id:
            continue

        path = version.get("path")

        if path and Path(path).exists():
            return True

    return False

def verify(action, profile=None, state=None):
    """
    Verify whether the proposed action is valid and needs human review.

    Stabilized plugin-only mode:
    1. plugin exists
    2. schema validation
    3. requires_confirmation
    """
    tool_name = action.tool_name


    if not tool_name:
        return "rejected_recoverable", "Error: tool_name is missing."

    plugin = get_plugin(tool_name)

    if plugin is None:
        return "rejected_terminal", (
            f"Error: tool '{tool_name}' is not registered as an analysis tool plugin."
        )

    requires_data_source = getattr(plugin, "requires_data_source", "any")

    if requires_data_source == "dataframe" and not _has_active_dataframe_dataset(
            profile=profile,
            state=state,
    ):
        return "rejected_recoverable", (
            f"Tool `{tool_name}` requires an active DataFrame dataset, "
            "but no active workspace dataset is currently available. "
            "Upload a dataset or materialize a SQL query result first."
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