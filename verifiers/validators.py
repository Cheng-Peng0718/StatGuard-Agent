from tools.registry import registry
from tools.tool_schema import validate_tool_call_schema


def _tool_requires_confirmation(tool_spec) -> bool:
    if tool_spec is None:
        return False

    if isinstance(tool_spec, dict):
        return bool(tool_spec.get("requires_confirmation", False))

    return bool(getattr(tool_spec, "requires_confirmation", False))


def verify(action, profile):
    """
    Verify whether the proposed action is valid and needs human review.
    Phase 0:
    1. tool exists
    2. schema validation
    3. requires_confirmation
    """
    tool_name = action.tool_name

    if not tool_name:
        return "rejected_recoverable", "Error: tool_name is missing."

    if tool_name not in registry.tools:
        return "rejected_terminal", f"Error: tool '{tool_name}' is not registered."

    tool_spec = registry.tools[tool_name]


##### DEBUG
    print("\n" + "=" * 40)
    print("[VERIFIER DEBUG]")
    print(f"tool_name = {tool_name}")
    print(f"tool_spec = {tool_spec}")
    print(f"requires_confirmation = {getattr(tool_spec, 'requires_confirmation', None)}")
    print(f"action.arguments = {getattr(action, 'arguments', {})}")
    print("=" * 40 + "\n")
##### DEBUG

    # 2. schema validation
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



    if _tool_requires_confirmation(tool_spec):

        ##### DEBUG
        print(f"[VERIFIER DECISION] {tool_name} -> needs_review")
        ##### DEBUG

        return "needs_review", (
            f"Action '{tool_name}' mutates data or is high-risk; user confirmation is required before execution."
        )

    ##### DEBUG
    print(f"[VERIFIER DECISION] {tool_name} -> allowed")
    ##### DEBUG

    return "allowed", "Validation passed; executing..."