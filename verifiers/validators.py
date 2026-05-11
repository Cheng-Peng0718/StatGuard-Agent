from core.analysis_tool_plugins.validation import validate_plugin_action


def verify(action, profile):
    """
    Runtime verifier.

    New architecture:
    - Tool existence comes from core.analysis_tool_plugins.registry.
    - Argument validation comes from plugin.argument_schema.
    - Human review decision comes from plugin.requires_confirmation.
    - tools/ is no longer authoritative.
    """
    result = validate_plugin_action(action, profile)

    print("\n" + "=" * 40)
    print("[VERIFIER DECISION]")
    print(f"tool_name = {getattr(action, 'tool_name', None)}")
    print(f"status = {result.status}")
    print(f"error_code = {result.error_code}")
    print(f"details = {result.details}")
    print("=" * 40 + "\n")

    return result.status, result.feedback, result