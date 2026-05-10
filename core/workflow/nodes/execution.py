from __future__ import annotations

from core.action_access import (
    get_action_arguments,
    get_action_id,
    get_action_tool_name,
)
from core.analysis_tool_plugins.execution import execute_analysis_tool
from core.context_builder import build_context
from core.execution_codec import execution_to_state_dict
from core.workflow.execution_fingerprints import has_duplicate_executed_action
from core.workflow.runtime_utils import sanitize_results, get_action_hash
from core.workflow.profile_access import get_context_profile

def execute_node(state: dict):
    action = state.get("current_action")
    tool_name = get_action_tool_name(action)
    arguments = get_action_arguments(action)

    if not action or not tool_name:
        message = "Error: No valid action provided."

        return {
            "current_execution": execution_to_state_dict(
                {
                    "status": "blocked",
                    "success": False,
                    "error_code": "NO_VALID_ACTION",
                    "message": message,
                    "artifacts": [],
                    "payload": {},
                },
                fallback_action_id=get_action_id(action),
                fallback_tool_name=tool_name or "unknown_tool",
            )
        }

    if has_duplicate_executed_action(
        state=state,
        tool_name=tool_name,
        arguments=arguments,
    ):
        current_hash = get_action_hash(tool_name, arguments)

        error_msg = (
            f"[System intervention]: Execution refused. You are calling '{tool_name}' "
            f"with parameters identical to a previous executed attempt.\n"
            f"To retry, change arguments or explicitly choose a different strategy."
        )

        print(
            f"[Fingerprint gate]: blocked duplicate executed action "
            f"{tool_name} (fp: {current_hash[:6]})"
        )

        return {
            "current_execution": execution_to_state_dict(
                {
                    "status": "blocked",
                    "success": False,
                    "error_code": "DUPLICATE_EXECUTION_ATTEMPT",
                    "message": error_msg,
                    "artifacts": [],
                    "payload": {
                        "tool_name": tool_name,
                        "arguments": arguments,
                        "action_hash": current_hash,
                    },
                },
                fallback_action_id=get_action_id(action),
                fallback_tool_name=tool_name,
            )
        }

    print(f"[Execute]: {tool_name}")

    context_pkg = build_context(
        step=state.get("current_step", 1),
        max_steps=state.get("max_steps", 20),
        user_request=state.get("user_request", "Not provided"),
        profile=get_context_profile(state),
        observations=state.get("observations", []),
        workspace_dir=state.get("workspace_dir", "./"),
        deliverable_check=state.get("deliverable_check"),
        data_versions=state.get("data_versions", []),
        active_data_version_id=state.get("active_data_version_id"),
        data_audit_log=state.get("data_audit_log", []),
    )

    exec_result = execute_analysis_tool(action, context_pkg)

    if hasattr(exec_result, "model_dump"):
        raw_payload = exec_result.model_dump()
    elif hasattr(exec_result, "dict"):
        raw_payload = exec_result.dict()
    else:
        raw_payload = exec_result

    safe_result = sanitize_results(raw_payload)

    return {
        "current_execution": execution_to_state_dict(
            safe_result,
            fallback_action_id=get_action_id(action),
            fallback_tool_name=tool_name,
        )
    }