import uuid
from typing import Any, Dict

from core.schema import ActionProposal, ToolExecutionResult, AgentContext
from core.analysis_tool_plugins import get_plugin


def _normalize_tool_result_payload(result_payload: Any) -> Dict[str, Any]:
    if not isinstance(result_payload, dict):
        return {
            "status": "ok",
            "message": "Tool returned a non-dict result.",
            "recoverable": False,
            "details": {"result": result_payload},
            "artifacts": [],
        }

    return result_payload


def _payload_from_result_payload(result_payload: Dict[str, Any]) -> Dict[str, Any]:
    details = result_payload.get("details", {})

    if not isinstance(details, dict):
        details = {"details": details}

    payload = dict(details)

    for key in [
        "data_version_update",
        "audit",
        "suggested_next_actions",
    ]:
        if key in result_payload:
            payload[key] = result_payload[key]

    return payload


def execute_analysis_tool(action: ActionProposal, context_pkg) -> ToolExecutionResult:
    """
    Unified execution entrypoint.

    Only core.analysis_tool_plugins is authoritative for tool execution.
    """
    execution_id = f"exec_{uuid.uuid4().hex[:8]}"
    tool_name = action.tool_name

    try:
        plugin = get_plugin(tool_name)

        if plugin is None:
            raise ValueError(
                f"Tool '{tool_name}' is not registered in analysis_tool_plugins."
            )

        if plugin.execute is None:
            raise ValueError(
                f"Tool '{tool_name}' has no execute function."
            )

        workspace_dir = getattr(context_pkg, "workspace_dir", "./")

        context = AgentContext(
            workspace_dir=workspace_dir,
            arguments=action.arguments,
            data_versions=getattr(context_pkg, "data_versions", []) or [],
            active_data_version_id=getattr(context_pkg, "active_data_version_id", None),
            data_audit_log=getattr(context_pkg, "data_audit_log", []) or [],
        )

        print(f"Running plugin tool: {tool_name}, arguments: {action.arguments}")

        result_payload = plugin.run(context)
        result_payload = _normalize_tool_result_payload(result_payload)

        status = result_payload.get("status", "ok")
        success = status in {"ok", "warning"}
        error_code = result_payload.get("error_code")
        message = result_payload.get("message")
        recoverable = result_payload.get("recoverable", False)
        artifacts = result_payload.get("artifacts", []) or []
        payload = _payload_from_result_payload(result_payload)

        return ToolExecutionResult(
            execution_id=execution_id,
            action_id=action.action_id,
            tool_name=tool_name,
            success=success,
            status=status,
            error_code=error_code,
            message=message,
            recoverable=recoverable,
            payload=payload,
            artifacts=artifacts,
        )

    except Exception as e:
        print(f"❌ Plugin execution crashed: {str(e)}")

        return ToolExecutionResult(
            execution_id=execution_id,
            action_id=getattr(action, "action_id", "unknown"),
            tool_name=tool_name,
            success=False,
            status="failed",
            error_code="TOOL_EXECUTION_EXCEPTION",
            message=f"Tool execution crashed: {str(e)}",
            recoverable=True,
            payload={
                "error_message": str(e),
                "exception_type": type(e).__name__,
            },
            artifacts=[],
        )