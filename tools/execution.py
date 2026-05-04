import uuid
from core.schema import ActionProposal, ToolExecutionResult, AgentContext
from tools.registry import registry
from tools.result_utils import normalize_tool_payload


def execute_tool(action: ActionProposal, context_pkg) -> ToolExecutionResult:
    execution_id = f"exec_{uuid.uuid4().hex[:8]}"
    tool_name = action.tool_name

    try:
        if tool_name not in registry.tools:
            raise ValueError(f"Tool '{tool_name}' is not registered.")

        tool_spec = registry.tools[tool_name]

        workspace_dir = getattr(context_pkg, "workspace_dir", "./")

        context = AgentContext(
            workspace_dir=workspace_dir,
            arguments=action.arguments,
            data_versions=getattr(context_pkg, "data_versions", []) or [],
            active_data_version_id=getattr(context_pkg, "active_data_version_id", None),
            data_audit_log=getattr(context_pkg, "data_audit_log", []) or [],
        )

        print(f"Running tool: {tool_name}, arguments: {action.arguments}")
        result_payload = tool_spec.func(context)

        if not isinstance(result_payload, dict):
            result_payload = {
                "status": "ok",
                "message": "Tool returned a non-dict result.",
                "details": {"result": result_payload},
            }

        status = result_payload.get("status", "ok")
        success = status in {"ok", "warning"}
        error_code = result_payload.get("error_code")
        message = result_payload.get("message")
        recoverable = result_payload.get("recoverable", False)
        artifacts = result_payload.get("artifacts", []) or []

        details = result_payload.get("details", {})
        if not isinstance(details, dict):
            details = {"details": details}

        payload = dict(details)

        # Preserve important top-level tool-return fields.
        for key in [
            "data_version_update",
            "audit",
            "suggested_next_actions",
        ]:
            if key in result_payload:
                payload[key] = result_payload[key]

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
        print(f"❌ Tool execution crashed: {str(e)}")
        return ToolExecutionResult(
            execution_id=execution_id,
            action_id=action.action_id,
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