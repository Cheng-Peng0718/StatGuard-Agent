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
            arguments=action.arguments
        )

        print(f"Running tool: {tool_name}, arguments: {action.arguments}")
        result_payload = tool_spec.func(context)
        normalized = normalize_tool_payload(result_payload)

        return ToolExecutionResult(
            execution_id=execution_id,
            action_id=action.action_id,
            tool_name=tool_name,
            success=normalized["success"],
            status=normalized["status"],
            error_code=normalized.get("error_code"),
            message=normalized.get("message"),
            recoverable=normalized.get("recoverable", False),
            payload=normalized.get("payload", {}),
            artifacts=normalized.get("artifacts", []),
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