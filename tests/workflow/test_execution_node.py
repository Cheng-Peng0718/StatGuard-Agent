from core.workflow.nodes.execution import execute_node


def test_execute_node_returns_structured_blocked_execution_without_action():
    result = execute_node({
        "current_action": None,
    })

    execution = result["current_execution"]

    assert isinstance(execution, dict)
    assert execution["status"] == "blocked"
    assert execution["success"] is False
    assert execution["error_code"] == "NO_VALID_ACTION"
    assert execution["tool_name"] == "unknown_tool"