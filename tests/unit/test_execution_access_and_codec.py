import json

from core.execution_access import (
    get_execution_artifacts,
    get_execution_error_code,
    get_execution_message,
    get_execution_payload,
    get_execution_status,
    get_execution_success,
)
from core.execution_codec import execution_to_state_dict, normalize_execution_view
from core.schema import ToolExecutionResult


def test_execution_codec_serializes_tool_execution_result():
    result = ToolExecutionResult(
        execution_id="exec_1",
        action_id="act_1",
        tool_name="get_summary_stats",
        success=True,
        status="ok",
        error_code=None,
        message="Done.",
        recoverable=False,
        payload={"rows": 3},
        artifacts=[],
    )

    payload = execution_to_state_dict(result)

    assert payload["execution_id"] == "exec_1"
    assert payload["action_id"] == "act_1"
    assert payload["tool_name"] == "get_summary_stats"
    assert payload["success"] is True
    assert payload["status"] == "ok"
    assert payload["payload"] == {"rows": 3}

    json.dumps(payload)


def test_execution_codec_normalizes_string_result():
    payload = execution_to_state_dict(
        "Error: No valid action provided.",
        fallback_action_id="act_missing",
        fallback_tool_name="unknown_tool",
    )

    assert payload["action_id"] == "act_missing"
    assert payload["tool_name"] == "unknown_tool"
    assert payload["success"] is False
    assert payload["status"] == "failed"
    assert payload["error_code"] == "NON_STRUCTURED_EXECUTION_RESULT"
    assert payload["payload"] == {"result": "Error: No valid action provided."}

    json.dumps(payload)


def test_execution_access_reads_dict_payload():
    execution = {
        "status": "failed",
        "success": False,
        "error_code": "TOOL_EXECUTION_EXCEPTION",
        "message": "Tool crashed.",
        "payload": {"exception_type": "ValueError"},
        "artifacts": [],
    }

    assert get_execution_status(execution) == "failed"
    assert get_execution_success(execution) is False
    assert get_execution_error_code(execution) == "TOOL_EXECUTION_EXCEPTION"
    assert get_execution_message(execution) == "Tool crashed."
    assert get_execution_payload(execution) == {"exception_type": "ValueError"}
    assert get_execution_artifacts(execution) == []


def test_normalize_execution_view_is_stable():
    execution = {
        "execution_id": "exec_2",
        "action_id": "act_2",
        "tool_name": "run_multiple_regression",
        "success": True,
        "status": "ok",
        "payload": {"r_squared": 0.8},
        "artifacts": [],
    }

    view = normalize_execution_view(execution)

    assert view["execution_id"] == "exec_2"
    assert view["status"] == "ok"
    assert view["success"] is True
    assert view["payload"] == {"r_squared": 0.8}