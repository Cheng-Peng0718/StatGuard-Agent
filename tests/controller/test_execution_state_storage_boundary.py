import json

from core.controller.backend_turn import (
    _finish,
    _normalize_state_executions_for_storage,
)
from core.schema import ToolExecutionResult


def _make_execution():
    return ToolExecutionResult(
        execution_id="exec_1",
        action_id="act_1",
        tool_name="get_summary_stats",
        success=True,
        status="ok",
        message="Done.",
        payload={"rows": 3},
        artifacts=[],
    )


def test_finish_serializes_current_execution_before_returning_state():
    result = _finish(
        state={
            "current_execution": _make_execution(),
            "user_request": "do summary stats",
            "messages": [],
        },
        node_trace=[],
    )

    stored = result.state["current_execution"]

    assert isinstance(stored, dict)
    assert stored["execution_id"] == "exec_1"
    assert stored["tool_name"] == "get_summary_stats"

    json.dumps(result.state)


def test_execution_storage_normalizer_handles_string_result():
    state = _normalize_state_executions_for_storage({
        "current_execution": "Error: No valid action provided.",
    })

    assert isinstance(state["current_execution"], dict)
    assert state["current_execution"]["status"] == "failed"
    assert state["current_execution"]["error_code"] == "NON_STRUCTURED_EXECUTION_RESULT"

    json.dumps(state)