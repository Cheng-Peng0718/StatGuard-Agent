import json

from core.controller.backend_turn import (
    _ensure_graph_action_object,
    _finish,
    _normalize_state_actions_for_storage,
)
from core.schema import ActionProposal


def _make_action(action_id: str = "act_1") -> ActionProposal:
    return ActionProposal(
        action_id=action_id,
        action_type="tool_call",
        tool_name="get_summary_stats",
        arguments={"columns": ["GPA"]},
        reasoning_summary="Compute summary statistics.",
    )


def test_finish_serializes_current_action_before_returning_state():
    action = _make_action()

    result = _finish(
        state={
            "current_action": action,
            "user_request": "do summary stats",
            "messages": [],
        },
        node_trace=[],
    )

    stored_action = result.state["current_action"]

    assert isinstance(stored_action, dict)
    assert stored_action["action_id"] == "act_1"
    assert stored_action["tool_name"] == "get_summary_stats"

    json.dumps(result.state)


def test_storage_normalizer_serializes_pending_action_too():
    action = _make_action("act_pending")

    state = _normalize_state_actions_for_storage({
        "pending_action": action,
    })

    assert isinstance(state["pending_action"], dict)
    assert state["pending_action"]["action_id"] == "act_pending"

    json.dumps(state)


def test_graph_action_object_rehydrates_current_and_pending_actions():
    action = _make_action()

    stored = {
        "current_action": action.model_dump(),
        "pending_action": action.model_dump(),
    }

    runtime_state = _ensure_graph_action_object(stored)

    assert isinstance(runtime_state["current_action"], ActionProposal)
    assert isinstance(runtime_state["pending_action"], ActionProposal)
    assert runtime_state["current_action"].tool_name == "get_summary_stats"