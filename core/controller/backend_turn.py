from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from core.workflow.nodes.interaction import (
    intent_router_node,
    advisory_answer_node,
)
from core.workflow.nodes.planning import plan_only_node
from core.workflow.nodes.plan_execution import execute_pending_plan_node
from core.workflow.nodes.verification import verify_node
from core.workflow.nodes.human_review import human_review_node
from core.workflow.nodes.execution import execute_node
from core.workflow.nodes.summarization import summarize_node
from core.workflow.nodes.finalization import (
    deliverable_gate_node,
    final_response_node,
)


from core.ui_adapter.snapshot import build_ui_snapshot

from core.action_codec import action_from_state, action_to_state_dict

from core.verification_access import get_verification_status
from core.verification_codec import verification_to_state_dict
from core.execution_codec import execution_to_state_dict

class BackendTurnResult(BaseModel):
    """
    Result of one backend-controller turn.

    This is the future UI-facing backend controller result.
    It is not a graph node and does not mutate input state in place.
    """
    status: Literal[
        "ok",
        "needs_review",
        "blocked",
        "error",
    ] = "ok"

    state: Dict[str, Any]
    ui_snapshot: Dict[str, Any]
    node_trace: List[str] = Field(default_factory=list)
    message: str | None = None


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}

_ACTION_STATE_FIELDS = ("current_action", "pending_action")
_VERIFICATION_STATE_FIELDS = ("current_verification",)
_EXECUTION_STATE_FIELDS = ("current_execution",)

def _get_field(value: Any, field_name: str, default=None):
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(field_name, default)

    return getattr(value, field_name, default)


def _apply_updates(state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    summarize_node returns observations and analysis_runs as deltas.
    Therefore, both observations and analysis_runs should be appended.
    """
    merged = dict(state)

    for key, value in (updates or {}).items():
        if key in {"observations", "analysis_runs"} and isinstance(value, list):
            existing = list(merged.get(key) or [])
            merged[key] = existing + value
            continue

        merged[key] = value

    return merged


def _verification_status(state: Dict[str, Any]) -> str | None:
    verification = state.get("current_verification")
    return get_verification_status(verification)

def _normalize_state_executions_for_storage(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert runtime execution objects back into JSON-safe dict payloads before
    returning state to UI/checkpoint boundaries.
    """
    normalized = dict(state)

    for field_name in _EXECUTION_STATE_FIELDS:
        if field_name in normalized:
            normalized[field_name] = execution_to_state_dict(
                normalized.get(field_name)
            )

    return normalized

def _has_current_action(state: Dict[str, Any]) -> bool:
    action = state.get("current_action")
    return action is not None and _get_field(action, "tool_name") is not None


def _finish(
    *,
    state: Dict[str, Any],
    node_trace: List[str],
    status: str = "ok",
    message: str | None = None,
) -> BackendTurnResult:
    state = _normalize_state_actions_for_storage(state)
    state = _normalize_state_verifications_for_storage(state)
    state = _normalize_state_executions_for_storage(state)

    ui_snapshot = build_ui_snapshot(state)

    # Store the latest snapshot for UI layers that want to keep it in state.
    state = dict(state)
    state["ui_snapshot"] = ui_snapshot

    return BackendTurnResult(
        status=status,
        state=state,
        ui_snapshot=ui_snapshot,
        node_trace=node_trace,
        message=message,
    )

def _action_to_graph_object(action: Any):
    """
    Rehydrate state/checkpoint action payload into the canonical runtime
    ActionProposal contract.

    The state boundary should remain JSON-safe, while legacy graph nodes can
    still receive a formal ActionProposal during the migration period.
    """
    return action_from_state(action)


def _ensure_graph_action_object(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Rehydrate JSON-safe action payloads into ActionProposal objects for
    legacy graph-node runtime execution.
    """
    state = dict(state)

    for field_name in _ACTION_STATE_FIELDS:
        action = state.get(field_name)

        if isinstance(action, dict):
            state[field_name] = _action_to_graph_object(action)

    return state

def _normalize_state_actions_for_storage(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert runtime action objects back into JSON-safe dict payloads before
    returning state to UI/checkpoint boundaries.
    """
    normalized = dict(state)

    for field_name in _ACTION_STATE_FIELDS:
        if field_name in normalized:
            normalized[field_name] = action_to_state_dict(normalized.get(field_name))

    return normalized

def _normalize_state_verifications_for_storage(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert runtime verification objects back into JSON-safe dict payloads before
    returning state to UI/checkpoint boundaries.
    """
    normalized = dict(state)

    for field_name in _VERIFICATION_STATE_FIELDS:
        if field_name in normalized:
            normalized[field_name] = verification_to_state_dict(
                normalized.get(field_name)
            )

    return normalized


def _run_verify_execute_summarize(
    state: Dict[str, Any],
    *,
    node_trace: List[str],
) -> tuple[Dict[str, Any], str, str | None]:
    """
    Run the standard action path:

    current_action
      -> verify_node
      -> if allowed: execute_node -> summarize_node
      -> if needs_review: stop and return needs_review
      -> if rejected: stop and return blocked

    This function does not override safety gates.
    """
    state = _ensure_graph_action_object(state)

    if not _has_current_action(state):
        return state, "blocked", "No current action is available."

    updates = verify_node(state)
    node_trace.append("verify_node")
    state = _apply_updates(state, updates)

    status = _verification_status(state)

    if status == "needs_review":
        return state, "needs_review", "Human review is required before execution."

    if status != "allowed":
        return state, "blocked", f"Verification did not allow execution: {status}"

    state = _ensure_graph_action_object(state)

    updates = execute_node(state)
    node_trace.append("execute_node")
    state = _apply_updates(state, updates)

    updates = summarize_node(state)
    node_trace.append("summarize_node")
    state = _apply_updates(state, updates)

    return state, "ok", None


def _handle_human_review_decision(
    state: Dict[str, Any],
    *,
    node_trace: List[str],
) -> tuple[Dict[str, Any], str, str | None]:
    """
    Handle a pending human-review decision.

    Approval:
      human_review_node -> execute_node -> summarize_node

    Rejection:
      human_review_node only; no execution.
    """
    state = _ensure_graph_action_object(state)
    decision = state.get("human_review_decision")

    if decision not in {"approved", "rejected"}:
        return state, "ok", None

    state = _ensure_graph_action_object(state)

    updates = human_review_node(state)
    node_trace.append("human_review_node")
    state = _apply_updates(state, updates)

    if decision == "rejected":
        return state, "blocked", "Human review rejected the pending action."

    status = _verification_status(state)

    if status != "allowed":
        return state, "blocked", f"Human review did not produce an allowed verification: {status}"

    state = _ensure_graph_action_object(state)

    updates = execute_node(state)
    node_trace.append("execute_node")
    state = _apply_updates(state, updates)

    updates = summarize_node(state)
    node_trace.append("summarize_node")
    state = _apply_updates(state, updates)

    return state, "ok", None



def run_backend_turn(state: Any) -> Dict[str, Any]:
    """
    Run one backend turn from the current state.

    Intended future UI usage:

        updates = apply_ui_event_to_state(state, event)
        state.update(updates)
        result = run_backend_turn(state)
        state = result["state"]
        snapshot = result["ui_snapshot"]

    This controller:
    - does not import Streamlit
    - does not import app.py
    - does not bypass verification
    - does not bypass human review
    - does not directly call LLMs
    - does not mutate the input state in place
    """
    current_state = dict(state) if isinstance(state, dict) else _as_dict(state)
    node_trace: List[str] = []

    try:
        latest_event = current_state.get("latest_ui_event") or {}
        latest_event_type = latest_event.get("event_type")

        if latest_event_type == "update_plan_step_choices":
            return _finish(
                state=current_state,
                node_trace=[],
                status="ok",
                message="Plan step choices were updated.",
            ).model_dump()

        # 1. Human-review continuation path.
        if current_state.get("human_review_decision") in {"approved", "rejected"}:
            current_state, status, message = _handle_human_review_decision(
                current_state,
                node_trace=node_trace,
            )
            return _finish(
                state=current_state,
                node_trace=node_trace,
                status=status,
                message=message,
            ).model_dump()

        # 2. Normal user-request path.
        updates = intent_router_node(current_state)
        node_trace.append("intent_router_node")
        current_state = _apply_updates(current_state, updates)

        intent = current_state.get("interaction_intent")

        if intent == "advisory":
            updates = advisory_answer_node(current_state)
            node_trace.append("advisory_answer_node")
            current_state = _apply_updates(current_state, updates)

            return _finish(
                state=current_state,
                node_trace=node_trace,
                status="ok",
            ).model_dump()

        if intent == "plan_only":
            updates = plan_only_node(current_state)
            node_trace.append("plan_only_node")
            current_state = _apply_updates(current_state, updates)

            return _finish(
                state=current_state,
                node_trace=node_trace,
                status="ok",
            ).model_dump()

        if intent == "execute_plan":
            updates = execute_pending_plan_node(current_state)
            node_trace.append("execute_pending_plan_node")
            current_state = _apply_updates(current_state, updates)

            if not _has_current_action(current_state):
                return _finish(
                    state=current_state,
                    node_trace=node_trace,
                    status="blocked",
                    message="No executable plan step is available.",
                ).model_dump()

            current_state, status, message = _run_verify_execute_summarize(
                current_state,
                node_trace=node_trace,
            )

            return _finish(
                state=current_state,
                node_trace=node_trace,
                status=status,
                message=message,
            ).model_dump()

        return _finish(
            state=current_state,
            node_trace=node_trace,
            status="blocked",
            message=f"Unsupported interaction intent: {intent}",
        ).model_dump()

    except Exception as exc:
        current_state = dict(current_state)
        current_state["controller_error"] = {
            "error_type": type(exc).__name__,
            "message": str(exc),
        }

        return _finish(
            state=current_state,
            node_trace=node_trace,
            status="error",
            message=str(exc),
        ).model_dump()