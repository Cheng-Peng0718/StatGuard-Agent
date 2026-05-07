from __future__ import annotations

from typing import Any, Dict, List, Literal

from pydantic import BaseModel, Field

from types import SimpleNamespace

from core.graph import (
    advisory_answer_node,
    execute_node,
    execute_pending_plan_node,
    human_review_node,
    intent_router_node,
    plan_only_node,
    summarize_node,
    verify_node,
)
from core.ui_adapter.snapshot import build_ui_snapshot


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


def _get_field(value: Any, field_name: str, default=None):
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(field_name, default)

    return getattr(value, field_name, default)


def _apply_updates(state: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply node updates in the backend controller.

    Important:
    LangGraph uses reducers for some GraphState fields, for example:

        observations: Annotated[list, operator.add]

    App V2 / backend controller does not run through LangGraph's reducer
    machinery, so we must emulate reducer semantics for delta-style fields.

    summarize_node returns observations as a delta:
        {"observations": [new_observation]}

    But analysis_runs is already returned as a full registry:
        existing_analysis_runs + [new_analysis_run]

    Therefore:
    - observations should be appended
    - analysis_runs should be replaced by the returned full list
    """
    merged = dict(state)

    for key, value in (updates or {}).items():
        if key == "observations" and isinstance(value, list):
            existing = list(merged.get("observations") or [])
            merged["observations"] = existing + value
            continue

        merged[key] = value

    return merged


def _verification_status(state: Dict[str, Any]) -> str | None:
    verification = state.get("current_verification")
    return _get_field(verification, "status")


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
    Rehydrate dict current_action into an object-like action for legacy graph nodes.

    BackendTurnResult.model_dump() and UI round-trips may turn ActionProposal
    into a plain dict. Some graph nodes still expect attribute access such as
    action.tool_name / action.arguments.
    """
    if action is None:
        return None

    if not isinstance(action, dict):
        return action

    return SimpleNamespace(
        action_id=action.get("action_id"),
        action_type=action.get("action_type"),
        tool_name=action.get("tool_name"),
        arguments=action.get("arguments") or {},
        reasoning_summary=(
            action.get("reasoning_summary")
            or action.get("summary")
            or action.get("message")
        ),
        task_contract=action.get("task_contract"),
        contract_update=action.get("contract_update"),
    )


def _ensure_graph_action_object(state: Dict[str, Any]) -> Dict[str, Any]:
    action = state.get("current_action")

    if isinstance(action, dict):
        state = dict(state)
        state["current_action"] = _action_to_graph_object(action)

    return state


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