from __future__ import annotations

import uuid
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


UI_EVENT_SCHEMA_VERSION = "ui_event_v1"


class UIEvent(BaseModel):
    """
    Stable event contract from UI to backend.

    UI should send these events instead of directly mutating GraphState.
    """
    event_id: str = Field(default_factory=lambda: f"ui_evt_{uuid.uuid4().hex[:8]}")
    schema_version: str = UI_EVENT_SCHEMA_VERSION

    event_type: Literal[
        "user_message",
        "approve_human_review",
        "reject_human_review",
        "run_plan",
        "cancel_plan",
        "select_plan_step",
        "update_plan_step_choices",
        "clear_runtime",
    ]

    payload: Dict[str, Any] = Field(default_factory=dict)
    metadata: Dict[str, Any] = Field(default_factory=dict)


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


def normalize_ui_event(event: Any) -> Dict[str, Any]:
    """
    Normalize a raw UI event into the canonical UIEvent dict.

    This does not mutate backend state.
    """
    if isinstance(event, UIEvent):
        return event.model_dump()

    event_dict = _as_dict(event)

    if not event_dict:
        raise ValueError("Invalid UI event: expected dict-like event.")

    normalized = UIEvent(
        event_id=event_dict.get("event_id") or f"ui_evt_{uuid.uuid4().hex[:8]}",
        schema_version=event_dict.get("schema_version") or UI_EVENT_SCHEMA_VERSION,
        event_type=event_dict["event_type"],
        payload=event_dict.get("payload") or {},
        metadata=event_dict.get("metadata") or {},
    )

    return normalized.model_dump()


def make_user_message_event(message: str, *, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return UIEvent(
        event_type="user_message",
        payload={
            "message": message,
        },
        metadata=metadata or {},
    ).model_dump()


def make_approve_human_review_event(
    *,
    action_hash: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {}

    if action_hash:
        payload["action_hash"] = action_hash

    return UIEvent(
        event_type="approve_human_review",
        payload=payload,
        metadata=metadata or {},
    ).model_dump()


def make_reject_human_review_event(
    *,
    reason: Optional[str] = None,
    action_hash: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {}

    if reason:
        payload["reason"] = reason

    if action_hash:
        payload["action_hash"] = action_hash

    return UIEvent(
        event_type="reject_human_review",
        payload=payload,
        metadata=metadata or {},
    ).model_dump()


def make_run_plan_event(*, metadata: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    return UIEvent(
        event_type="run_plan",
        payload={},
        metadata=metadata or {},
    ).model_dump()


def make_cancel_plan_event(
    *,
    reason: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    payload = {}

    if reason:
        payload["reason"] = reason

    return UIEvent(
        event_type="cancel_plan",
        payload=payload,
        metadata=metadata or {},
    ).model_dump()

def make_update_plan_step_choices_event(
    *,
    step_id: str,
    choices: Dict[str, Any],
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not step_id:
        raise ValueError("step_id is required.")

    if not isinstance(choices, dict):
        raise TypeError("choices must be a dictionary.")

    return UIEvent(
        event_type="update_plan_step_choices",
        payload={
            "step_id": step_id,
            "choices": choices,
        },
        metadata=metadata or {},
    ).model_dump()

def _is_empty_choice(value: Any) -> bool:
    if value is None:
        return True

    if isinstance(value, str):
        return not value.strip()

    if isinstance(value, list):
        return len(value) == 0

    return False


def _apply_plan_step_choices(
    *,
    pending_plan: Dict[str, Any],
    step_id: str,
    choices: Dict[str, Any],
) -> Dict[str, Any]:
    plan = {
        **pending_plan,
        "steps": [
            dict(step)
            for step in (pending_plan.get("steps") or [])
        ],
    }

    matched = False

    for step in plan["steps"]:
        if step.get("step_id") != step_id:
            continue

        matched = True

        variables = dict(step.get("variables") or {})
        arguments = dict(step.get("arguments") or {})

        for key, value in choices.items():
            if _is_empty_choice(value):
                continue

            variables[key] = value
            arguments[key] = value

        required = list(step.get("required_user_choices") or [])
        remaining = [
            choice
            for choice in required
            if _is_empty_choice(variables.get(choice))
        ]

        step["variables"] = variables
        step["arguments"] = arguments
        step["required_user_choices"] = remaining

        if not remaining:
            step["status"] = "ready"
            step["execution_ready"] = True

            if step.get("execution_status") in {None, "None"}:
                step["execution_status"] = "not_started"
        else:
            step["status"] = "needs_user_choice"
            step["execution_ready"] = False

        break

    if not matched:
        raise ValueError(f"No plan step found for step_id={step_id}")

    return plan


def apply_ui_event_to_state(state: Any, event: Any) -> Dict[str, Any]:
    """
    Convert a normalized UI event into GraphState updates.

    This function does not execute graph nodes.
    It only prepares state updates for the backend graph to consume.
    """
    event_dict = normalize_ui_event(event)

    event_type = event_dict["event_type"]
    payload = event_dict.get("payload") or {}

    if event_type == "user_message":
        message = payload.get("message")

        if not isinstance(message, str) or not message.strip():
            raise ValueError("user_message event requires non-empty payload.message.")

        return {
            "user_request": message.strip(),
            "latest_ui_event": event_dict,
        }

    if event_type == "run_plan":
        return {
            "user_request": "run the plan",
            "latest_ui_event": event_dict,
        }

    if event_type == "approve_human_review":
        return {
            "human_review_decision": "approved",
            "human_review_action_hash": payload.get("action_hash"),
            "latest_ui_event": event_dict,
        }

    if event_type == "reject_human_review":
        return {
            "human_review_decision": "rejected",
            "human_review_rejection_reason": payload.get("reason"),
            "human_review_action_hash": payload.get("action_hash"),
            "latest_ui_event": event_dict,
        }

    if event_type == "cancel_plan":
        return {
            "plan_status": "cancelled",
            "plan_execution_status": "cancelled",
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
            "current_plan_step_id": None,
            "action_origin": None,
            "latest_ui_event": event_dict,
        }

    if event_type == "select_plan_step":
        step_id = payload.get("step_id")

        if not step_id:
            raise ValueError("select_plan_step event requires payload.step_id.")

        return {
            "selected_plan_step_id": step_id,
            "latest_ui_event": event_dict,
        }

    if event_type == "update_plan_step_choices":
        step_id = payload.get("step_id")
        choices = payload.get("choices") or {}

        if not step_id:
            raise ValueError("update_plan_step_choices event requires payload.step_id.")

        if not isinstance(choices, dict):
            raise TypeError("update_plan_step_choices payload.choices must be a dict.")

        state_dict = _as_dict(state)
        pending_plan = state_dict.get("pending_plan")

        if not pending_plan:
            raise ValueError("Cannot update plan step choices because no pending_plan exists.")

        updated_plan = _apply_plan_step_choices(
            pending_plan=pending_plan,
            step_id=step_id,
            choices=choices,
        )

        return {
            "pending_plan": updated_plan,
            "latest_ui_event": event_dict,
            "assistant_response": {
                "response_type": "plan_step_choices_updated",
                "content": (
                    "Plan step choices were updated. "
                    "You can click Run plan to continue."
                ),
                "source_node": "ui_event_adapter",
                "metadata": {
                    "step_id": step_id,
                },
            },
        }

    if event_type == "clear_runtime":
        return {
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
            "current_plan_step_id": None,
            "action_origin": None,
            "latest_ui_event": event_dict,
        }

    raise ValueError(f"Unsupported UI event type: {event_type}")