from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.schema import ActionProposal


def normalize_action_payload(action: Any) -> dict | None:
    """
    Normalize any supported action representation into a plain dict payload.

    This is for state/checkpoint/UI-safe storage. It does not return a
    Pydantic object.
    """
    if action is None:
        return None

    if isinstance(action, ActionProposal):
        payload = action.model_dump()
    elif isinstance(action, Mapping):
        payload = dict(action)
    else:
        payload = {
            "action_id": getattr(action, "action_id", None),
            "action_type": getattr(action, "action_type", None),
            "tool_name": getattr(action, "tool_name", None),
            "arguments": getattr(action, "arguments", None),
            "reasoning_summary": getattr(action, "reasoning_summary", None),
            "task_contract": getattr(action, "task_contract", None),
            "contract_update": getattr(action, "contract_update", None),
        }

    if not payload.get("reasoning_summary"):
        payload["reasoning_summary"] = (
            payload.get("summary")
            or payload.get("message")
            or "No reasoning summary provided."
        )

    if not payload.get("arguments"):
        payload["arguments"] = {}

    return payload


def action_to_state_dict(action: Any) -> dict | None:
    """
    Convert an action to a JSON-safe dict for state storage.
    """
    payload = normalize_action_payload(action)

    if payload is None:
        return None

    # Validate once, then dump back to a clean dict.
    return ActionProposal.model_validate(payload).model_dump()


def action_from_state(action: Any) -> ActionProposal | None:
    """
    Rehydrate a state/checkpoint action payload into ActionProposal for runtime.

    Runtime graph nodes may still expect a formal action contract while the
    state remains JSON-safe.
    """
    payload = normalize_action_payload(action)

    if payload is None:
        return None

    return ActionProposal.model_validate(payload)