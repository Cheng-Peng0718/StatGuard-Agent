from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_action_field(action: Any, field_name: str, default: Any = None) -> Any:
    """
    Read an action field from either a Pydantic/object action or a dict action.

    This is a migration helper. It lets graph/controller code stop caring
    whether current_action is currently represented as ActionProposal or as
    a JSON-safe dict.
    """
    if action is None:
        return default

    if isinstance(action, Mapping):
        return action.get(field_name, default)

    return getattr(action, field_name, default)


def get_action_id(action: Any, default: str = "unknown") -> str:
    return get_action_field(action, "action_id", default)


def get_action_type(action: Any, default: str | None = None) -> str | None:
    return get_action_field(action, "action_type", default)


def get_action_tool_name(action: Any, default: str | None = None) -> str | None:
    return get_action_field(action, "tool_name", default)


def get_action_arguments(action: Any) -> dict:
    arguments = get_action_field(action, "arguments", {})

    if arguments is None:
        return {}

    if isinstance(arguments, dict):
        return arguments

    return {}


def get_action_reasoning_summary(
    action: Any,
    default: str = "No reasoning summary provided.",
) -> str:
    return (
        get_action_field(action, "reasoning_summary", None)
        or get_action_field(action, "summary", None)
        or get_action_field(action, "message", None)
        or default
    )


def has_action_tool_name(action: Any) -> bool:
    return bool(get_action_tool_name(action))