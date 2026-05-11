from __future__ import annotations

import uuid
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from core.analysis_tool_plugins import get_plugin


class RepairAttempt(BaseModel):
    """
    Immutable record of one attempted repair.

    This is an audit record, not an executable action.
    """
    repair_attempt_id: str
    source_action_id: Optional[str] = None
    source_tool_name: Optional[str] = None

    repair_status: Literal[
        "proposed",
        "applied",
        "succeeded",
        "failed",
        "skipped",
        "terminal",
    ] = "proposed"

    decision_status: Optional[str] = None
    error_code: Optional[str] = None
    reason: Optional[str] = None

    repair_type: Literal[
        "argument_repair",
        "method_fallback",
        "ask_user",
        "none",
    ] = "none"

    proposed_arguments: Dict[str, Any] = Field(default_factory=dict)
    proposed_tool_name: Optional[str] = None

    message: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class RepairAttemptLog(BaseModel):
    attempts: List[RepairAttempt] = Field(default_factory=list)


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


def _get_plugin_max_attempts(tool_name: Optional[str]) -> int:
    if not tool_name:
        return 0

    plugin = get_plugin(tool_name)

    if plugin is None:
        return 0

    repair_policy = getattr(plugin, "repair_policy", None)

    if repair_policy is None:
        return 0

    return int(getattr(repair_policy, "max_attempts", 0) or 0)


def normalize_repair_attempts(value: Any) -> List[Dict[str, Any]]:
    if value is None:
        return []

    if isinstance(value, RepairAttemptLog):
        return [attempt.model_dump() for attempt in value.attempts]

    if isinstance(value, list):
        return [_as_dict(item) for item in value]

    if isinstance(value, dict):
        attempts = value.get("attempts")

        if isinstance(attempts, list):
            return [_as_dict(item) for item in attempts]

    return []


def count_repair_attempts_for_action(
    repair_attempts: Any,
    *,
    source_action_id: Optional[str],
) -> int:
    attempts = normalize_repair_attempts(repair_attempts)

    if not source_action_id:
        return 0

    return sum(
        1
        for attempt in attempts
        if attempt.get("source_action_id") == source_action_id
    )


def can_attempt_repair(
    *,
    repair_decision: Any,
    repair_attempts: Any,
    current_action: Any,
) -> bool:
    """
    Decide whether another repair attempt is allowed.

    This does not create or apply a repair.
    """
    decision = _as_dict(repair_decision)
    decision_status = decision.get("status")

    if decision_status not in {"repairable", "needs_user"}:
        return False

    tool_name = decision.get("tool_name") or _get_field(current_action, "tool_name")
    action_id = _get_field(current_action, "action_id")

    max_attempts = _get_plugin_max_attempts(tool_name)

    if max_attempts <= 0:
        return False

    used_attempts = count_repair_attempts_for_action(
        repair_attempts,
        source_action_id=action_id,
    )

    return used_attempts < max_attempts


def make_repair_attempt(
    *,
    repair_decision: Any,
    current_action: Any,
    repair_type: str = "none",
    proposed_arguments: Optional[Dict[str, Any]] = None,
    proposed_tool_name: Optional[str] = None,
    message: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a repair attempt record from current action + repair decision.

    This function only records intent. It does not mutate the action or execute tools.
    """
    decision = _as_dict(repair_decision)

    attempt = RepairAttempt(
        repair_attempt_id=f"repair_{uuid.uuid4().hex[:8]}",
        source_action_id=_get_field(current_action, "action_id"),
        source_tool_name=decision.get("tool_name") or _get_field(current_action, "tool_name"),
        repair_status="proposed",
        decision_status=decision.get("status"),
        error_code=decision.get("error_code"),
        reason=decision.get("reason"),
        repair_type=repair_type,
        proposed_arguments=proposed_arguments or {},
        proposed_tool_name=proposed_tool_name,
        message=message,
        metadata=metadata or {},
    )

    return attempt.model_dump()


def append_repair_attempt(
    repair_attempts: Any,
    attempt: Any,
) -> List[Dict[str, Any]]:
    attempts = normalize_repair_attempts(repair_attempts)
    attempts.append(_as_dict(attempt))
    return attempts