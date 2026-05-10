from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.execution_access import (
    get_execution_action_id,
    get_execution_artifacts,
    get_execution_error_code,
    get_execution_id,
    get_execution_message,
    get_execution_payload,
    get_execution_recoverable,
    get_execution_status,
    get_execution_success,
    get_execution_tool_name,
)


def execution_to_state_dict(
    execution: Any,
    *,
    fallback_action_id: str = "unknown",
    fallback_tool_name: str | None = None,
) -> dict | None:
    """
    Convert execution runtime object/dict/string into JSON-safe state payload.

    ToolExecutionResult is the canonical runtime result. This codec also accepts
    dicts or non-structured values so state, repair, and reporting boundaries stay
    stable.
    """
    if execution is None:
        return None

    if isinstance(execution, Mapping):
        payload = dict(execution)
    elif hasattr(execution, "model_dump"):
        payload = execution.model_dump()
    elif hasattr(execution, "dict"):
        payload = execution.dict()
    else:
        payload = {
            "execution_id": None,
            "action_id": fallback_action_id,
            "tool_name": fallback_tool_name,
            "success": False,
            "status": "failed",
            "error_code": "NON_STRUCTURED_EXECUTION_RESULT",
            "message": str(execution),
            "recoverable": True,
            "payload": {"result": execution},
            "artifacts": [],
        }

    if not payload.get("action_id"):
        payload["action_id"] = fallback_action_id

    if not payload.get("tool_name"):
        payload["tool_name"] = fallback_tool_name

    if "success" not in payload or payload.get("success") is None:
        status = payload.get("status")
        payload["success"] = status in {"ok", "warning"}

    if not payload.get("status"):
        payload["status"] = "ok" if payload.get("success") else "failed"

    if payload.get("payload") is None:
        payload["payload"] = {}

    if not isinstance(payload.get("payload"), dict):
        payload["payload"] = {"result": payload.get("payload")}

    if payload.get("artifacts") is None:
        payload["artifacts"] = []

    if not isinstance(payload.get("artifacts"), list):
        payload["artifacts"] = []

    return payload


def normalize_execution_view(
    execution: Any,
    *,
    fallback_action_id: str = "unknown",
    fallback_tool_name: str | None = None,
) -> dict | None:
    """
    Return a normalized execution dict suitable for summarize/UI/report logic.
    """
    payload = execution_to_state_dict(
        execution,
        fallback_action_id=fallback_action_id,
        fallback_tool_name=fallback_tool_name,
    )

    if payload is None:
        return None

    return {
        "execution_id": get_execution_id(payload),
        "action_id": get_execution_action_id(payload, fallback_action_id),
        "tool_name": get_execution_tool_name(payload, fallback_tool_name),
        "success": get_execution_success(payload, default=False),
        "status": get_execution_status(payload),
        "error_code": get_execution_error_code(payload),
        "message": get_execution_message(payload),
        "recoverable": get_execution_recoverable(payload),
        "payload": get_execution_payload(payload),
        "artifacts": get_execution_artifacts(payload),
    }