from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_execution_field(execution: Any, field_name: str, default: Any = None) -> Any:
    if execution is None:
        return default

    if isinstance(execution, Mapping):
        return execution.get(field_name, default)

    return getattr(execution, field_name, default)


def get_execution_id(execution: Any, default: str | None = None) -> str | None:
    return get_execution_field(execution, "execution_id", default)


def get_execution_action_id(execution: Any, default: str = "unknown") -> str:
    return get_execution_field(execution, "action_id", default) or default


def get_execution_tool_name(execution: Any, default: str | None = None) -> str | None:
    return get_execution_field(execution, "tool_name", default)


def get_execution_status(execution: Any, default: str = "failed") -> str:
    status = get_execution_field(execution, "status", None)
    if status:
        return status

    success = get_execution_success(execution, default=None)
    if success is True:
        return "ok"
    if success is False:
        return "failed"

    return default


def get_execution_success(execution: Any, default: bool | None = None) -> bool | None:
    success = get_execution_field(execution, "success", default)
    if success is None:
        return default
    return bool(success)


def get_execution_error_code(execution: Any, default: str | None = None) -> str | None:
    return get_execution_field(execution, "error_code", default)


def get_execution_message(execution: Any, default: str | None = None) -> str | None:
    return get_execution_field(execution, "message", default)


def get_execution_recoverable(execution: Any, default: bool = False) -> bool:
    return bool(get_execution_field(execution, "recoverable", default))


def get_execution_payload(execution: Any) -> dict:
    payload = get_execution_field(execution, "payload", {})

    if isinstance(payload, dict):
        return dict(payload)

    return {"result": payload}


def get_execution_artifacts(execution: Any) -> list:
    artifacts = get_execution_field(execution, "artifacts", [])

    if isinstance(artifacts, list):
        return artifacts

    return []