from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


class StateSerializationIssue(BaseModel):
    severity: Literal["warning", "error"]
    code: str
    message: str
    path: str
    details: Dict[str, Any] = Field(default_factory=dict)


class StateSerializationAuditResult(BaseModel):
    status: Literal["ok", "warning", "error"]
    issues: List[StateSerializationIssue] = Field(default_factory=list)
    safe_state: Dict[str, Any] = Field(default_factory=dict)

    @property
    def has_errors(self) -> bool:
        return any(issue.severity == "error" for issue in self.issues)

    @property
    def has_warnings(self) -> bool:
        return any(issue.severity == "warning" for issue in self.issues)


def _type_name(value: Any) -> str:
    return f"{type(value).__module__}.{type(value).__name__}"


def _append_issue(
    issues: List[StateSerializationIssue],
    *,
    severity: Literal["warning", "error"],
    code: str,
    message: str,
    path: str,
    details: Optional[Dict[str, Any]] = None,
):
    issues.append(
        StateSerializationIssue(
            severity=severity,
            code=code,
            message=message,
            path=path,
            details=details or {},
        )
    )


def to_jsonable(
    value: Any,
    *,
    path: str = "$",
    issues: Optional[List[StateSerializationIssue]] = None,
) -> Any:
    """
    Convert backend state values into JSON-safe values.

    This is intentionally conservative:
    - Pydantic models are normalized via model_dump()
    - dataclasses are normalized via asdict()
    - Path objects become strings
    - tuples/sets become lists
    - unsupported custom objects become repr(...) and emit an error
    """
    if issues is None:
        issues = []

    if value is None or isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, Path):
        _append_issue(
            issues,
            severity="warning",
            code="PATH_NORMALIZED",
            message="Path object was converted to string.",
            path=path,
            details={"type": _type_name(value)},
        )
        return str(value)

    if hasattr(value, "model_dump"):
        _append_issue(
            issues,
            severity="warning",
            code="PYDANTIC_MODEL_NORMALIZED",
            message="Pydantic model was converted with model_dump().",
            path=path,
            details={"type": _type_name(value)},
        )

        dumped = value.model_dump()
        return to_jsonable(dumped, path=path, issues=issues)

    if dataclasses.is_dataclass(value):
        _append_issue(
            issues,
            severity="warning",
            code="DATACLASS_NORMALIZED",
            message="Dataclass object was converted with dataclasses.asdict().",
            path=path,
            details={"type": _type_name(value)},
        )

        return to_jsonable(dataclasses.asdict(value), path=path, issues=issues)

    if isinstance(value, dict):
        result = {}

        for key, item in value.items():
            safe_key = str(key)
            result[safe_key] = to_jsonable(
                item,
                path=f"{path}.{safe_key}",
                issues=issues,
            )

        return result

    if isinstance(value, list):
        return [
            to_jsonable(item, path=f"{path}[{idx}]", issues=issues)
            for idx, item in enumerate(value)
        ]

    if isinstance(value, tuple):
        _append_issue(
            issues,
            severity="warning",
            code="TUPLE_NORMALIZED",
            message="Tuple was converted to list.",
            path=path,
            details={"type": _type_name(value)},
        )

        return [
            to_jsonable(item, path=f"{path}[{idx}]", issues=issues)
            for idx, item in enumerate(value)
        ]

    if isinstance(value, set):
        _append_issue(
            issues,
            severity="warning",
            code="SET_NORMALIZED",
            message="Set was converted to list.",
            path=path,
            details={"type": _type_name(value)},
        )

        return [
            to_jsonable(item, path=f"{path}[{idx}]", issues=issues)
            for idx, item in enumerate(sorted(value, key=str))
        ]

    if hasattr(value, "isoformat"):
        try:
            _append_issue(
                issues,
                severity="warning",
                code="ISOFORMAT_OBJECT_NORMALIZED",
                message="Object with isoformat() was converted to string.",
                path=path,
                details={"type": _type_name(value)},
            )
            return value.isoformat()
        except Exception:
            pass

    _append_issue(
        issues,
        severity="error",
        code="UNSUPPORTED_CUSTOM_OBJECT",
        message="Unsupported custom object cannot be safely serialized.",
        path=path,
        details={
            "type": _type_name(value),
            "repr": repr(value),
        },
    )

    return repr(value)


def audit_state_serialization(state: Any) -> StateSerializationAuditResult:
    """
    Backend-only GraphState serialization audit.

    This does not mutate graph state and does not alter routing.
    It checks whether state can be converted into a JSON/checkpoint-safe dict.
    """
    issues: List[StateSerializationIssue] = []

    safe_state = to_jsonable(state, path="$", issues=issues)

    try:
        json.dumps(safe_state)
    except TypeError as exc:
        _append_issue(
            issues,
            severity="error",
            code="JSON_DUMPS_FAILED",
            message="JSON serialization failed after normalization.",
            path="$",
            details={"error": str(exc)},
        )

    if any(issue.severity == "error" for issue in issues):
        status = "error"
    elif issues:
        status = "warning"
    else:
        status = "ok"

    return StateSerializationAuditResult(
        status=status,
        issues=issues,
        safe_state=safe_state if isinstance(safe_state, dict) else {"value": safe_state},
    )


def make_checkpoint_safe_state(state: Any) -> Dict[str, Any]:
    """
    Return a JSON-safe copy of state.

    This is the helper future checkpoint/UI layers should use before persisting
    or rendering backend state.
    """
    result = audit_state_serialization(state)
    return result.safe_state