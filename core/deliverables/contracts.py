from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TaskContract(BaseModel):
    """
    Normalized backend contract for final-answer deliverable gating.

    This is not a UI schema.
    It tells DeliverableGate what evidence must exist before a final answer
    is allowed.
    """
    required_tools: List[str] = Field(default_factory=list)
    required_artifacts: List[str] = Field(default_factory=list)
    required_deliverables: List[str] = Field(default_factory=list)

    success_criteria: List[str] = Field(default_factory=list)

    allow_partial: bool = False

    metadata: Dict[str, Any] = Field(default_factory=dict)


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return [value]

    if isinstance(value, list):
        return [str(v) for v in value if v is not None]

    if isinstance(value, tuple):
        return [str(v) for v in value if v is not None]

    return []


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


def normalize_task_contract(value: Any) -> TaskContract:
    """
    Convert legacy/free-form task_contract values into a normalized TaskContract.

    Supported legacy keys:
    - required_tools
    - required_artifacts
    - required_deliverables
    - success_criteria
    - allow_partial
    """
    data = _as_dict(value)

    if not data:
        return TaskContract()

    return TaskContract(
        required_tools=_as_list(data.get("required_tools")),
        required_artifacts=_as_list(data.get("required_artifacts")),
        required_deliverables=_as_list(data.get("required_deliverables")),
        success_criteria=_as_list(data.get("success_criteria")),
        allow_partial=bool(data.get("allow_partial", False)),
        metadata={
            k: v
            for k, v in data.items()
            if k not in {
                "required_tools",
                "required_artifacts",
                "required_deliverables",
                "success_criteria",
                "allow_partial",
            }
        },
    )