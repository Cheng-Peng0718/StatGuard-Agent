from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional
from typing import Literal

AssistantResponseType = Literal[
    "dataset_loaded",
    "advisory",
    "plan",
    "plan_step_choices_updated",
    "plan_execution_status",
    "final_answer",
    "clarification",
    "error",
]

from pydantic import BaseModel, Field


class AssistantResponse(BaseModel):
    response_id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4().hex[:8]}")

    response_type: AssistantResponseType

    content: str
    source_node: str

    data_version_id: Optional[str] = None
    plan_id: Optional[str] = None
    plan_status: Optional[str] = None

    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)


def make_assistant_response(
    *,
    response_type: AssistantResponseType,
    content: str,
    source_node: str,
    data_version_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    plan_status: Optional[str] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    response = AssistantResponse(
        response_type=response_type,
        content=content,
        source_node=source_node,
        data_version_id=data_version_id,
        plan_id=plan_id,
        plan_status=plan_status,
        artifacts=artifacts or [],
        metadata=metadata or {},
    )

    return response.model_dump()

def make_response_update(
    *,
    response_type: AssistantResponseType,
    content: str,
    source_node: str,
    data_version_id: Optional[str] = None,
    plan_id: Optional[str] = None,
    plan_status: Optional[str] = None,
    artifacts: Optional[List[Dict[str, Any]]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Canonical graph-state update for user-visible assistant responses.

    New graph nodes should return assistant_response through this helper,
    not final_answer.
    """
    return {
        "assistant_response": make_assistant_response(
            response_type=response_type,
            content=content,
            source_node=source_node,
            data_version_id=data_version_id,
            plan_id=plan_id,
            plan_status=plan_status,
            artifacts=artifacts,
            metadata=metadata,
        )
    }