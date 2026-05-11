from __future__ import annotations

import uuid
from typing import Any, Dict, Literal, Optional

from pydantic import BaseModel, Field


class RepairProposal(BaseModel):
    """
    A proposed repair for a failed/rejected action.

    This is not executed directly. It must be verified before use.
    """
    repair_proposal_id: str

    source_action_id: Optional[str] = None
    source_tool_name: Optional[str] = None

    proposal_type: Literal[
        "argument_repair",
        "method_fallback",
        "ask_user",
        "no_op",
    ]

    proposed_tool_name: Optional[str] = None
    proposed_arguments: Dict[str, Any] = Field(default_factory=dict)

    reason: str
    risk_level: Literal["low", "medium", "high"] = "medium"

    requires_user: bool = False
    requires_confirmation: bool = False

    source_error_code: Optional[str] = None
    source_repair_decision_status: Optional[str] = None

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


def _get_field(value: Any, field_name: str, default=None):
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(field_name, default)

    return getattr(value, field_name, default)


def make_no_op_repair_proposal(
    *,
    repair_decision: Any,
    current_action: Any,
    reason: str,
) -> Dict[str, Any]:
    decision = _as_dict(repair_decision)

    proposal = RepairProposal(
        repair_proposal_id=f"repair_prop_{uuid.uuid4().hex[:8]}",
        source_action_id=_get_field(current_action, "action_id"),
        source_tool_name=decision.get("tool_name") or _get_field(current_action, "tool_name"),
        proposal_type="no_op",
        proposed_tool_name=None,
        proposed_arguments={},
        reason=reason,
        risk_level="low",
        requires_user=False,
        requires_confirmation=False,
        source_error_code=decision.get("error_code"),
        source_repair_decision_status=decision.get("status"),
        metadata={
            "proposal_source": "backend_deterministic",
        },
    )

    return proposal.model_dump()


def make_ask_user_repair_proposal(
    *,
    repair_decision: Any,
    current_action: Any,
    prompt: str,
    missing_fields: Optional[list[str]] = None,
) -> Dict[str, Any]:
    decision = _as_dict(repair_decision)

    proposal = RepairProposal(
        repair_proposal_id=f"repair_prop_{uuid.uuid4().hex[:8]}",
        source_action_id=_get_field(current_action, "action_id"),
        source_tool_name=decision.get("tool_name") or _get_field(current_action, "tool_name"),
        proposal_type="ask_user",
        proposed_tool_name=None,
        proposed_arguments={},
        reason=prompt,
        risk_level="low",
        requires_user=True,
        requires_confirmation=False,
        source_error_code=decision.get("error_code"),
        source_repair_decision_status=decision.get("status"),
        metadata={
            "proposal_source": "backend_deterministic",
            "missing_fields": missing_fields or [],
        },
    )

    return proposal.model_dump()


def make_argument_repair_proposal(
    *,
    repair_decision: Any,
    current_action: Any,
    proposed_arguments: Dict[str, Any],
    reason: str,
    requires_confirmation: bool = False,
    risk_level: str = "medium",
) -> Dict[str, Any]:
    decision = _as_dict(repair_decision)

    proposal = RepairProposal(
        repair_proposal_id=f"repair_prop_{uuid.uuid4().hex[:8]}",
        source_action_id=_get_field(current_action, "action_id"),
        source_tool_name=decision.get("tool_name") or _get_field(current_action, "tool_name"),
        proposal_type="argument_repair",
        proposed_tool_name=_get_field(current_action, "tool_name"),
        proposed_arguments=proposed_arguments or {},
        reason=reason,
        risk_level=risk_level,
        requires_user=False,
        requires_confirmation=requires_confirmation,
        source_error_code=decision.get("error_code"),
        source_repair_decision_status=decision.get("status"),
        metadata={
            "proposal_source": "backend_deterministic",
        },
    )

    return proposal.model_dump()


def make_method_fallback_repair_proposal(
    *,
    repair_decision: Any,
    current_action: Any,
    fallback_tool_name: str,
    proposed_arguments: Dict[str, Any],
    reason: str,
    requires_confirmation: bool = False,
    risk_level: str = "medium",
) -> Dict[str, Any]:
    decision = _as_dict(repair_decision)

    proposal = RepairProposal(
        repair_proposal_id=f"repair_prop_{uuid.uuid4().hex[:8]}",
        source_action_id=_get_field(current_action, "action_id"),
        source_tool_name=decision.get("tool_name") or _get_field(current_action, "tool_name"),
        proposal_type="method_fallback",
        proposed_tool_name=fallback_tool_name,
        proposed_arguments=proposed_arguments or {},
        reason=reason,
        risk_level=risk_level,
        requires_user=False,
        requires_confirmation=requires_confirmation,
        source_error_code=decision.get("error_code"),
        source_repair_decision_status=decision.get("status"),
        metadata={
            "proposal_source": "backend_deterministic",
        },
    )

    return proposal.model_dump()