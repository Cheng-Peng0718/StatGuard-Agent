from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PlanStep(BaseModel):
    step_id: str
    title: str

    tool_name: Optional[str] = None
    method_family: str = "general"

    status: str = "needs_user_choice"
    execution_ready: bool = False

    purpose: str = ""
    rationale: str = ""

    variables: Dict[str, Any] = Field(default_factory=dict)
    arguments: Dict[str, Any] = Field(default_factory=dict)

    candidate_variables: Dict[str, List[str]] = Field(default_factory=dict)
    required_user_choices: List[str] = Field(default_factory=list)

    applicability_check: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    suggested_alternatives: List[str] = Field(default_factory=list)

    requires_confirmation: bool = False
    mutates_data: bool = False


class PlanProposal(BaseModel):
    plan_id: str

    user_request: str
    data_version_id: str

    mode: str = "plan_only"
    status: str = "draft"

    summary: str = ""
    assumptions: List[str] = Field(default_factory=list)

    steps: List[PlanStep] = Field(default_factory=list)
    blocked_or_not_recommended: List[PlanStep] = Field(default_factory=list)

    requires_user_confirmation_before_execution: bool = True

    warnings: List[str] = Field(default_factory=list)