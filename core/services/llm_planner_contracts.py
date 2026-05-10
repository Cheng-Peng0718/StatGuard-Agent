from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMPlannerToolView(BaseModel):
    tool_name: str
    display_name: str = ""
    method_family: str = "general"

    supported_goal_types: List[str] = Field(default_factory=list)
    planning_tags: List[str] = Field(default_factory=list)
    default_plan_purpose: str = ""

    argument_schema: Dict[str, Any] = Field(default_factory=dict)
    variable_roles: List[Dict[str, Any]] = Field(default_factory=list)
    task_argument_bindings: List[Dict[str, Any]] = Field(default_factory=list)
    required_planning_choices: List[str] = Field(default_factory=list)

    requires_confirmation: bool = False
    mutates_data: bool = False
    expected_deliverables: List[str] = Field(default_factory=list)


class LLMPlannerDatasetView(BaseModel):
    dataset_name: str = ""
    data_version_id: str = ""
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    columns: Dict[str, Any] = Field(default_factory=dict)
    dataset_summary: Dict[str, Any] = Field(default_factory=dict)
    capability_summary: List[Dict[str, Any]] = Field(default_factory=list)


class LLMPlannerInput(BaseModel):
    user_request: str = ""
    interaction_intent: str = ""
    task_spec: Optional[Dict[str, Any]] = None

    dataset: LLMPlannerDatasetView
    tools: List[LLMPlannerToolView] = Field(default_factory=list)

    planner_instructions: List[str] = Field(default_factory=list)


class LLMPlanStepDraft(BaseModel):
    title: str = ""
    tool_name: Optional[str] = None
    purpose: str = ""
    rationale: str = ""

    arguments: Dict[str, Any] = Field(default_factory=dict)
    variables: Dict[str, Any] = Field(default_factory=dict)

    required_user_choices: List[str] = Field(default_factory=list)
    expected_deliverables: List[str] = Field(default_factory=list)


class LLMPlanDraft(BaseModel):
    user_goal: str = ""
    summary: str = ""
    assumptions: List[str] = Field(default_factory=list)
    steps: List[LLMPlanStepDraft] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)