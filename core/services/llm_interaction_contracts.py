from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LLMInteractionDatasetView(BaseModel):
    dataset_available: bool = False
    dataset_name: str = ""
    data_version_id: str = ""
    n_rows: Optional[int] = None
    n_cols: Optional[int] = None
    columns: Dict[str, Any] = Field(default_factory=dict)


class LLMInteractionDraft(BaseModel):
    """
    Structured LLM output for interpreting a user message.

    This is not a tool call. It only decides interaction intent and an optional
    high-level TaskSpec draft.
    """
    intent: str = Field(
        default="unknown",
        description=(
            "One of: advisory, plan_analysis, direct_analysis, "
            "execute_plan, modify_data, clarification, unknown"
        ),
    )
    user_goal: str = ""
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)

    clarification_question: str = ""

    goal_type: str = "analysis_recommendation"
    target_variables: List[str] = Field(default_factory=list)
    predictor_variables: List[str] = Field(default_factory=list)
    grouping_variables: List[str] = Field(default_factory=list)
    requested_methods: List[str] = Field(default_factory=list)

    constraints: Dict[str, Any] = Field(default_factory=dict)
    rationale: str = ""