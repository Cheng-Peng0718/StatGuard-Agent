from __future__ import annotations

import json
import os
from typing import Any, Dict

from core.domain.intent import IntentDecision
from core.domain.task import TaskSpec
from core.services.llm_interaction_contracts import (
    LLMInteractionDatasetView,
    LLMInteractionDraft,
)


_ALLOWED_INTENTS = {
    "advisory",
    "plan_analysis",
    "direct_analysis",
    "execute_plan",
    "modify_data",
    "clarification",
    "unknown",
}


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        if isinstance(dumped, dict):
            return dict(dumped)

    return {}


def _column_view(dataset_profile_v2: Dict[str, Any]) -> Dict[str, Any]:
    columns = dataset_profile_v2.get("columns") or {}

    if not isinstance(columns, dict):
        return {}

    result = {}

    for name, profile in columns.items():
        if hasattr(profile, "model_dump"):
            profile = profile.model_dump()

        if not isinstance(profile, dict):
            continue

        result[str(name)] = {
            "semantic_type": profile.get("semantic_type"),
            "dtype": profile.get("dtype"),
            "n_missing": profile.get("n_missing"),
            "missing_rate": profile.get("missing_rate"),
            "n_unique": profile.get("n_unique"),
            "examples": profile.get("examples", []),
        }

    return result


def build_llm_interaction_input(
    user_message: str,
    *,
    state: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    state = dict(state or {})
    dataset_profile_v2 = _as_dict(state.get("dataset_profile_v2"))
    dataset_summary = _as_dict(state.get("dataset_summary"))

    dataset_view = LLMInteractionDatasetView(
        dataset_available=bool(dataset_profile_v2),
        dataset_name=(
            dataset_profile_v2.get("dataset_name")
            or state.get("dataset_name")
            or ""
        ),
        data_version_id=(
            dataset_profile_v2.get("data_version_id")
            or state.get("active_data_version_id")
            or ""
        ),
        n_rows=dataset_summary.get("n_rows"),
        n_cols=dataset_summary.get("n_cols"),
        columns=_column_view(dataset_profile_v2),
    )

    return {
        "user_message": user_message or "",
        "dataset": dataset_view.model_dump(),
        "existing_pending_plan": state.get("pending_plan"),
        "allowed_intents": sorted(_ALLOWED_INTENTS),
        "instructions": [
            "Classify the user's interaction intent.",
            "Do not choose a concrete tool. Produce a high-level TaskSpec only.",
            "Use dataset columns only if they are present in the dataset view.",
            "If the user asks to run an existing plan, use intent execute_plan.",
            "If the user asks for a plan or recommendations, use intent plan_analysis.",
            "If the user asks to modify data, use intent modify_data.",
            "If the user asks for a direct analysis, use intent direct_analysis.",
            "If the request is vague, use intent clarification or unknown.",
        ],
    }


def _interaction_system_prompt() -> str:
    return (
        "You are an interaction parser for a data analysis agent. "
        "You do not execute tools. "
        "You classify the user's message and produce a structured intent/task draft. "
        "Return only the structured output matching the schema."
    )


def _interaction_user_prompt(payload: Dict[str, Any]) -> str:
    return (
        "Parse this user interaction for the analysis agent.\n\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def _make_structured_interaction_llm():
    from langchain_openai import ChatOpenAI

    model = (
        os.getenv("ANALYSIS_AGENT_INTERACTION_MODEL")
        or os.getenv("ANALYSIS_AGENT_PLANNER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )

    temperature_raw = os.getenv("ANALYSIS_AGENT_INTERACTION_TEMPERATURE", "0")

    try:
        temperature = float(temperature_raw)
    except ValueError:
        temperature = 0.0

    return ChatOpenAI(
        model=model,
        temperature=temperature,
    ).with_structured_output(
        LLMInteractionDraft,
        method="function_calling",
    )


def generate_llm_interaction_draft(
    payload: Dict[str, Any],
) -> LLMInteractionDraft:
    structured_llm = _make_structured_interaction_llm()

    result = structured_llm.invoke([
        ("system", _interaction_system_prompt()),
        ("user", _interaction_user_prompt(payload)),
    ])

    return LLMInteractionDraft.model_validate(result)


def normalize_interaction_draft(
    draft: LLMInteractionDraft,
    *,
    user_message: str,
) -> IntentDecision:
    intent = draft.intent

    if intent not in _ALLOWED_INTENTS:
        intent = "unknown"

    task_spec = None

    if intent in {
        "plan_analysis",
        "direct_analysis",
        "modify_data",
    }:
        task_spec = TaskSpec(
            goal_type=draft.goal_type or "analysis_recommendation",
            user_goal=draft.user_goal or user_message,
            source_user_request=user_message,
            target_variables=draft.target_variables,
            predictor_variables=draft.predictor_variables,
            grouping_variables=draft.grouping_variables,
            requested_methods=draft.requested_methods,
            constraints=draft.constraints,
            confidence=draft.confidence,
        )

    return IntentDecision(
        intent=intent,
        confidence=draft.confidence,
        reason=draft.rationale or draft.clarification_question,
        task_spec=task_spec,
        should_execute=intent in {"direct_analysis", "modify_data"},
    )


def legacy_interaction_intent_from_decision(decision: IntentDecision) -> str:
    if decision.intent == "advisory":
        return "advisory"

    if decision.intent == "plan_analysis":
        return "plan_only"

    if decision.intent == "execute_plan":
        return "execute_plan"

    if decision.intent in {"direct_analysis", "modify_data"}:
        return "direct_tool"

    return "unknown"


def decide_llm_interaction_intent(
    user_message: str,
    *,
    state: Dict[str, Any] | None = None,
) -> IntentDecision:
    payload = build_llm_interaction_input(
        user_message,
        state=state,
    )
    draft = generate_llm_interaction_draft(payload)

    return normalize_interaction_draft(
        draft,
        user_message=user_message,
    )