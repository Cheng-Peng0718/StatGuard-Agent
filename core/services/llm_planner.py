from __future__ import annotations
import os
import uuid
from typing import Any, Dict, List

from core.dataset_intelligence.schemas import CapabilityMap, DatasetProfileV2
from core.planning.schemas import PlanProposal, PlanStep
from core.planning.verifier import verify_plan

from core.analysis_tool_plugins import PLUGIN_REGISTRY, ensure_plugins_loaded
from core.analysis_tool_plugins.manifest import build_tool_manifests
from core.planning.schemas import PlanProposal
from core.services.llm_planner_contracts import (
    LLMPlanDraft,
    LLMPlannerDatasetView,
    LLMPlannerInput,
    LLMPlannerToolView,
    LLMPlanStepDraft,
)


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


def _column_profile_view(dataset_profile_v2: Dict[str, Any]) -> Dict[str, Any]:
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


def _capability_summary_view(capability_map: Dict[str, Any]) -> List[Dict[str, Any]]:
    capabilities = capability_map.get("capabilities") or []

    if not isinstance(capabilities, list):
        return []

    result = []

    for capability in capabilities:
        if hasattr(capability, "model_dump"):
            capability = capability.model_dump()

        if not isinstance(capability, dict):
            continue

        result.append({
            "tool_name": capability.get("tool_name"),
            "display_name": capability.get("display_name"),
            "method_family": capability.get("method_family"),
            "status": capability.get("status"),
            "reason": capability.get("reason"),
            "required_user_choices": capability.get("required_user_choices", []),
            "requires_confirmation": capability.get("requires_confirmation", False),
            "mutates_data": capability.get("mutates_data", False),
        })

    return result


def build_llm_planner_input(state: Dict[str, Any]) -> LLMPlannerInput:
    dataset_profile_v2 = _as_dict(state.get("dataset_profile_v2"))
    dataset_summary = _as_dict(state.get("dataset_summary"))
    capability_map = _as_dict(state.get("capability_map"))

    ensure_plugins_loaded()
    manifests = build_tool_manifests(dict(PLUGIN_REGISTRY))

    tool_views = [
        LLMPlannerToolView(
            tool_name=manifest.tool_name,
            display_name=manifest.display_name,
            method_family=manifest.method_family,
            supported_goal_types=manifest.supported_goal_types,
            planning_tags=manifest.planning_tags,
            default_plan_purpose=manifest.default_plan_purpose,
            argument_schema=manifest.argument_schema,
            variable_roles=manifest.variable_roles,
            task_argument_bindings=manifest.task_argument_bindings,
            required_planning_choices=manifest.required_planning_choices,
            requires_confirmation=manifest.requires_confirmation,
            mutates_data=manifest.mutates_data,
            expected_deliverables=manifest.expected_deliverables,
        )
        for manifest in manifests.values()
    ]

    dataset_view = LLMPlannerDatasetView(
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
        columns=_column_profile_view(dataset_profile_v2),
        dataset_summary=dataset_summary,
        capability_summary=_capability_summary_view(capability_map),
    )

    return LLMPlannerInput(
        user_request=state.get("user_request", ""),
        interaction_intent=state.get("interaction_intent", ""),
        task_spec=_as_dict(state.get("task_spec")) or None,
        dataset=dataset_view,
        tools=tool_views,
        planner_instructions=[
            "Use the dataset profile and tool manifests to propose a statistically appropriate plan.",
            "Do not invent tools. Every executable step must use a tool_name from the provided tools list.",
            "Do not execute tools. Return a plan only.",
            "If required variables or operation choices are missing, mark them as required_user_choices.",
            "Mutating tools require explicit user confirmation before execution.",
        ],
    )

def _planner_system_prompt() -> str:
    return (
        "You are a senior data analysis planning agent. "
        "Your job is to create a statistically appropriate analysis plan, not to execute tools. "
        "You must use only tool names provided in the tool manifest list. "
        "If a required variable, argument, or operation choice is missing, include it in required_user_choices. "
        "Do not invent tools, columns, files, or results. "
        "Mutating tools must not be marked execution-ready unless the user explicitly requested the mutation; "
        "they still require confirmation downstream. "
        "Return only the structured plan draft matching the provided schema."
    )

def _make_structured_planner_llm():
    from langchain_openai import ChatOpenAI

    model = (
        os.getenv("ANALYSIS_AGENT_PLANNER_MODEL")
        or os.getenv("OPENAI_MODEL")
        or "gpt-4.1-mini"
    )

    temperature_raw = os.getenv("ANALYSIS_AGENT_PLANNER_TEMPERATURE", "0")

    try:
        temperature = float(temperature_raw)
    except ValueError:
        temperature = 0.0

    return ChatOpenAI(
        model=model,
        temperature=temperature,
    ).with_structured_output(
        LLMPlanDraft,
        method="function_calling",
    )

def _planner_user_prompt(planner_input: LLMPlannerInput) -> str:
    return (
        "Create a data analysis plan from this planner input.\n\n"
        f"{planner_input.model_dump_json(indent=2)}"
    )

def generate_llm_plan_draft(
    planner_input: LLMPlannerInput,
) -> LLMPlanDraft:
    """
    Generate a structured LLM plan draft.

    This function is the only place in the planner service that calls the LLM.
    Tests should monkeypatch this function rather than calling external APIs.
    """
    structured_llm = _make_structured_planner_llm()

    result = structured_llm.invoke([
        ("system", _planner_system_prompt()),
        ("user", _planner_user_prompt(planner_input)),
    ])

    return LLMPlanDraft.model_validate(result)

def _make_step_id(tool_name: str | None) -> str:
    safe_name = (tool_name or "non_tool_step").replace(" ", "_").replace("-", "_")
    return f"step_{safe_name}_{uuid.uuid4().hex[:6]}"


def _manifest_index_by_tool() -> Dict[str, Any]:
    ensure_plugins_loaded()
    manifests = build_tool_manifests(dict(PLUGIN_REGISTRY))
    return dict(manifests)


def _capability_index_by_tool(capability_map: CapabilityMap) -> Dict[str, Any]:
    return {
        capability.tool_name: capability
        for capability in capability_map.capabilities
    }


def _merge_unique_strings(*values: List[str]) -> List[str]:
    result = []

    for items in values:
        for item in items or []:
            if isinstance(item, str) and item and item not in result:
                result.append(item)

    return result


def _sanitize_arguments(arguments: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in dict(arguments or {}).items()
        if value is not None and value != "" and value != []
    }


def _draft_step_to_plan_step(
    draft_step: LLMPlanStepDraft,
    *,
    capability_map: CapabilityMap,
) -> PlanStep:
    manifests = _manifest_index_by_tool()
    capabilities = _capability_index_by_tool(capability_map)

    tool_name = draft_step.tool_name

    manifest = manifests.get(tool_name) if tool_name else None
    capability = capabilities.get(tool_name) if tool_name else None

    title = draft_step.title
    if not title and capability is not None:
        title = capability.display_name
    if not title and manifest is not None:
        title = manifest.display_name
    if not title:
        title = tool_name or "Planner note"

    method_family = "general"
    if capability is not None:
        method_family = capability.method_family
    elif manifest is not None:
        method_family = manifest.method_family

    requires_confirmation = False
    mutates_data = False
    if manifest is not None:
        requires_confirmation = manifest.requires_confirmation
        mutates_data = manifest.mutates_data
    elif capability is not None:
        requires_confirmation = capability.requires_confirmation
        mutates_data = capability.mutates_data

    capability_status = capability.status if capability is not None else "not_applicable"

    status = draft_step.status or capability_status
    if status not in {
        "needs_user_choice",
        "ready",
        "blocked",
        "not_applicable",
        "not_recommended",
        "completed",
        "failed",
    }:
        status = capability_status

    required_user_choices = _merge_unique_strings(
        draft_step.required_user_choices,
        capability.required_user_choices if capability is not None else [],
    )

    execution_ready = (
        tool_name is not None
        and capability is not None
        and status == "ready"
        and not required_user_choices
        and not requires_confirmation
    )

    purpose = draft_step.purpose or (
        manifest.default_plan_purpose
        if manifest is not None
        else ""
    )

    rationale = draft_step.rationale or purpose

    warnings = _merge_unique_strings(
        draft_step.warnings,
        capability.warnings if capability is not None else [],
    )

    expected_deliverables = _merge_unique_strings(
        draft_step.expected_deliverables,
        manifest.expected_deliverables if manifest is not None else [],
    )

    return PlanStep(
        step_id=_make_step_id(tool_name),
        title=title,
        purpose=purpose,
        goal=purpose,
        rationale=rationale,
        tool_name=tool_name,
        method_family=method_family,
        status=status,
        execution_ready=execution_ready,
        variables=_sanitize_arguments(draft_step.variables),
        arguments=_sanitize_arguments(draft_step.arguments),
        candidate_variables=(
            capability.candidate_variables
            if capability is not None
            else {}
        ),
        required_user_choices=required_user_choices,
        applicability_check=(
            {
                "status": capability.status,
                "reason": capability.reason,
            }
            if capability is not None
            else {
                "status": "not_applicable",
                "reason": "Tool is not available in the current capability map.",
            }
        ),
        warnings=warnings,
        suggested_alternatives=(
            capability.suggested_alternatives
            if capability is not None
            else []
        ),
        expected_deliverables=expected_deliverables,
        requires_confirmation=requires_confirmation,
        mutates_data=mutates_data,
    )


def normalize_llm_plan_draft(
    *,
    draft: LLMPlanDraft,
    state: Dict[str, Any],
) -> PlanProposal:
    """
    Normalize an LLM-produced plan draft into the canonical PlanProposal contract.

    The LLM may propose a plan, but this function owns:
    - PlanProposal construction;
    - PlanStep construction;
    - capability/tool metadata merge;
    - execution readiness;
    - final verify_plan call.
    """
    capability_map = CapabilityMap.model_validate(state.get("capability_map"))
    dataset_profile = DatasetProfileV2.model_validate(state.get("dataset_profile_v2"))

    steps = [
        _draft_step_to_plan_step(
            step,
            capability_map=capability_map,
        )
        for step in draft.steps
    ]

    blocked_steps = [
        step
        for step in steps
        if step.status in {
            "blocked",
            "not_applicable",
            "not_recommended",
        }
    ]

    active_steps = [
        step
        for step in steps
        if step.status not in {
            "blocked",
            "not_applicable",
            "not_recommended",
        }
    ]

    plan = PlanProposal(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        user_goal=draft.user_goal,
        user_request=state.get("user_request", ""),
        task_spec=None,
        data_version_id=capability_map.data_version_id,
        mode="plan_only",
        status="draft",
        summary=draft.summary or "Generated an LLM-planned analysis plan.",
        assumptions=draft.assumptions,
        warnings=draft.warnings,
        steps=active_steps,
        blocked_or_not_recommended=blocked_steps,
        requires_user_confirmation_before_execution=True,
    )

    return verify_plan(plan, dataset_profile)


def create_llm_plan_from_state(state: Dict[str, Any]) -> PlanProposal:
    """
    LLM-first planner service boundary.

    Active implementation:
    - builds the LLM planner input contract;
    - asks the structured LLM planner for an LLMPlanDraft;
    - normalizes the draft into the canonical PlanProposal;
    - verifies the plan before returning it.
    """
    planner_input = build_llm_planner_input(state)
    draft = generate_llm_plan_draft(planner_input)

    return normalize_llm_plan_draft(
        draft=draft,
        state=state,
    )