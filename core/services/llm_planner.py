from __future__ import annotations

from typing import Any, Dict, List

from core.analysis_tool_plugins import PLUGIN_REGISTRY, ensure_plugins_loaded
from core.analysis_tool_plugins.manifest import build_tool_manifests
from core.planning.schemas import PlanProposal
from core.services.intelligent_planner import create_plan_from_state
from core.services.llm_planner_contracts import (
    LLMPlannerDatasetView,
    LLMPlannerInput,
    LLMPlannerToolView,
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


def create_llm_plan_from_state(state: Dict[str, Any]) -> PlanProposal:
    """
    LLM-first planner service boundary.

    Current implementation:
    - builds the LLM planner input contract;
    - temporarily delegates plan generation to deterministic fallback;
    - future implementation will call an LLM and normalize the structured output.
    """
    build_llm_planner_input(state)
    return create_plan_from_state(state)