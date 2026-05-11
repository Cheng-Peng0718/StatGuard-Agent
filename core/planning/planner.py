from __future__ import annotations

import uuid
from typing import List

from core.dataset_intelligence.schemas import CapabilityMap, AnalysisCapability
from core.planning.schemas import PlanProposal, PlanStep


def _make_step_id(tool_name: str) -> str:
    safe_name = tool_name.replace(" ", "_").replace("-", "_")
    return f"step_{safe_name}_{uuid.uuid4().hex[:6]}"


def _capability_to_plan_step(capability: AnalysisCapability) -> PlanStep:
    execution_ready = (
        capability.status == "ready"
        and not capability.required_user_choices
    )

    return PlanStep(
        step_id=_make_step_id(capability.tool_name),
        title=capability.display_name,
        tool_name=capability.tool_name,
        method_family=capability.method_family,
        status=capability.status,
        execution_ready=execution_ready,
        purpose=capability.reason,
        rationale=capability.reason,
        variables={},
        arguments={},
        candidate_variables=capability.candidate_variables,
        required_user_choices=capability.required_user_choices,
        applicability_check={
            "status": capability.status,
            "reason": capability.reason,
        },
        warnings=capability.warnings,
        suggested_alternatives=capability.suggested_alternatives,
        requires_confirmation=capability.requires_confirmation,
        mutates_data=capability.mutates_data,
    )


def build_plan_from_capability_map(
    *,
    user_request: str,
    capability_map: CapabilityMap,
    include_not_applicable: bool = True,
) -> PlanProposal:
    """
    Generic plan builder.

    This function does not know about regression, chi-square, t-test, etc.
    It only converts verified data-aware capabilities into a draft plan.
    """
    ready_or_candidate_steps: List[PlanStep] = []
    blocked_steps: List[PlanStep] = []

    for capability in capability_map.capabilities:
        step = _capability_to_plan_step(capability)

        if capability.status in {"ready", "needs_user_choice"}:
            ready_or_candidate_steps.append(step)
        elif include_not_applicable:
            blocked_steps.append(step)

    summary = (
        "Generated a data-aware analysis plan from the current capability map. "
        "No tools have been executed."
    )

    return PlanProposal(
        plan_id=f"plan_{uuid.uuid4().hex[:8]}",
        user_request=user_request,
        data_version_id=capability_map.data_version_id,
        mode="plan_only",
        status="draft",
        summary=summary,
        assumptions=[
            "This plan is based on the current dataset profile and plugin capability contracts.",
            "Steps that require user choices are not execution-ready.",
            "No analysis tools have been executed.",
        ],
        steps=ready_or_candidate_steps,
        blocked_or_not_recommended=blocked_steps,
        requires_user_confirmation_before_execution=True,
        warnings=capability_map.warnings,
    )