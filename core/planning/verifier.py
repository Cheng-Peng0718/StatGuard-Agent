from __future__ import annotations

from typing import Any, Dict, List

from core.analysis_tool_plugins import get_plugin
from core.analysis_tool_plugins.base import ApplicabilityResult
from core.dataset_intelligence.schemas import DatasetProfileV2
from core.planning.schemas import PlanProposal, PlanStep

NO_ROLE_READY_TOOLS = {
    "get_summary_stats",
    "missingness_report",
    "get_correlation_matrix",
}

AUTO_READY_TOOLS = {
    "get_summary_stats",
    "missingness_report",
    "get_correlation_matrix",
}

def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def _column_exists(profile: DatasetProfileV2, column_name: str) -> bool:
    return column_name in profile.columns


def _column_semantic_type(profile: DatasetProfileV2, column_name: str) -> str | None:
    col = profile.columns.get(column_name)
    if col is None:
        return None
    return col.semantic_type


def _verify_variable_roles(step: PlanStep, profile: DatasetProfileV2, plugin) -> PlanStep:
    """
    Generic variable-role verification.

    This does not contain method-specific rules.
    It only checks the plugin's declared VariableRoleSpec contract.
    """
    role_specs = getattr(plugin, "variable_roles", []) or []

    if step.tool_name in AUTO_READY_TOOLS:
        step.required_user_choices = [
            choice
            for choice in (step.required_user_choices or [])
            if choice != "analysis variables"
        ]

        step.warnings = [
            warning
            for warning in (step.warnings or [])
            if "Plugin has no variable role contract" not in warning
        ]

    if not role_specs:
        if (
                step.tool_name in NO_ROLE_READY_TOOLS
                and step.status not in {"blocked", "not_applicable"}
        ):
            step.status = "ready"
            step.execution_ready = True

            step.required_user_choices = [
                choice
                for choice in (step.required_user_choices or [])
                if choice != "analysis variables"
            ]

            step.warnings = [
                warning
                for warning in (step.warnings or [])
                if "Plugin has no variable role contract" not in warning
            ]

            step.arguments = step.arguments or {}

            return step

        if step.status == "ready":
            step.status = "needs_user_choice"
            step.execution_ready = False

            if "analysis variables" not in step.required_user_choices:
                step.required_user_choices.append("analysis variables")

            step.warnings.append(
                "Plugin has no variable role contract; cannot mark step execution-ready from planning."
            )

        return step

    all_roles_ready = True

    for role in role_specs:
        selected = _as_list(step.variables.get(role.role_name))

        if role.required and not selected:
            all_roles_ready = False

            if role.role_name not in step.required_user_choices:
                step.required_user_choices.append(role.role_name)

            continue

        if not selected:
            continue

        if role.max_variables is not None and len(selected) > role.max_variables:
            step.status = "blocked"
            step.execution_ready = False
            step.warnings.append(
                f"Role '{role.role_name}' allows at most {role.max_variables} variable(s)."
            )
            return step

        if len(selected) < role.min_variables:
            all_roles_ready = False

            if role.role_name not in step.required_user_choices:
                step.required_user_choices.append(role.role_name)

            continue

        for col_name in selected:
            if not isinstance(col_name, str):
                step.status = "blocked"
                step.execution_ready = False
                step.warnings.append(
                    f"Role '{role.role_name}' contains a non-string column reference."
                )
                return step

            if not _column_exists(profile, col_name):
                step.status = "blocked"
                step.execution_ready = False
                step.warnings.append(
                    f"Column '{col_name}' does not exist in the current dataset."
                )
                return step

            semantic_type = _column_semantic_type(profile, col_name)

            if role.allowed_semantic_types and semantic_type not in role.allowed_semantic_types:
                step.status = "not_applicable"
                step.execution_ready = False
                step.warnings.append(
                    f"Column '{col_name}' has semantic_type='{semantic_type}', "
                    f"but role '{role.role_name}' requires one of {role.allowed_semantic_types}."
                )
                return step

    if all_roles_ready and not step.warnings:
        step.status = "ready"
        step.execution_ready = True
        step.required_user_choices = []

    else:
        if step.status not in {"blocked", "not_applicable"}:
            step.status = "needs_user_choice"
        step.execution_ready = False

    return step


def verify_plan_step(step: PlanStep, profile: DatasetProfileV2) -> PlanStep:
    """
    Verify a PlanStep against the unified AnalysisToolPlugin contract.

    This function is generic and method-agnostic.
    """
    if not step.tool_name:
        step.status = "blocked"
        step.execution_ready = False
        step.warnings.append("Plan step has no tool_name.")
        return step

    plugin = get_plugin(step.tool_name)

    if plugin is None:
        step.status = "blocked"
        step.execution_ready = False
        step.warnings.append(f"Unknown tool '{step.tool_name}'.")
        return step

    if getattr(plugin, "execute", None) is None:
        step.status = "blocked"
        step.execution_ready = False
        step.warnings.append(f"Tool '{step.tool_name}' is not executable.")
        return step

    checker = getattr(plugin, "applicability_checker", None)

    if checker is not None:
        try:
            result: ApplicabilityResult = checker(
                profile=profile,
                variables=step.variables,
                mode="plan_verification",
            )

            step.status = result.status
            step.execution_ready = result.status == "ready" and not result.required_user_choices
            step.required_user_choices = result.required_user_choices
            step.candidate_variables = {
                **step.candidate_variables,
                **result.candidate_variables,
            }
            step.warnings.extend(result.warnings)
            step.suggested_alternatives = result.suggested_alternatives
            step.applicability_check = result.to_dict()

            return step

        except Exception as e:
            step.status = "blocked"
            step.execution_ready = False
            step.warnings.append(
                f"Applicability checker failed: {type(e).__name__}: {e}"
            )
            return step

    return _verify_variable_roles(step, profile, plugin)


def verify_plan(plan: PlanProposal, profile: DatasetProfileV2) -> PlanProposal:
    verified_steps = []
    blocked_steps = list(plan.blocked_or_not_recommended)

    for step in plan.steps:
        verified = verify_plan_step(step, profile)

        if verified.status in {"blocked", "not_applicable"}:
            blocked_steps.append(verified)
        else:
            verified_steps.append(verified)

    plan.steps = verified_steps
    plan.blocked_or_not_recommended = blocked_steps

    if any(step.status == "needs_user_choice" for step in plan.steps):
        plan.status = "partially_ready"
    elif all(step.execution_ready for step in plan.steps) and plan.steps:
        plan.status = "verified"
    else:
        plan.status = "draft"

    return plan