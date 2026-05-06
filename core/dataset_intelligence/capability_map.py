from __future__ import annotations

from typing import Dict, List

from core.analysis_tool_plugins import PLUGIN_REGISTRY
from core.analysis_tool_plugins.base import ApplicabilityResult
from core.dataset_intelligence.schemas import (
    AnalysisCapability,
    CapabilityMap,
    DatasetProfileV2,
)


def _candidate_columns_for_semantic_types(
    profile: DatasetProfileV2,
    allowed_semantic_types: List[str],
) -> List[str]:
    candidates = []

    for name, col in profile.columns.items():
        if col.semantic_type in allowed_semantic_types:
            candidates.append(name)

    return candidates


def _generic_applicability_from_variable_roles(plugin, profile: DatasetProfileV2) -> ApplicabilityResult:
    """
    Generic fallback when a plugin has variable_roles but no custom checker.

    This is intentionally conservative:
    - Explicit PlanningPolicy.required_user_choices always prevent auto-ready.
    - Required variable roles require user choice unless already selected.
    - Plugins without planning contracts stay conservative.
    """
    planning_policy = getattr(plugin, "planning_policy", None)

    policy_required_choices = []
    if planning_policy is not None:
        policy_required_choices = list(
            getattr(planning_policy, "required_user_choices", []) or []
        )

    if planning_policy and not planning_policy.include_in_capability_map:
        return ApplicabilityResult(
            status="blocked",
            reason="Plugin is excluded from capability-map planning.",
            warnings=["Plugin planning_policy.include_in_capability_map=False."],
        )

    if (
        planning_policy
        and planning_policy.ready_without_user_variables
        and not getattr(plugin, "variable_roles", None)
        and not policy_required_choices
    ):
        return ApplicabilityResult(
            status="ready",
            reason=(
                planning_policy.planning_description
                or "This tool can run without user-selected variables."
            ),
            required_user_choices=[],
            candidate_variables={},
            warnings=[],
        )

    candidate_variables: Dict[str, List[str]] = {}
    required_user_choices: List[str] = list(policy_required_choices)
    warnings: List[str] = []

    if not getattr(plugin, "variable_roles", None):
        return ApplicabilityResult(
            status="needs_user_choice",
            reason=(
                "No variable-role contract is registered, or this plugin requires "
                "non-column user choices before execution."
            ),
            required_user_choices=required_user_choices or ["analysis goal and variables"],
            candidate_variables={},
            warnings=[
                "Plugin is not execution-ready during planning without explicit user choices."
            ],
        )

    for role in plugin.variable_roles:
        candidates = _candidate_columns_for_semantic_types(
            profile,
            role.allowed_semantic_types,
        )

        candidate_variables[role.role_name] = candidates

        if role.required and role.user_must_select:
            if role.role_name not in required_user_choices:
                required_user_choices.append(role.role_name)

        if role.required and not candidates:
            warnings.append(
                f"No candidate variables found for role '{role.role_name}'."
            )

    if warnings:
        return ApplicabilityResult(
            status="not_applicable",
            reason="Required variable roles have no compatible columns.",
            required_user_choices=required_user_choices,
            candidate_variables=candidate_variables,
            warnings=warnings,
        )

    if required_user_choices:
        return ApplicabilityResult(
            status="needs_user_choice",
            reason="Compatible variables exist, but user choices are required before execution.",
            required_user_choices=required_user_choices,
            candidate_variables=candidate_variables,
            warnings=warnings,
        )

    return ApplicabilityResult(
        status="ready",
        reason="Plugin variable-role contract is satisfied.",
        required_user_choices=[],
        candidate_variables=candidate_variables,
        warnings=warnings,
    )


def build_capability_map(
    profile: DatasetProfileV2,
    *,
    plugins=None,
) -> CapabilityMap:
    """
    Build a data-aware capability map from the current dataset profile
    and the unified analysis_tool_plugins contract.

    This is method-generic.
    It does not hard-code regression, chi-square, etc.
    """
    if plugins is None:
        plugins = PLUGIN_REGISTRY

    capabilities: List[AnalysisCapability] = []
    warnings: List[str] = []

    for tool_name, plugin in sorted(plugins.items()):
        if getattr(plugin, "execute", None) is None:
            continue

        checker = getattr(plugin, "applicability_checker", None)

        if checker is not None:
            try:
                result = checker(
                    profile=profile,
                    variables={},
                    mode="planning",
                )
            except Exception as e:
                result = ApplicabilityResult(
                    status="blocked",
                    reason=f"Applicability checker failed: {e}",
                    warnings=[type(e).__name__],
                )
        else:
            result = _generic_applicability_from_variable_roles(plugin, profile)

        capabilities.append(
            AnalysisCapability(
                tool_name=plugin.tool_name,
                display_name=plugin.display_name,
                method_family=getattr(plugin, "method_family", "general"),
                status=result.status,
                reason=result.reason,
                candidate_variables=result.candidate_variables,
                required_user_choices=result.required_user_choices,
                warnings=result.warnings,
                suggested_alternatives=result.suggested_alternatives,
                requires_confirmation=getattr(plugin, "requires_confirmation", False),
                mutates_data=getattr(plugin, "mutates_data", False),
            )
        )

    return CapabilityMap(
        data_version_id=profile.data_version_id,
        capabilities=capabilities,
        warnings=warnings,
    )