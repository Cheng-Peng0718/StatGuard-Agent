from __future__ import annotations

import uuid
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from core.analysis_tool_plugins import get_plugin
from core.analysis_tool_plugins.validation import validate_plugin_action
from core.schema import ActionProposal


TERMINAL_EXECUTION_STATUSES = {
    "running",
    "completed",
    "failed",
    "skipped",
    "blocked",
}


class PlanStepReadiness(BaseModel):
    """
    Single readiness assessment for converting a PlanStep into an ActionProposal.

    This is the only gate that pending-plan execution should trust.
    """
    executable: bool
    status: str
    reason: str

    missing_required_arguments: List[str] = Field(default_factory=list)
    missing_user_choices: List[str] = Field(default_factory=list)

    validation_error_code: Optional[str] = None
    validation_details: Dict[str, Any] = Field(default_factory=dict)

    # Keep this as Any to avoid coupling this Pydantic model to the exact
    # ActionProposal implementation details.
    action: Optional[Any] = None


def _get_required_argument_names(plugin: Any) -> List[str]:
    schema = getattr(plugin, "argument_schema", None)

    if schema is None:
        return []

    required = getattr(schema, "required", {}) or {}

    if isinstance(required, dict):
        return list(required.keys())

    return list(required)


def _get_policy_required_choices(plugin: Any) -> List[str]:
    planning_policy = getattr(plugin, "planning_policy", None)

    if planning_policy is None:
        return []

    return list(getattr(planning_policy, "required_user_choices", []) or [])


def _get_step_arguments(step: Dict[str, Any]) -> Dict[str, Any]:
    arguments = step.get("arguments") or {}

    if not isinstance(arguments, dict):
        return {}

    return arguments


def _missing_keys(arguments: Dict[str, Any], keys: List[str]) -> List[str]:
    missing = []

    for key in keys:
        if key not in arguments or arguments.get(key) is None:
            missing.append(key)

    return missing


def _build_action_from_step(step: Dict[str, Any]) -> ActionProposal:
    return ActionProposal(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        action_type="tool_call",
        tool_name=step["tool_name"],
        arguments=_get_step_arguments(step),
        reasoning_summary=(
            f"Executing verified plan step {step.get('step_id')} "
            f"using tool {step.get('tool_name')}."
        ),
    )


def assess_plan_step_readiness(
    step: Dict[str, Any],
    profile: Any = None,
) -> PlanStepReadiness:
    """
    Decide whether a PlanStep can be converted into an ActionProposal.

    This is method-generic:
    - no clean_data special case
    - no regression special case
    - no ANOVA special case

    All method-specific requirements must come from the unified plugin contract.
    """
    execution_status = step.get("execution_status")

    if execution_status in TERMINAL_EXECUTION_STATUSES:
        return PlanStepReadiness(
            executable=False,
            status="not_executable",
            reason=f"Step already has execution_status={execution_status}.",
        )

    if step.get("status") != "ready":
        return PlanStepReadiness(
            executable=False,
            status="not_ready",
            reason=f"Step status is {step.get('status')}, not ready.",
            missing_user_choices=step.get("required_user_choices", []) or [],
        )

    if step.get("execution_ready") is not True:
        return PlanStepReadiness(
            executable=False,
            status="not_ready",
            reason="Step execution_ready is not True.",
            missing_user_choices=step.get("required_user_choices", []) or [],
        )

    tool_name = step.get("tool_name")

    if not tool_name:
        return PlanStepReadiness(
            executable=False,
            status="blocked",
            reason="Step has no tool_name.",
        )

    plugin = get_plugin(tool_name)

    if plugin is None:
        return PlanStepReadiness(
            executable=False,
            status="blocked",
            reason=f"Unknown tool: {tool_name}.",
        )

    if getattr(plugin, "execute", None) is None:
        return PlanStepReadiness(
            executable=False,
            status="blocked",
            reason=f"Tool {tool_name} is not executable.",
        )

    arguments = _get_step_arguments(step)

    required_arguments = _get_required_argument_names(plugin)
    policy_required_choices = _get_policy_required_choices(plugin)

    missing_required_arguments = _missing_keys(arguments, required_arguments)
    missing_policy_choices = _missing_keys(arguments, policy_required_choices)

    missing_all = sorted(set(missing_required_arguments + missing_policy_choices))

    if missing_all:
        return PlanStepReadiness(
            executable=False,
            status="needs_user_choice",
            reason="Step is missing required execution arguments.",
            missing_required_arguments=missing_required_arguments,
            missing_user_choices=missing_all,
        )

    action = _build_action_from_step(step)

    validation = validate_plugin_action(action, profile=profile)

    if validation.status in {"rejected_recoverable", "rejected_terminal"}:
        return PlanStepReadiness(
            executable=False,
            status="validation_failed",
            reason=validation.feedback,
            validation_error_code=validation.error_code,
            validation_details=validation.details or {},
        )

    return PlanStepReadiness(
        executable=True,
        status=validation.status,
        reason=validation.feedback,
        validation_details=validation.details or {},
        action=action,
    )