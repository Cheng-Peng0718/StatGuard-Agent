from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from core.planning.readiness import (
    PlanStepReadiness,
    assess_plan_step_readiness,
)


def find_next_executable_step(
    pending_plan: Dict[str, Any],
    profile: Any = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[PlanStepReadiness]]:
    """
    Return the next truly executable plan step.

    This function does NOT trust step["execution_ready"] by itself.
    Every candidate step must pass PlanStepReadinessGate.
    """
    steps = pending_plan.get("steps", []) or []

    first_blocker: Optional[PlanStepReadiness] = None

    for step in steps:
        assessment = assess_plan_step_readiness(
            step=step,
            profile=profile,
        )

        if assessment.executable:
            return step, assessment

        if first_blocker is None:
            first_blocker = assessment

    return None, first_blocker


def mark_plan_step_started(
    pending_plan: Dict[str, Any],
    step_id: str,
    action_id: str,
) -> Dict[str, Any]:
    plan = dict(pending_plan)
    steps = []

    for step in plan.get("steps", []) or []:
        step = dict(step)

        if step.get("step_id") == step_id:
            step["execution_status"] = "running"
            step["action_id"] = action_id

        steps.append(step)

    plan["steps"] = steps
    plan["status"] = "executing"

    return plan


def mark_plan_step_after_execution(
    pending_plan: Dict[str, Any],
    step_id: str,
    *,
    success: bool,
    execution_id: str | None = None,
    message: str | None = None,
) -> Dict[str, Any]:
    plan = dict(pending_plan)
    steps = []

    for step in plan.get("steps", []) or []:
        step = dict(step)

        if step.get("step_id") == step_id:
            step["execution_status"] = "completed" if success else "failed"
            step["last_execution_id"] = execution_id
            step["last_execution_message"] = message

        steps.append(step)

    plan["steps"] = steps

    remaining_ready = []

    for step in steps:
        if step.get("execution_status") in {
            "running",
            "completed",
            "failed",
            "skipped",
            "blocked",
        }:
            continue

        if step.get("status") == "ready" and step.get("execution_ready") is True:
            remaining_ready.append(step)

    if remaining_ready:
        plan["status"] = "partially_executed"
    else:
        plan["status"] = "completed" if success else "partially_failed"

    return plan