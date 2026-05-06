from __future__ import annotations

from core.planning.schemas import PlanProposal, PlanStep


def _render_step(step: PlanStep, index: int) -> str:
    lines = []

    lines.append(f"{index}. {step.title}")
    lines.append(f"   Status: {step.status}")

    if step.purpose:
        lines.append(f"   Purpose: {step.purpose}")

    if step.execution_ready:
        lines.append("   Execution: ready, but not run yet.")
    else:
        lines.append("   Execution: not ready to run yet.")

    if step.required_user_choices:
        choices = ", ".join(step.required_user_choices)
        lines.append(f"   Needs your choice: {choices}")

    if step.candidate_variables:
        lines.append("   Candidate variables:")

        for role, cols in step.candidate_variables.items():
            preview = ", ".join(cols[:8])
            if len(cols) > 8:
                preview += ", ..."
            lines.append(f"     - {role}: {preview}")

    if step.warnings:
        lines.append("   Warnings:")

        for warning in step.warnings:
            lines.append(f"     - {warning}")

    return "\n".join(lines)


def render_plan_for_user(plan: PlanProposal) -> str:
    lines = []

    lines.append("Here is a data-aware analysis plan. I have not run anything yet.")
    lines.append("")
    lines.append(f"Plan ID: {plan.plan_id}")
    lines.append(f"Data version: {plan.data_version_id}")
    lines.append(f"Plan status: {plan.status}")
    lines.append("")

    if plan.summary:
        lines.append(plan.summary)
        lines.append("")

    if plan.steps:
        lines.append("Recommended / candidate steps:")
        lines.append("")

        for i, step in enumerate(plan.steps, start=1):
            lines.append(_render_step(step, i))
            lines.append("")

    if plan.blocked_or_not_recommended:
        lines.append("Not recommended or currently blocked:")
        lines.append("")

        for i, step in enumerate(plan.blocked_or_not_recommended, start=1):
            lines.append(_render_step(step, i))
            lines.append("")

    lines.append("No tools have been executed.")
    lines.append("To execute ready steps later, say: `run the plan`.")

    return "\n".join(lines)