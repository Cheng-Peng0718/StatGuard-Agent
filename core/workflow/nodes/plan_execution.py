from __future__ import annotations

from core.action_access import (
    get_action_arguments,
    get_action_id,
    get_action_tool_name,
)
from core.planning.dependencies import modeling_blocked_by_pending_cleaning
from core.planning.execution_queue import (
    find_next_executable_step,
    mark_plan_step_started,
)
from core.responses import make_response_update
from core.workflow.profile_access import (
    get_context_profile,
    get_dataset_profile_v2,
)

def execute_pending_plan_node(state: dict):
    print("\n" + "=" * 40)
    print("[EXECUTE PENDING PLAN NODE ENTERED]")
    print(f"plan_status = {state.get('plan_status')}")
    print(f"has_pending_plan = {state.get('pending_plan') is not None}")
    print("=" * 40 + "\n")

    pending_plan = state.get("pending_plan")

    if not pending_plan:
        content = (
            "There is no pending plan to execute. "
            "Please ask me to make a plan first."
        )

        updates = make_response_update(
            response_type="plan_execution_status",
            content=content,
            source_node="execute_pending_plan",
            data_version_id=state.get("active_data_version_id"),
            metadata={
                "reason": "no_pending_plan",
            },
        )

        updates.update({
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
            "plan_execution_status": "no_pending_plan",
        })

        return updates

    next_step, readiness = find_next_executable_step(
        pending_plan,
        profile=get_context_profile(state),
    )

    if next_step is None or readiness is None or not readiness.executable:
        reason = readiness.reason if readiness is not None else "No candidate step found."
        missing_choices = readiness.missing_user_choices if readiness is not None else []

        lines = []
        lines.append("The pending plan has no execution-ready steps.")
        lines.append("")
        lines.append(f"Reason: {reason}")

        if missing_choices:
            lines.append("")
            lines.append("Missing choices:")
            for choice in missing_choices:
                lines.append(f"- {choice}")

        lines.append("")
        lines.append("No tools were executed.")

        content = "\n".join(lines)

        updates = make_response_update(
            response_type="plan_execution_status",
            content=content,
            source_node="execute_pending_plan",
            data_version_id=state.get("active_data_version_id"),
            plan_id=pending_plan.get("plan_id"),
            plan_status=pending_plan.get("status"),
            metadata={
                "reason": "no_executable_step",
                "readiness": readiness.model_dump() if readiness is not None else None,
            },
        )

        updates.update({
            "plan_execution_status": "blocked_no_ready_steps",
            "current_action": None,
            "current_execution": None,
            "current_verification": None,
        })

        return updates

    dataset_profile = get_dataset_profile_v2(state)

    if dataset_profile is not None:

        if modeling_blocked_by_pending_cleaning(
            step=next_step,
            pending_plan=pending_plan,
            profile=dataset_profile,
        ):
            step_id = next_step.get("step_id")
            tool_name = next_step.get("tool_name")

            content = (
                "This modeling step is waiting for data cleaning to complete "
                "because the current dataset has missing values."
            )

            updates = make_response_update(
                response_type="plan_execution_status",
                content=content,
                source_node="execute_pending_plan",
                data_version_id=state.get("active_data_version_id"),
                plan_id=pending_plan.get("plan_id"),
                plan_status=pending_plan.get("status"),
                metadata={
                    "reason": "modeling_blocked_by_pending_cleaning",
                    "step_id": step_id,
                    "tool_name": tool_name,
                },
            )

            updates.update({
                "plan_execution_status": "blocked_pending_data_cleaning",
                "current_action": None,
                "current_execution": None,
                "current_verification": None,
            })

            return updates

    action = readiness.action

    updated_plan = mark_plan_step_started(
        pending_plan,
        step_id=next_step["step_id"],
        action_id=get_action_id(action),
    )

    print("\n" + "=" * 40)
    print("[EXECUTE PENDING PLAN]")
    print(f"plan_id = {pending_plan.get('plan_id')}")
    print(f"step_id = {next_step.get('step_id')}")
    print(f"tool_name = {get_action_tool_name(action)}")
    print(f"arguments = {get_action_arguments(action)}")
    print(f"readiness_status = {readiness.status}")
    print("=" * 40 + "\n")

    return {
        "pending_plan": updated_plan,
        "plan_status": updated_plan.get("status"),
        "current_plan_step_id": next_step["step_id"],
        "plan_execution_status": "started_step",

        # Important for S4:
        # This action came from a pending plan, not a direct user tool request.
        "action_origin": "pending_plan",

        # Existing verify -> human_review / execute path.
        "current_action": action,
        "current_execution": None,
        "current_verification": None,
    }