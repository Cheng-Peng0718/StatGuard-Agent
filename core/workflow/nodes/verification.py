from __future__ import annotations

import uuid

from verifiers.validators import verify

from core.action_access import (
    get_action_arguments,
    get_action_id,
    get_action_tool_name,
)
from core.planning.execution_queue import mark_plan_step_after_execution
from core.responses import make_assistant_response
from core.schema import Observation
from core.verification_access import (
    get_verification_details,
    get_verification_error_code,
    get_verification_feedback,
    get_verification_status,
    set_verification_fields,
)
from core.workflow.repair_runtime import attach_repair_decision
from core.workflow.runtime_utils import get_action_hash
from core.workflow.verification_feedback import attach_verification_blocked_response
from core.workflow.profile_access import require_dataset_profile_v2

def verify_node(state: dict):
    """
    Verification node.

    Contract:
    verify()
      -> normalize/access verification fields
      -> attach action_hash
      -> rejected: observation + optional pending-plan failure
      -> attach repair_decision
      -> attach verification-blocked assistant_response
      -> return stable updates
    """
    action = state["current_action"]

    status, feedback, verify_result = verify(action, require_dataset_profile_v2(state))

    tool_name = get_action_tool_name(action)
    arguments = get_action_arguments(action)
    action_id = get_action_id(action)

    action_hash = get_action_hash(tool_name, arguments)

    verification_details = get_verification_details(verify_result)
    verification_details["action_hash"] = action_hash
    verify_result = set_verification_fields(
        verify_result,
        details=verification_details,
    )

    verify_status = get_verification_status(verify_result)
    verify_feedback = get_verification_feedback(verify_result)
    verify_error_code = get_verification_error_code(verify_result)
    verify_details = get_verification_details(verify_result)

    print("\n" + "=" * 40)
    print("[VERIFY NODE DEBUG]")
    print(f"tool_name = {tool_name}")
    print(f"verify_result.status = {verify_status}")
    print(f"verify_result.error_code = {verify_error_code}")
    print(f"verify_result.feedback = {verify_feedback}")
    print(f"verify_result.details = {verify_details}")
    print("=" * 40 + "\n")

    if verify_status in ["rejected_recoverable", "rejected_terminal"]:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action_id,
            tool_name=tool_name,
            arguments=getattr(action, "arguments", {}) or {},
            status="rejected",
            success=False,
            error_code=verify_error_code or "VERIFICATION_FAILED",
            message=verify_feedback,
            artifacts=[],
            summary=(
                f"Validation failed for {tool_name}: "
                f"{verify_feedback}"
            ),
            structured_data={
                "status": "rejected",
                "success": False,
                "error_code": verify_error_code or "VERIFICATION_FAILED",
                "message": verify_feedback,
                "details": verify_details,
            },
            raw_data={
                "verification": (
                    verify_result.model_dump()
                    if hasattr(verify_result, "model_dump")
                    else verify_result
                ),
                "recoverable": verify_status == "rejected_recoverable",
            },
        )

        updates = {
            "current_verification": verify_result,
            "observations": [obs.model_dump()],
        }

        # Verification rejection is also a repair-decision input.
        # Attach repair metadata before clearing current_action, because
        # repair classification needs the source action/tool.
        clear_after_repair = {}

        current_plan_step_id = state.get("current_plan_step_id")
        pending_plan = state.get("pending_plan")

        if current_plan_step_id and pending_plan:
            updated_plan = mark_plan_step_after_execution(
                pending_plan,
                step_id=current_plan_step_id,
                success=False,
                execution_id=None,
                message=verify_feedback,
            )

            content = (
                "I tried to execute the next ready step in the pending plan, "
                "but it failed validation before execution.\n\n"
                f"Tool: {tool_name}\n"
                f"Reason: {verify_feedback}\n\n"
                "I marked this plan step as failed and stopped execution to avoid a retry loop."
            )

            updates.update({
                "pending_plan": updated_plan,
                "plan_status": updated_plan.get("status"),
                "plan_execution_status": "step_verification_failed",
                "assistant_response": make_assistant_response(
                    response_type="error",
                    content=content,
                    source_node="verify",
                    data_version_id=state.get("active_data_version_id"),
                    plan_id=updated_plan.get("plan_id"),
                    plan_status=updated_plan.get("status"),
                    metadata={
                        "error_code": verify_error_code,
                        "tool_name": tool_name,
                        "step_id": current_plan_step_id,
                    },
                ),
            })

            clear_after_repair = {
                "current_plan_step_id": None,
                "current_action": None,
                "current_execution": None,
            }

        updates = attach_repair_decision(state, updates)
        updates = attach_verification_blocked_response(state, updates)
        updates.update(clear_after_repair)

        return updates

    updates = {
        "current_verification": verify_result,
        "human_review_required": False,
    }

    return attach_repair_decision(state, updates)