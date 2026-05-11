from __future__ import annotations

import uuid

from core.action_access import (
    get_action_arguments,
    get_action_id,
    get_action_tool_name,
)
from core.action_codec import action_to_state_dict
from core.responses import make_assistant_response
from core.schema import Observation
from core.verification_access import (
    get_verification_details,
    get_verification_feedback,
    get_verification_status,
    set_verification_fields,
)
from core.verification_codec import verification_to_state_dict


def human_review_node(state: dict):
    """
    Human review node.

    This node does not execute tools.
    It converts verification review state into one of:
    - pending review state
    - approved verification state
    - explicit user rejection record
    - stale/mismatched review error
    """
    vr = state.get("current_verification")
    action = state.get("current_action")

    if vr is None or action is None:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id="unknown",
            tool_name=None,
            arguments={},
            status="rejected",
            success=False,
            error_code="MISSING_REVIEW_STATE",
            message="Human review node was reached without verification or action.",
            artifacts=[],
            summary="Human review could not proceed because verification/action state was missing.",
            structured_data={
                "status": "rejected",
                "success": False,
                "error_code": "MISSING_REVIEW_STATE",
            },
            raw_data={},
        )
        return {"observations": [obs.model_dump()]}

    tool_name = get_action_tool_name(action)
    arguments = get_action_arguments(action)

    vr_details = get_verification_details(vr)
    vr_status = get_verification_status(vr)
    feedback = get_verification_feedback(vr)

    action_id = get_action_id(action)
    action_payload = action_to_state_dict(action)
    verification_payload = verification_to_state_dict(vr)

    canonical_arguments = vr_details.get("canonical_arguments") or arguments

    human_review_decision = state.get("human_review_decision")

    submitted_action_hash = state.get("human_review_action_hash")
    expected_action_hash = vr_details.get("action_hash")

    if (
        human_review_decision in {"approved", "rejected"}
        and submitted_action_hash
        and expected_action_hash
        and submitted_action_hash != expected_action_hash
    ):
        content = (
            "The human-review decision did not match the current pending action. "
            "This can happen if the UI was stale or the action changed before approval. "
            "No tool was executed. Please review the current action again."
        )

        return {
            "human_review_required": True,
            "pending_action": action_payload,
            "current_action": action,
            "current_verification": vr,
            "human_review_decision": None,
            "human_review_rejection_reason": None,
            "assistant_response": make_assistant_response(
                response_type="error",
                content=content,
                source_node="human_review",
                data_version_id=state.get("active_data_version_id"),
                metadata={
                    "error_code": "HUMAN_REVIEW_ACTION_HASH_MISMATCH",
                    "submitted_action_hash": submitted_action_hash,
                    "expected_action_hash": expected_action_hash,
                    "tool_name": tool_name,
                    "action_id": action_id,
                },
            ),
        }

    if vr_status == "needs_review" and human_review_decision == "rejected":
        reason = (
            state.get("human_review_rejection_reason")
            or "User rejected the human-review action."
        )

        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action_id,
            tool_name=tool_name,
            arguments=canonical_arguments,
            status="rejected",
            success=False,
            error_code="HUMAN_REVIEW_REJECTED",
            message=reason,
            artifacts=[],
            summary=(
                f"Human review rejected action {tool_name}. "
                f"Reason: {reason}"
            ),
            structured_data={
                "status": "rejected",
                "success": False,
                "error_code": "HUMAN_REVIEW_REJECTED",
                "message": reason,
                "pending_action": action_payload or {},
            },
            raw_data={
                "verification": verification_payload or {},
                "pending_action": action_payload or {},
                "human_review_decision": "rejected",
            },
        )

        return {
            "human_review_required": False,
            "pending_action": None,
            "human_review_decision": None,
            "human_review_rejection_reason": None,
            "current_action": None,
            "current_verification": None,
            "observations": [obs.model_dump()],
        }

    if vr_status == "needs_review" and human_review_decision == "approved":
        print("[HUMAN REVIEW] User approved needs_review action; routing to execute.")

        approved_feedback = (
            "Human review approved this action for execution."
        )

        if isinstance(vr, dict):
            approved_vr = {
                **vr,
                "status": "allowed",
                "feedback": approved_feedback,
            }

        elif hasattr(vr, "model_copy"):
            approved_vr = vr.model_copy(
                update={
                    "status": "allowed",
                    "feedback": approved_feedback,
                }
            )

        else:
            approved_vr = set_verification_fields(
                vr,
                status="allowed",
                feedback=approved_feedback,
            )

        return {
            "current_verification": approved_vr,
            "human_review_required": False,
            "pending_action": None,
            "human_review_decision": None,
            "current_action": action,
        }

    if vr_status == "allowed":
        print("[HUMAN REVIEW] User approved action; routing to execute.")
        return {}

    if vr_status == "needs_review":
        return {
            "human_review_required": True,
            "pending_action": action_payload,
            "current_action": action,
            "current_verification": vr,
        }

    if vr_status in {"rejected_recoverable", "rejected_terminal"}:
        obs = Observation(
            observation_id=f"obs_{uuid.uuid4().hex[:8]}",
            source_action_id=action_id,
            tool_name=tool_name,
            arguments=arguments,
            status="rejected",
            success=False,
            error_code="VERIFICATION_FAILED",
            message=feedback,
            artifacts=[],
            summary=f"Action {tool_name} was rejected by verifier: {feedback}",
            structured_data={
                "status": vr_status,
                "success": False,
                "error_code": "VERIFICATION_FAILED",
                "message": feedback,
            },
            raw_data={
                "verification": verification_payload or {},
            },
        )

        return {"observations": [obs.model_dump()]}

    obs = Observation(
        observation_id=f"obs_{uuid.uuid4().hex[:8]}",
        source_action_id=action_id,
        tool_name=tool_name,
        arguments=arguments,
        status="rejected",
        success=False,
        error_code="UNHANDLED_HUMAN_REVIEW_STATUS",
        message=f"Unhandled verification status in human_review_node: {vr_status}",
        artifacts=[],
        summary=f"Unhandled human review status: {vr_status}. Tool was not executed.",
        structured_data={
            "status": vr_status,
            "success": False,
            "error_code": "UNHANDLED_HUMAN_REVIEW_STATUS",
        },
        raw_data={
            "verification": verification_payload or {},
        },
    )

    return {"observations": [obs.model_dump()]}