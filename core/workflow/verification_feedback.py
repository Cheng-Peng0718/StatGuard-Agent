from __future__ import annotations

from typing import Any, Dict

from core.action_access import get_action_tool_name
from core.responses import make_response_update
from core.verification_access import (
    get_verification_error_code,
    get_verification_feedback,
    get_verification_status,
)


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def _proposal_summary(proposal: Dict[str, Any]) -> str | None:
    if not proposal:
        return None

    proposal_type = proposal.get("proposal_type")
    reason = proposal.get("reason")

    if proposal_type == "argument_repair":
        return (
            "A deterministic argument repair is available. "
            "Review the proposed arguments before retrying."
        )

    if proposal_type == "ask_user":
        missing = proposal.get("metadata", {}).get("missing_fields") or []
        if missing:
            return (
                "More user input is needed before this action can run. "
                f"Missing fields: {', '.join(missing)}."
            )

        return "More user input is needed before this action can run."

    if proposal_type == "no_op":
        return reason or "No automatic repair proposal is available."

    return reason or "A repair proposal is available."


def attach_verification_blocked_response(state: dict, updates: dict) -> dict:
    """
    Attach a user-visible assistant_response for verification rejections.

    This does not repair, retry, execute tools, or change routing.
    It only explains why verification stopped the turn and surfaces repair metadata.
    """
    if updates.get("assistant_response") is not None:
        return updates

    merged = dict(state)
    merged.update(updates)

    verification = merged.get("current_verification")
    status = get_verification_status(verification)

    if status not in {"rejected_recoverable", "rejected_terminal"}:
        return updates

    action = merged.get("current_action")
    tool_name = get_action_tool_name(action) or "unknown_tool"

    feedback = get_verification_feedback(verification)
    error_code = get_verification_error_code(verification)

    repair_decision = _as_dict(merged.get("repair_decision"))
    repair_proposal = _as_dict(merged.get("repair_proposal"))

    decision_status = repair_decision.get("status")
    proposal_text = _proposal_summary(repair_proposal)

    lines = [
        "I could not run the requested action because it failed verification.",
        "",
        f"Tool: {tool_name}",
        f"Verification status: {status}",
    ]

    if error_code:
        lines.append(f"Error code: {error_code}")

    if feedback:
        lines.extend(["", f"Reason: {feedback}"])

    if decision_status:
        lines.extend(["", f"Repair decision: {decision_status}"])

    if proposal_text:
        lines.extend(["", f"Suggested next step: {proposal_text}"])

    if status == "rejected_recoverable":
        lines.extend([
            "",
            "No tool was executed. The action can be corrected and retried in a later turn.",
        ])
    else:
        lines.extend([
            "",
            "No tool was executed. This verification failure is terminal for the current action.",
        ])

    response_update = make_response_update(
        response_type="error",
        content="\n".join(lines),
        source_node="verify",
        data_version_id=merged.get("active_data_version_id"),
        metadata={
            "semantic_type": "verification_blocked",
            "verification_status": status,
            "tool_name": tool_name,
            "error_code": error_code,
            "repair_decision": repair_decision or None,
            "repair_proposal": repair_proposal or None,
        },
    )

    updates.update(response_update)
    return updates