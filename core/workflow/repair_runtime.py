from __future__ import annotations

from core.repair.decision import evaluate_repair_decision
from core.repair.attempts import (
    append_repair_attempt,
    can_attempt_repair,
    make_repair_attempt,
)
from core.repair.proposal_generator import generate_repair_proposal

from typing import Any

from core.action_access import get_action_id
from core.execution_codec import execution_to_state_dict


def attach_repair_decision(state: dict, updates: dict) -> dict:
    """
    Attach observe-only repair decision to graph updates.

    This does not retry, reroute, or mutate actions. It only records whether
    the current verification/execution failure appears repairable.
    """
    repair_state = dict(state)
    repair_state.update(updates)

    decision = evaluate_repair_decision(repair_state)
    updates["repair_decision"] = decision.model_dump()

    if decision.status != "no_repair_needed":
        print("\n" + "=" * 40)
        print("[REPAIR DECISION]")
        print(decision.model_dump())
        print("=" * 40 + "\n")

    updates = attach_repair_proposal(state, updates)
    return attach_repair_attempt_if_allowed(state, updates)


def attach_repair_proposal(state: dict, updates: dict) -> dict:
    """
    Observe-only repair proposal generation.

    This does not apply repairs, retry tools, call LLMs, or change routing.
    It only records what deterministic repair proposal would be available.
    """
    repair_state = dict(state)
    pre_update_action = repair_state.get("current_action")

    repair_state.update(updates)

    repair_decision = updates.get("repair_decision") or repair_state.get("repair_decision")

    if not isinstance(repair_decision, dict):
        return updates

    if repair_decision.get("status") == "no_repair_needed":
        return updates

    # summarize_node may clear current_action in updates.
    # The proposal still needs the original source action.
    current_action = repair_state.get("current_action") or pre_update_action

    if current_action is None:
        return updates

    proposal = generate_repair_proposal(
        repair_decision=repair_decision,
        current_action=current_action,
    )

    updates["repair_proposal"] = proposal

    print("\n" + "=" * 40)
    print("[REPAIR PROPOSAL]")
    print(proposal)
    print("=" * 40 + "\n")

    return updates


def attach_repair_attempt_if_allowed(state: dict, updates: dict) -> dict:
    """
    Observe-only repair attempt logging.

    This does not apply repairs, retry tools, call LLMs, or change routing.
    It only records a proposed repair attempt when the current repair_decision
    says the failure is repairable / needs_user and the tool policy allows
    another attempt.
    """
    repair_state = dict(state)
    pre_update_action = repair_state.get("current_action")

    repair_state.update(updates)

    repair_decision = updates.get("repair_decision") or repair_state.get("repair_decision")

    # summarize_node clears current_action in updates after archiving the execution.
    # Repair attempts still need the original source action for traceability.
    current_action = repair_state.get("current_action") or pre_update_action

    if current_action is not None:
        repair_state["current_action"] = current_action

    repair_attempts = repair_state.get("repair_attempts", []) or []

    if not can_attempt_repair(
        repair_decision=repair_decision,
        repair_attempts=repair_attempts,
        current_action=current_action,
    ):
        return updates

    decision_status = repair_decision.get("status") if isinstance(repair_decision, dict) else None

    if decision_status == "needs_user":
        repair_type = "ask_user"
        message = "Repair requires user-provided choices or missing roles."
    else:
        repair_type = "argument_repair"
        message = "A backend repair attempt is possible, but S13D only records the proposal."

    attempt = make_repair_attempt(
        repair_decision=repair_decision,
        current_action=current_action,
        repair_type=repair_type,
        proposed_arguments={},
        proposed_tool_name=None,
        message=message,
        metadata={
            "observe_only": True,
            "stage": "S13D",
        },
    )

    updates["repair_attempts"] = append_repair_attempt(
        repair_attempts,
        attempt,
    )

    print("\n" + "=" * 40)
    print("[REPAIR ATTEMPT PROPOSED]")
    print(attempt)
    print("=" * 40 + "\n")

    return updates

def attach_repair_after_summarize(
    *,
    state: dict,
    updates: dict,
    current_action: Any,
    raw_result: Any,
    tool_name: str | None,
) -> dict:
    """
    Observe-only repair decision after summarize_node has archived execution.

    summarize_node clears current_action/current_execution in updates, but repair
    evaluation still needs the pre-clear action and execution result. This helper
    builds the repair view explicitly and keeps graph.py from owning repair logic.
    """
    repair_state = dict(state)

    repair_state["current_execution"] = execution_to_state_dict(
        raw_result,
        fallback_action_id=get_action_id(current_action),
        fallback_tool_name=tool_name,
    )

    repair_decision = evaluate_repair_decision(repair_state)
    updates["repair_decision"] = repair_decision.model_dump()

    if repair_decision.status != "no_repair_needed":
        print("\n" + "=" * 40)
        print("[REPAIR DECISION]")
        print(repair_decision.model_dump())
        print("=" * 40 + "\n")

    updates = attach_repair_proposal(repair_state, updates)
    updates = attach_repair_attempt_if_allowed(repair_state, updates)

    return updates