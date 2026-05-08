from __future__ import annotations
from core.action_access import get_action_type
from core.verification_access import get_verification_status


def route_after_intent(state: dict):
    intent = state.get("interaction_intent")

    print(f"[ROUTE AFTER INTENT] intent = {intent}")

    if intent == "advisory":
        return "advisory_answer"

    if intent == "plan_only":
        return "plan_only"

    if intent == "execute_plan":
        return "execute_pending_plan"

    # Direct tool requests and unknown requests go to the unified supervisor path.
    return "supervisor"

def route_after_execute_pending_plan(state: dict):
    action = state.get("current_action")

    if action is not None:
        return "verify"

    return "end"

def route_after_supervisor(state: dict):
    """
    After supervisor:
    - tool_call -> verify
    - final_answer / ask_user -> deliverable_gate
    - max_steps reached -> end
    """
    action = state.get("current_action")

    if action and get_action_type(action) in ["final_answer", "ask_user"]:
        print("[ROUTE AFTER SUPERVISOR] final_answer -> deliverable_gate")
        return "deliverable_gate"

    if state.get("current_step", 0) >= state.get("max_steps", 12):
        print("[ROUTE AFTER SUPERVISOR] max_steps -> end")
        return "end"

    return "verify"

def route_after_verify(state: dict):
    """
    After verification:
    - allowed: execute the tool
    - needs_review: interrupt before human_review and wait for user approval
    - rejected_*: do not execute; go back to build_context so Supervisor can rethink/respond
    """

    # If a pending-plan action fails verification, do not loop back and continue
    # the same "run the plan" turn.
    if (
        state.get("action_origin") == "pending_plan"
        and state.get("current_verification") is not None
    ):
        verification = state.get("current_verification")
        status = get_verification_status(verification)

        if status in {"rejected_recoverable", "rejected_terminal"}:
            return "end"

    if state.get("plan_execution_status") == "step_verification_failed":
        return "end"

    vr = state.get("current_verification")

    if vr is None:
        print("[ROUTE AFTER VERIFY] no verification result -> build_context")
        return "build_context"

    status = get_verification_status(vr)

    print(f"[ROUTE AFTER VERIFY] status = {status}")

    if status == "allowed":
        return "execute"

    if status == "needs_review":
        return "human_review"

    if status in {"rejected_recoverable", "rejected_terminal"}:
        return "build_context"

    return "build_context"

def route_after_review(state: dict):
    """
    After human_review:
    - if user approved, execute the original pending action
    - otherwise go back to build_context and let Supervisor rethink/respond
    """
    vr = state.get("current_verification")

    if vr is None:
        print("[ROUTE AFTER REVIEW] no current_verification -> build_context")
        return "build_context"

    status = get_verification_status(vr)

    print(f"[ROUTE AFTER REVIEW] status = {status}")

    if status == "allowed":
        return "execute"

    return "build_context"