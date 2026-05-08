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

def route_after_summarize(state: dict):
    # A single "run the plan" turn executes at most one PlanStep.
    # If the action came from pending_plan, stop after summarize.
    if state.get("action_origin") == "pending_plan":
        return "end"

    if state.get("current_step", 0) >= state.get("max_steps", 12):
        return "end"

    observations = state.get("observations", []) or []
    last_obs = observations[-1] if observations else {}

    status = last_obs.get("status") if isinstance(last_obs, dict) else None
    error_code = last_obs.get("error_code") if isinstance(last_obs, dict) else None

    raw_data = last_obs.get("raw_data", {}) if isinstance(last_obs, dict) else {}
    recoverable = False

    if isinstance(raw_data, dict):
        recoverable = bool(raw_data.get("recoverable", False))

    # Successful or warning result: continue normal loop.
    if status in {"ok", "warning"}:
        return "build_context"

    # Human confirmation required is an interrupt/review state, not a tool failure.
    if error_code == "HUMAN_CONFIRMATION_REQUIRED":
        return "end"

    # Recoverable tool/schema failures can go back once or twice.
    # For now, use max_steps as the safety brake.
    if status in {"blocked", "failed", "rejected"} and recoverable:
        return "build_context"

    # Non-recoverable failures should stop and let final answer/report explain blocker.
    return "end"

def route_after_deliverable_gate(state: dict):
    """
    If deliverables are satisfied, allow final_answer to end.
    If deliverables are missing, go back to build_context so Supervisor can continue.
    """
    deliverable_check = state.get("deliverable_check") or {}

    if isinstance(deliverable_check, dict):
        status = deliverable_check.get("status")
    else:
        status = getattr(deliverable_check, "status", None)

    print(f"[ROUTE AFTER DELIVERABLE GATE] status = {status}")

    if status == "ok":
        return "final_response"

    if status in {"needs_more_work", "missing", "blocked"}:
        return "build_context"

    return "end"