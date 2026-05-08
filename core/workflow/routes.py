from __future__ import annotations


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