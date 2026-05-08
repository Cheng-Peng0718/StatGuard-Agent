from core.workflow.routes import (
    route_after_execute_pending_plan,
    route_after_intent,
    route_after_supervisor,
    route_after_verify,
    route_after_review,
)


def test_route_after_intent_routes_advisory():
    assert route_after_intent({"interaction_intent": "advisory"}) == "advisory_answer"


def test_route_after_intent_routes_plan_only():
    assert route_after_intent({"interaction_intent": "plan_only"}) == "plan_only"


def test_route_after_intent_routes_execute_plan():
    assert route_after_intent({"interaction_intent": "execute_plan"}) == "execute_pending_plan"


def test_route_after_intent_routes_unknown_to_supervisor():
    assert route_after_intent({"interaction_intent": "unknown"}) == "supervisor"


def test_route_after_intent_routes_missing_to_supervisor():
    assert route_after_intent({}) == "supervisor"

def test_route_after_execute_pending_plan_routes_to_verify_when_action_exists():
    assert route_after_execute_pending_plan({
        "current_action": {
            "action_id": "act_1",
            "tool_name": "get_summary_stats",
        }
    }) == "verify"


def test_route_after_execute_pending_plan_routes_to_end_without_action():
    assert route_after_execute_pending_plan({
        "current_action": None,
    }) == "end"


def test_route_after_execute_pending_plan_routes_to_end_when_missing_action():
    assert route_after_execute_pending_plan({}) == "end"

def test_route_after_supervisor_routes_final_answer_to_deliverable_gate():
    assert route_after_supervisor({
        "current_action": {
            "action_type": "final_answer",
            "arguments": {},
        },
        "current_step": 1,
        "max_steps": 12,
    }) == "deliverable_gate"


def test_route_after_supervisor_routes_ask_user_to_deliverable_gate():
    assert route_after_supervisor({
        "current_action": {
            "action_type": "ask_user",
            "arguments": {},
        },
        "current_step": 1,
        "max_steps": 12,
    }) == "deliverable_gate"


def test_route_after_supervisor_routes_max_steps_to_end():
    assert route_after_supervisor({
        "current_action": {
            "action_type": "tool_call",
            "tool_name": "get_summary_stats",
            "arguments": {},
        },
        "current_step": 12,
        "max_steps": 12,
    }) == "end"


def test_route_after_supervisor_routes_tool_call_to_verify():
    assert route_after_supervisor({
        "current_action": {
            "action_type": "tool_call",
            "tool_name": "get_summary_stats",
            "arguments": {},
        },
        "current_step": 1,
        "max_steps": 12,
    }) == "verify"


def test_route_after_supervisor_routes_missing_action_to_verify_before_max_steps():
    assert route_after_supervisor({
        "current_action": None,
        "current_step": 1,
        "max_steps": 12,
    }) == "verify"

def test_route_after_verify_routes_allowed_to_execute():
    assert route_after_verify({
        "current_verification": {
            "status": "allowed",
            "feedback": "ok",
            "details": {},
        }
    }) == "execute"


def test_route_after_verify_routes_needs_review_to_human_review():
    assert route_after_verify({
        "current_verification": {
            "status": "needs_review",
            "feedback": "requires approval",
            "details": {},
        }
    }) == "human_review"


def test_route_after_verify_routes_rejected_to_build_context():
    assert route_after_verify({
        "current_verification": {
            "status": "rejected_recoverable",
            "feedback": "missing arguments",
            "details": {},
        }
    }) == "build_context"


def test_route_after_verify_routes_missing_verification_to_build_context():
    assert route_after_verify({
        "current_verification": None,
    }) == "build_context"


def test_route_after_verify_ends_for_rejected_pending_plan_action():
    assert route_after_verify({
        "action_origin": "pending_plan",
        "current_verification": {
            "status": "rejected_recoverable",
            "feedback": "invalid planned action",
            "details": {},
        },
    }) == "end"


def test_route_after_verify_ends_for_step_verification_failed_status():
    assert route_after_verify({
        "plan_execution_status": "step_verification_failed",
        "current_verification": {
            "status": "rejected_recoverable",
            "feedback": "failed",
            "details": {},
        },
    }) == "end"

def test_route_after_review_routes_allowed_to_execute():
    assert route_after_review({
        "current_verification": {
            "status": "allowed",
            "feedback": "approved",
            "details": {},
        }
    }) == "execute"


def test_route_after_review_routes_missing_verification_to_build_context():
    assert route_after_review({
        "current_verification": None,
    }) == "build_context"


def test_route_after_review_routes_rejected_to_build_context():
    assert route_after_review({
        "current_verification": {
            "status": "rejected_recoverable",
            "feedback": "not approved",
            "details": {},
        }
    }) == "build_context"


def test_route_after_review_routes_needs_review_to_build_context():
    assert route_after_review({
        "current_verification": {
            "status": "needs_review",
            "feedback": "still waiting",
            "details": {},
        }
    }) == "build_context"