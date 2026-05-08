from core.workflow.routes import route_after_intent


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