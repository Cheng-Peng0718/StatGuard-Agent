from core.graph import route_after_summarize


def test_route_after_summarize_ends_for_pending_plan_action():
    state = {
        "action_origin": "pending_plan",
    }

    assert route_after_summarize(state) == "end"