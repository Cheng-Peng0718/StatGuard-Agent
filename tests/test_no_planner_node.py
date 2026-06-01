"""
Architecture invariant: there is NO planner node / plan->execute path.

The project deliberately uses a Supervisor-driven, evidence-coverage feedback
loop -- the LLM decides WHAT to investigate one step at a time, and the
deterministic plugins decide HOW to compute it correctly. An early-stage
``planner_node`` (a plan->execute stub that built an ``analysis_plan`` list)
was kept registered but bypassed by a hard-coded router for a long time.

That bypassed node was an *armed* orphan: a single careless edit to the router
could have silently reintroduced the plan->execute paradigm the project
abandoned, and nothing would have caught it. The node and its helpers have been
removed; these tests make the removal permanent by failing loudly if anyone
reintroduces a planner node, a "planner" route, or the plan-building helpers.
"""

import core.graph as graph_module
from core.graph import app


def _compiled_nodes():
    return set(app.get_graph().nodes.keys())


def _compiled_edges():
    return [(e.source, e.target) for e in app.get_graph().edges]


def test_no_planner_node_in_compiled_graph():
    nodes = _compiled_nodes()
    assert "planner" not in nodes, (
        "A 'planner' node was reintroduced into the compiled graph. The project "
        "uses a Supervisor-driven loop, not plan->execute. Remove the planner node."
    )


def test_build_context_hands_off_only_to_coverage_brief():
    # The single live entry route must stay build_context -> coverage_brief.
    targets = sorted(t for s, t in _compiled_edges() if s == "build_context")
    assert targets == ["coverage_brief"], (
        "build_context must route only to coverage_brief (then supervisor). "
        f"Found targets: {targets}. Do not add a 'planner' or other branch here."
    )


def test_no_node_routes_to_a_planner():
    targets = {t for _, t in _compiled_edges()}
    assert "planner" not in targets, (
        "Some node routes to 'planner'. The plan->execute path was removed; "
        "do not reintroduce it."
    )


def test_plan_building_helpers_are_gone():
    # These symbols were the plan->execute seed. They must not come back.
    for symbol in ("planner_node", "router_gate", "call_llm_to_plan",
                   "call_llm_to_route", "parse_plan"):
        assert not hasattr(graph_module, symbol), (
            f"core.graph.{symbol} was reintroduced. This is the plan->execute "
            "stub the project deliberately removed."
        )