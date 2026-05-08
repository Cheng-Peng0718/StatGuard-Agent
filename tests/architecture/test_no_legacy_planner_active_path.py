from pathlib import Path


def test_legacy_planner_not_registered_in_active_graph():
    graph_text = Path("core/workflow/routes.py").read_text(encoding="utf-8")

    assert 'workflow.add_node("planner"' not in graph_text
    assert 'workflow.add_edge("planner", "supervisor")' not in graph_text


def test_intent_router_does_not_call_router_gate():
    graph_text = Path("core/workflow/routes.py").read_text(encoding="utf-8")

    start = graph_text.index("def route_after_intent")
    rest = graph_text[start + 1:]
    next_def_offset = rest.find("\ndef ")
    body = graph_text[start:] if next_def_offset == -1 else graph_text[start:start + 1 + next_def_offset]

    assert "router_gate(" not in body
    assert 'return "supervisor"' in body


def test_legacy_planner_functions_removed_from_graph():
    graph_text = Path("core/workflow/routes.py").read_text(encoding="utf-8")

    forbidden_defs = [
        "def router_gate",
        "def planner_node",
        "def call_llm_to_route",
        "def call_llm_to_plan",
        "def parse_plan",
        "def parse_llm_plan",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text