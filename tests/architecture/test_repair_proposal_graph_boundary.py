from pathlib import Path


def test_graph_attaches_repair_proposal_observe_only():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    assert "generate_repair_proposal" in graph_text
    assert "def _attach_repair_proposal" in graph_text
    assert "repair_proposal" in graph_text


def test_graph_repair_proposal_hook_does_not_execute_tools():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    start = graph_text.index("def _attach_repair_proposal")
    rest = graph_text[start + 1:]
    next_def_offset = rest.find("\ndef ")
    body = graph_text[start:] if next_def_offset == -1 else graph_text[start:start + 1 + next_def_offset]

    forbidden_fragments = [
        "execute_tool",
        "run_tool",
        "invoke(",
        "send",
    ]

    offenders = [
        fragment for fragment in forbidden_fragments
        if fragment in body
    ]

    assert offenders == []