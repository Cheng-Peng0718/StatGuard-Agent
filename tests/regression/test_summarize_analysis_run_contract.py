from pathlib import Path


def test_summarize_node_builds_analysis_run_from_observation_contract():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    start = graph_text.index("def summarize_node")
    rest = graph_text[start + 1:]
    next_def_offset = rest.find("\ndef ")
    body = graph_text[start:] if next_def_offset == -1 else graph_text[start:start + 1 + next_def_offset]

    assert "build_analysis_run_from_observation(" in body
    assert "observation=" in body