from pathlib import Path


def test_summarize_node_records_failed_tool_runs_as_analysis_runs():
    graph_text = Path("core/workflow/nodes/summarization.py").read_text(encoding="utf-8")

    start = graph_text.index("def summarize_node")
    rest = graph_text[start + 1:]
    next_def_offset = rest.find("\ndef ")
    body = graph_text[start:] if next_def_offset == -1 else graph_text[start:start + 1 + next_def_offset]

    assert "build_analysis_run_from_observation(" in body
    assert "observation=refined_observation" in body

    # S12C: summarize_node must not restrict AnalysisRun creation to successful statuses only.
    assert 'if status in {"ok", "warning"}' not in body
    assert "if status in {'ok', 'warning'}" not in body