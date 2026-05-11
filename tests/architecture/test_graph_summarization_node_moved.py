from pathlib import Path


def test_summarize_node_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    summarization_text = Path("core/workflow/nodes/summarization.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "def summarize_node" not in graph_text
    assert "def summarize_node" in summarization_text


def test_core_graph_imports_summarize_node_from_workflow_nodes():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert (
        "from core.workflow.nodes.summarization import summarize_node"
        in graph_text
    )

    forbidden_imports = [
        "from core.analysis_runs import build_analysis_run_from_observation",
        "extract_data_version_update",
        "validate_data_version_update",
        "normalize_execution_view",
        "attach_execution_audit",
        "attach_repair_after_summarize",
    ]

    for forbidden in forbidden_imports:
        assert forbidden not in graph_text


def test_summarization_node_keeps_delta_analysis_runs_contract():
    summarization_text = Path("core/workflow/nodes/summarization.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert 'updates["analysis_runs"] = [analysis_run]' in summarization_text
    assert 'existing_runs = state.get("analysis_runs", []) or []' not in summarization_text
    assert 'updates["analysis_runs"] = existing_runs + [analysis_run]' not in summarization_text