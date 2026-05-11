from pathlib import Path


def test_graph_state_declares_analysis_runs_as_append_reducer():
    text = Path("core/state.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "analysis_runs: Annotated[list, operator.add]" in text


def test_summarize_node_returns_analysis_runs_delta_not_full_registry():
    text = Path("core/workflow/nodes/summarization.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert 'existing_runs = state.get("analysis_runs", []) or []' not in text
    assert 'updates["analysis_runs"] = existing_runs + [analysis_run]' not in text
    assert 'updates["analysis_runs"] = [analysis_run]' in text


def test_backend_controller_appends_observations_and_analysis_runs():
    text = Path("core/controller/backend_turn.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert 'if key in {"observations", "analysis_runs"} and isinstance(value, list):' in text