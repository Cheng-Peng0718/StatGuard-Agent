from pathlib import Path


def test_runtime_utils_live_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )
    runtime_text = Path("core/workflow/runtime_utils.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    for forbidden in [
        "def sanitize_results",
        "def get_action_hash",
    ]:
        assert forbidden not in graph_text

    for required in [
        "def sanitize_results",
        "def get_action_hash",
    ]:
        assert required in runtime_text


def test_core_graph_does_not_import_runtime_utils_helpers():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.runtime_utils import" not in graph_text
    assert "sanitize_results" not in graph_text
    assert "get_action_hash" not in graph_text