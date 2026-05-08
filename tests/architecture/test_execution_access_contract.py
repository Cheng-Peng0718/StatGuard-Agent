from pathlib import Path


def test_summarize_node_uses_execution_codec_instead_of_manual_raw_result_branch():
    text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.execution_codec import normalize_execution_view, execution_to_state_dict" in text
    assert "execution_view = normalize_execution_view(" in text
    assert "if isinstance(raw_result, dict):" not in text
    assert 'error_code = "NON_STRUCTURED_EXECUTION_RESULT"' not in text


def test_backend_turn_normalizes_execution_at_finish_boundary():
    text = Path("core/controller/backend_turn.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.execution_codec import execution_to_state_dict" in text
    assert "_EXECUTION_STATE_FIELDS = (\"current_execution\",)" in text
    assert "def _normalize_state_executions_for_storage" in text
    assert "state = _normalize_state_executions_for_storage(state)" in text


def test_ui_snapshot_uses_execution_codec():
    text = Path("core/ui_adapter/snapshot.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.execution_codec import normalize_execution_view" in text
    assert "execution_view = normalize_execution_view(execution)" in text