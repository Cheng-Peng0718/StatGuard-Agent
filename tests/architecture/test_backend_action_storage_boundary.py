from pathlib import Path


def test_backend_turn_normalizes_actions_at_finish_boundary():
    text = Path("core/controller/backend_turn.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.action_codec import action_from_state, action_to_state_dict" in text
    assert ' _ACTION_STATE_FIELDS = ("current_action", "pending_action")' not in text

    assert "_ACTION_STATE_FIELDS = (\"current_action\", \"pending_action\")" in text
    assert "def _normalize_state_actions_for_storage" in text
    assert "state = _normalize_state_actions_for_storage(state)" in text
    assert "state = _ensure_graph_action_object(state)" in text