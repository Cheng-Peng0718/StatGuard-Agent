from pathlib import Path


def test_ui_snapshot_uses_runtime_codecs_for_action_verification_execution():
    text = Path("core/ui_adapter/snapshot.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.action_codec import normalize_action_payload" in text
    assert "from core.verification_codec import verification_to_state_dict" in text
    assert "from core.execution_codec import normalize_execution_view" in text

    assert "action_dict = normalize_action_payload(action)" in text
    assert "verification_dict = verification_to_state_dict(verification)" in text
    assert "execution_view = normalize_execution_view(execution)" in text

    forbidden_patterns = [
        '_get_field(verification, "status")',
        '_get_field(verification, "feedback")',
        '_get_field(verification, "error_code")',
        '_get_field(verification, "details"',
    ]

    for pattern in forbidden_patterns:
        assert pattern not in text