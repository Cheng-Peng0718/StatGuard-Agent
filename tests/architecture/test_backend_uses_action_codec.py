from pathlib import Path


def test_backend_turn_uses_action_codec_for_action_rehydration():
    text = Path("core/controller/backend_turn.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.action_codec import action_from_state" in text
    assert "ActionProposal.model_validate(payload)" not in text
    assert "SimpleNamespace(" not in text