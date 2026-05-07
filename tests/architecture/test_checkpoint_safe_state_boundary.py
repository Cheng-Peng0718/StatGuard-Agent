from pathlib import Path


def test_checkpoint_safe_state_smoke_does_not_import_ui_app():
    text = Path(
        "tests/integration/test_checkpoint_safe_state_smoke.py"
    ).read_text(encoding="utf-8")

    assert "import app" not in text
    assert "from app" not in text
    assert "streamlit" not in text.lower()


def test_checkpoint_safe_state_smoke_uses_serialization_boundary():
    text = Path(
        "tests/integration/test_checkpoint_safe_state_smoke.py"
    ).read_text(encoding="utf-8")

    assert "make_checkpoint_safe_state" in text
    assert "audit_state_serialization" in text
    assert "json.dumps" in text