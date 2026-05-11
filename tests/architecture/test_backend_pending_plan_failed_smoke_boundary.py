from pathlib import Path


def test_backend_pending_plan_failed_smoke_does_not_import_ui_app():
    text = Path(
        "tests/integration/test_backend_pending_plan_failed_step_smoke.py"
    ).read_text(encoding="utf-8")

    assert "import app" not in text
    assert "from app" not in text
    assert "streamlit" not in text.lower()


def test_backend_pending_plan_failed_smoke_uses_backend_nodes_only():
    text = Path(
        "tests/integration/test_backend_pending_plan_failed_step_smoke.py"
    ).read_text(encoding="utf-8")

    assert "execute_pending_plan_node" in text
    assert "verify_node" in text
    assert "summarize_node" in text

    # S14D deliberately avoids real tool execution.
    assert "execute_node(" not in text