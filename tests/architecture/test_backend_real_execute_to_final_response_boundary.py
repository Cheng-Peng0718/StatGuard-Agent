from pathlib import Path


def test_real_execute_to_final_response_smoke_does_not_import_ui_app():
    text = Path(
        "tests/integration/test_backend_real_execute_to_final_response_smoke.py"
    ).read_text(encoding="utf-8")

    assert "import app" not in text
    assert "from app" not in text
    assert "streamlit" not in text.lower()


def test_real_execute_to_final_response_smoke_uses_backend_nodes():
    text = Path(
        "tests/integration/test_backend_real_execute_to_final_response_smoke.py"
    ).read_text(encoding="utf-8")

    assert "execute_pending_plan_node" in text
    assert "verify_node" in text
    assert "execute_node" in text
    assert "summarize_node" in text
    assert "deliverable_gate_node" in text
    assert "final_response_node" in text