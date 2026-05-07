from pathlib import Path


def test_backend_intent_smoke_does_not_import_ui_app():
    text = Path("tests/integration/test_backend_intent_flow_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "import app" not in text
    assert "from app" not in text
    assert "streamlit" not in text.lower()


def test_backend_intent_smoke_uses_backend_nodes_only():
    text = Path("tests/integration/test_backend_intent_flow_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "intent_router_node" in text
    assert "advisory_answer_node" in text
    assert "plan_only_node" in text
    assert "execute_pending_plan_node" in text