from pathlib import Path


def test_backend_smoke_tests_do_not_import_ui_app():
    text = Path("tests/integration/test_backend_graph_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "import app" not in text
    assert "from app" not in text
    assert "streamlit" not in text.lower()


def test_backend_smoke_tests_use_backend_nodes_only():
    text = Path("tests/integration/test_backend_graph_smoke.py").read_text(
        encoding="utf-8"
    )

    assert "summarize_node" in text
    assert "deliverable_gate_node" in text
    assert "final_response_node" in text