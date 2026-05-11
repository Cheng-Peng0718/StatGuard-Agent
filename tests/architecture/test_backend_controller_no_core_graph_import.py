from pathlib import Path


def test_backend_controller_does_not_import_core_graph():
    text = Path("core/controller/backend_turn.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.graph import" not in text
    assert "import core.graph" not in text


def test_legacy_entrypoints_do_not_expect_global_compiled_graph_app():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "app = create_graph_app()" not in graph_text
    assert "def create_graph_app" in graph_text