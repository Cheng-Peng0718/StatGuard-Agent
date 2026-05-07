from pathlib import Path


def test_old_app_file_is_treated_as_legacy_ui_boundary():
    app_path = Path("app.py")

    # app.py may or may not exist in some backend-only deployments.
    # If it exists, it is treated as legacy UI, not backend source of truth.
    if not app_path.exists():
        return

    doc_text = Path("docs/16_ui_adapter_contract.md").read_text(encoding="utf-8")

    assert "The old `app.py` is considered legacy UI." in doc_text
    assert "Do not add new business logic to app.py." in doc_text


def test_backend_graph_does_not_import_old_app():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    assert "import app" not in graph_text
    assert "from app" not in graph_text


def test_backend_ui_adapter_does_not_import_old_app():
    for path in Path("core/ui_adapter").rglob("*.py"):
        text = path.read_text(encoding="utf-8")

        assert "import app" not in text
        assert "from app" not in text