from pathlib import Path


def test_app_backend_snapshot_does_not_import_workflow_nodes_or_ui():
    text = Path("core/app_backend/snapshot.py").read_text(encoding="utf-8")

    forbidden = [
        "core.workflow.nodes",
        "create_graph_app",
        "run_graph_once",
        "streamlit",
        "st.",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_app_backend_package_exists():
    assert Path("core/app_backend/__init__.py").exists()
    assert Path("core/app_backend/snapshot.py").exists()