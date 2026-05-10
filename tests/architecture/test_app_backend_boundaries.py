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


def test_dataset_upload_backend_does_not_invoke_graph_or_import_ui():
    text = Path("core/app_backend/dataset_upload.py").read_text(
        encoding="utf-8"
    )

    forbidden = [
        "create_graph_app",
        "run_graph_once",
        "core.workflow.nodes",
        "streamlit",
        "st.",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_dataset_upload_backend_uses_data_version_and_snapshot_contracts():
    text = Path("core/app_backend/dataset_upload.py").read_text(
        encoding="utf-8"
    )

    assert "create_initial_data_version" in text
    assert "refresh_dataset_context_from_path" in text
    assert "build_ui_snapshot" in text

def test_turn_backend_uses_graph_runner_and_snapshot_contracts_only():
    text = Path("core/app_backend/turn.py").read_text(encoding="utf-8")

    assert "run_graph_once" in text
    assert "build_ui_snapshot" in text

    forbidden = [
        "create_graph_app",
        "core.workflow.nodes",
        "workflow.add_node",
        "streamlit",
        "st.",
    ]

    for phrase in forbidden:
        assert phrase not in text