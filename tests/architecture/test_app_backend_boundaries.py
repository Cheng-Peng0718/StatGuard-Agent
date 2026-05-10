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

def test_plan_runner_uses_turn_contract_not_workflow_nodes():
    text = Path("core/app_backend/plan_runner.py").read_text(
        encoding="utf-8"
    )

    assert "run_user_turn" in text
    assert "build_ui_snapshot" in text

    forbidden = [
        "create_graph_app",
        "run_graph_once",
        "core.workflow.nodes",
        "execute_pending_plan_node",
        "verify_node",
        "execute_node",
        "summarize_node",
        "streamlit",
        "st.",
    ]

    for phrase in forbidden:
        assert phrase not in text

def test_session_backend_does_not_import_graph_nodes_or_ui():
    text = Path("core/app_backend/session.py").read_text(
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


def test_app_backend_public_api_exports_ui_contract_functions():
    text = Path("core/app_backend/__init__.py").read_text(
        encoding="utf-8"
    )

    expected = [
        "create_app_session",
        "initialize_dataset_session_from_file",
        "run_user_turn",
        "run_pending_plan_until_pause",
        "build_ui_snapshot",
    ]

    for name in expected:
        assert name in text


def test_backend_contract_smoke_tests_use_public_app_backend_api():
    text = Path("tests/app_backend/test_backend_contract_smoke.py").read_text(
        encoding="utf-8"
    )

    expected = [
        "create_app_session",
        "initialize_dataset_session_from_file",
        "run_user_turn",
        "run_pending_plan_until_pause",
    ]

    for name in expected:
        assert name in text

    forbidden = [
        "core.workflow.nodes",
        "create_graph_app",
        "execute_pending_plan_node",
        "verify_node",
        "execute_node",
        "summarize_node",
    ]

    for phrase in forbidden:
        assert phrase not in text

def test_backend_turn_does_not_reappear():
    forbidden_paths = [
        Path("core/controller/backend_turn.py"),
        Path("core/backend_turn.py"),
        Path("backend_turn.py"),
    ]

    for path in forbidden_paths:
        assert not path.exists()


def test_app_backend_names_do_not_hide_backend_turn_reimplementation():
    app_backend_dir = Path("core/app_backend")

    for path in app_backend_dir.glob("*.py"):
        text = path.read_text(encoding="utf-8")

        forbidden = [
            "backend_turn",
            "run_backend_turn",
            "controller.backend_turn",
        ]

        for phrase in forbidden:
            assert phrase not in text

def test_upload_and_context_refresh_use_canonical_tabular_reader():
    upload_text = Path("core/app_backend/dataset_upload.py").read_text(
        encoding="utf-8"
    )
    context_text = Path("core/data/context_refresh.py").read_text(
        encoding="utf-8"
    )

    assert "load_tabular_dataframe" in upload_text
    assert "load_tabular_dataframe" in context_text

    assert "pd.read_excel" not in upload_text
    assert "pd.read_csv" not in upload_text

def test_tabular_reader_centralizes_excel_engine_selection():
    text = Path("core/data/tabular_io.py").read_text(encoding="utf-8")

    assert 'engine="openpyxl"' in text
    assert 'engine="xlrd"' in text
    assert "SUPPORTED_TABULAR_EXTENSIONS" in text