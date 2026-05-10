from pathlib import Path


UI_FILES = [
    "app.py",
    "ui/state.py",
    "ui/panels.py",
    "ui/renderers.py",
]


def test_app_py_is_thin_layout_orchestrator():
    text = Path("app.py").read_text(encoding="utf-8")

    assert "def render_layout" in text
    assert "def main" in text

    forbidden = [
        "initialize_dataset_session_from_file",
        "run_user_turn",
        "run_pending_plan_until_pause",
        "st.file_uploader",
        "st.chat_input",
        "st.dataframe",
        "st.json",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_ui_package_does_not_import_graph_or_workflow_nodes():
    forbidden = [
        "core.graph",
        "create_graph_app",
        "run_graph_once",
        "core.workflow.nodes",
        "execute_pending_plan_node",
        "verify_node",
        "execute_node",
        "summarize_node",
        "validate_plugin_action",
        "execute_analysis_tool",
    ]

    for path in UI_FILES:
        text = Path(path).read_text(encoding="utf-8")

        for phrase in forbidden:
            assert phrase not in text


def test_ui_panels_use_app_backend_public_api_only():
    text = Path("ui/panels.py").read_text(encoding="utf-8")

    expected = [
        "initialize_dataset_session_from_file",
        "run_user_turn",
        "run_pending_plan_until_pause",
    ]

    for phrase in expected:
        assert phrase in text

    forbidden = [
        "core.runtime",
        "core.graph",
        "core.workflow",
        "core.analysis_tool_plugins",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_ui_state_owns_session_state_helpers():
    text = Path("ui/state.py").read_text(encoding="utf-8")

    assert "def ensure_session_state" in text
    assert "def current_snapshot" in text
    assert "def sync_assistant_response_to_chat" in text