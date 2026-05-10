from pathlib import Path


def test_streamlit_ui_uses_app_backend_public_api_only():
    text = "\n".join([
        Path("app.py").read_text(encoding="utf-8"),
        Path("ui/state.py").read_text(encoding="utf-8"),
        Path("ui/panels.py").read_text(encoding="utf-8"),
        Path("ui/renderers.py").read_text(encoding="utf-8"),
    ])

    expected = [
        "create_app_session",
        "initialize_dataset_session_from_file",
        "run_user_turn",
        "run_pending_plan_until_pause",
    ]

    for name in expected:
        assert name in text

    forbidden = [
        "create_graph_app",
        "run_graph_once",
        "core.workflow.nodes",
        "build_context_node",
        "intent_router_node",
        "execute_pending_plan_node",
        "supervisor_node",
        "verify_node",
        "execute_node",
        "summarize_node",
        "validate_plugin_action",
        "execute_analysis_tool",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_streamlit_ui_renders_snapshot_instead_of_deep_state_contracts():
    text = "\n".join([
        Path("app.py").read_text(encoding="utf-8"),
        Path("ui/state.py").read_text(encoding="utf-8"),
        Path("ui/panels.py").read_text(encoding="utf-8"),
        Path("ui/renderers.py").read_text(encoding="utf-8"),
    ])
    forbidden = [
        '["dataset_profile_v2"]',
        '["dataset_profile"]',
        '["pending_plan"]',
        '["analysis_runs"]',
        '["observations"]',
        '["current_verification"]',
        '["current_execution"]',
    ]

    for phrase in forbidden:
        assert phrase not in text

    assert "snapshot" in text