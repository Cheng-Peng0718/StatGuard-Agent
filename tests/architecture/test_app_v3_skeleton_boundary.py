from pathlib import Path


APP_V3 = Path("ui/app_v3.py")


def test_app_v3_exists():
    assert APP_V3.exists()


def test_app_v3_uses_backend_adapter_controller_boundary():
    text = APP_V3.read_text(encoding="utf-8")

    required = [
        "apply_ui_event_to_state",
        "run_backend_turn",
        "build_ui_snapshot",
        "prepare_uploaded_dataset_state",
        "make_user_message_event",
        "make_run_plan_event",
        "make_approve_human_review_event",
        "make_reject_human_review_event",
    ]

    for item in required:
        assert item in text


def test_app_v3_does_not_call_graph_nodes_directly():
    text = APP_V3.read_text(encoding="utf-8")

    forbidden = [
        "verify_node(",
        "execute_node(",
        "summarize_node(",
        "execute_pending_plan_node(",
        "plan_only_node(",
        "from core.graph import",
        "import core.graph",
    ]

    offenders = [
        item
        for item in forbidden
        if item in text
    ]

    assert offenders == []


def test_app_v3_component_files_exist():
    expected = [
        "ui/components/system_status.py",
        "ui/components/chat_panel.py",
        "ui/components/plan_timeline.py",
        "ui/components/active_workspace.py",
        "ui/components/action_bar.py",
        "ui/components/debug_panel.py",
        "ui/styles/app_v3.css",
    ]

    for path in expected:
        assert Path(path).exists()


def test_app_v3_mentions_one_screen_sections():
    text = APP_V3.read_text(encoding="utf-8")

    required = [
        "render_system_status",
        "render_chat_panel",
        "render_active_workspace",
        "render_plan_timeline",
        "render_action_bar",
        "render_debug_panel",
    ]

    for item in required:
        assert item in text