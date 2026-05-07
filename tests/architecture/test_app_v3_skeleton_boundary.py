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

def test_app_v3_active_workspace_renders_choice_controls():
    text = Path("ui/components/active_workspace.py").read_text(encoding="utf-8")

    required = [
        "render_choice_form",
        "on_save_choices",
        "required_user_choices",
        "Save choices for this step",
        "st.form",
        "multiselect",
        "selectbox",
    ]

    for item in required:
        assert item in text


def test_app_v3_active_workspace_prioritizes_human_review():
    text = Path("ui/components/active_workspace.py").read_text(encoding="utf-8")

    assert "determine_active_focus" in text
    assert "human_review.get(\"required\")" in text
    assert "return \"human_review\"" in text
    assert "render_human_review_focus" in text


def test_app_v3_uses_update_plan_step_choices_event():
    text = Path("ui/app_v3.py").read_text(encoding="utf-8")

    assert "make_update_plan_step_choices_event" in text
    assert "def on_save_choices" in text
    assert "on_save_choices=on_save_choices" in text


def test_app_v3_action_bar_prioritizes_human_review_message():
    text = Path("ui/components/action_bar.py").read_text(encoding="utf-8")

    assert "Human review is required" in text
    assert "Approve" in text
    assert "Reject" in text