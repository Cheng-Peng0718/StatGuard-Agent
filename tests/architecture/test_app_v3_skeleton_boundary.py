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

def test_app_v3_dataset_upload_button_is_always_rendered():
    text = Path("ui/components/active_workspace.py").read_text(encoding="utf-8")

    assert "Load dataset" in text
    assert "disabled=uploaded_file is None" in text
    assert "on_dataset_upload(df, uploaded_file.name)" in text

def test_app_v3_renders_report_panel_without_legacy_app():
    text = Path("ui/app_v3.py").read_text(encoding="utf-8")

    assert "render_report_panel" in text
    assert "from ui.components.report_panel import render_report_panel" in text
    assert "app.py" not in text


def test_app_v3_report_panel_uses_report_export_adapter():
    text = Path("ui/components/report_panel.py").read_text(encoding="utf-8")

    assert "build_report_package_from_state" in text
    assert "st.download_button" in text
    assert "Download Markdown" in text
    assert "Download HTML" in text

    forbidden = [
        "build_markdown_report(",
        "build_html_report_from_state(",
        "from core.graph",
        "execute_node",
        "verify_node",
        "summarize_node",
    ]

    offenders = [
        item
        for item in forbidden
        if item in text
    ]

    assert offenders == []

def test_app_v3_action_bar_is_inside_workspace_not_below_right_sidebar():
    text = Path("ui/app_v3.py").read_text(encoding="utf-8")
    main_text = text.split("def main()", 1)[1]

    assert "workspace, right = st.columns" in main_text
    assert "with workspace:" in main_text
    assert "with right:" in main_text
    assert "render_action_bar(" in main_text

    assert (
        main_text.index("with workspace:")
        < main_text.index("render_action_bar(")
        < main_text.index("with right:")
    )


def test_app_v3_action_bar_is_sticky():
    text = Path("ui/styles/app_v3.css").read_text(encoding="utf-8")

    assert ".app-v3-action-bar" in text
    assert "position: sticky" in text
    assert "bottom: 0" in text

def test_app_v3_active_workspace_uses_insight_cards():
    text = Path("ui/components/active_workspace.py").read_text(encoding="utf-8")

    assert "build_insight_card_from_run" in text
    assert "What was computed" in text
    assert "Key findings" in text
    assert "Caveats" in text
    assert "Recommended next steps" in text

def test_insight_cards_builder_does_not_branch_on_tool_names():
    text = Path("core/ui_adapter/insight_cards.py").read_text(
        encoding="utf-8"
    )

    forbidden_fragments = [
        "if tool_name ==",
        "elif tool_name ==",
        "match tool_name",
    ]

    offenders = [
        item
        for item in forbidden_fragments
        if item in text
    ]

    assert offenders == []


def test_insight_cards_use_insight_spec_registry():
    text = Path("core/ui_adapter/insight_cards.py").read_text(
        encoding="utf-8"
    )

    assert "get_insight_spec" in text
    assert "spec.display_name" in text
    assert "spec.recommended_next_steps" in text