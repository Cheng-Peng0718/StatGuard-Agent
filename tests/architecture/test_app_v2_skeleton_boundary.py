from pathlib import Path


def test_app_v2_exists():
    assert Path("ui/app_v2.py").exists()


def test_app_v2_uses_ui_adapters_and_backend_controller():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    required = [
        "apply_ui_event_to_state",
        "build_ui_snapshot",
        "run_backend_turn",
        "make_user_message_event",
        "make_run_plan_event",
        "make_approve_human_review_event",
        "make_reject_human_review_event",
        "prepare_uploaded_dataset_state",
    ]

    for item in required:
        assert item in text


def test_app_v2_does_not_import_raw_graph_nodes():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    forbidden = [
        "from core.graph import",
        "import core.graph",
        "execute_node",
        "verify_node",
        "summarize_node",
        "deliverable_gate_node",
        "intent_router_node",
        "plan_only_node",
        "execute_pending_plan_node",
    ]

    offenders = [
        item
        for item in forbidden
        if item in text
    ]

    assert offenders == []


def test_app_v2_does_not_mutate_backend_runtime_fields_directly():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    forbidden = [
        'backend_state["current_action"]',
        'backend_state["current_execution"]',
        'backend_state["current_verification"]',
        'backend_state["human_review_decision"]',
        'backend_state["user_request"]',
        "backend_state['current_action']",
        "backend_state['current_execution']",
        "backend_state['current_verification']",
        "backend_state['human_review_decision']",
        "backend_state['user_request']",
    ]

    offenders = [
        item
        for item in forbidden
        if item in text
    ]

    assert offenders == []


def test_app_v2_declares_expected_session_state_keys():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    expected = [
        "backend_state",
        "ui_snapshot",
        "chat_history",
        "uploaded_dataset_info",
        "last_error",
    ]

    for key in expected:
        assert key in text

def test_app_v2_upload_uses_dataset_upload_adapter():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    assert "prepare_uploaded_dataset_state" in text
    assert "pd.read_csv" in text

    forbidden_direct_state_fragments = [
        'backend_state["data_versions"]',
        'backend_state["active_data_version_id"]',
        'backend_state["dataset_profile"]',
        "backend_state['data_versions']",
        "backend_state['active_data_version_id']",
        "backend_state['dataset_profile']",
    ]

    offenders = [
        fragment
        for fragment in forbidden_direct_state_fragments
        if fragment in text
    ]

    assert offenders == []

def test_app_v2_bootstraps_project_root_before_core_imports():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")
    lines = text.splitlines()

    bootstrap_idx = None
    first_core_import_idx = None

    for idx, line in enumerate(lines):
        if "sys.path.insert" in line:
            bootstrap_idx = idx

        if line.startswith("from core.") or line.startswith("import core."):
            first_core_import_idx = idx
            break

    assert bootstrap_idx is not None
    assert first_core_import_idx is not None
    assert bootstrap_idx < first_core_import_idx

def test_app_v2_renders_blocked_no_ready_steps_message():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    assert "blocked_no_ready_steps" in text
    assert "No executable plan step is currently ready" in text
    assert "required_user_choices" in text
    assert "candidate_variables" in text

def test_app_v2_renders_plan_choice_controls():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    assert "render_plan_step_choice_controls" in text
    assert "make_update_plan_step_choices_event" in text
    assert "required_user_choices" in text
    assert "Save choices for this step" in text
    assert "blocked_no_ready_steps" in text

def test_app_v2_choice_controls_not_nested_under_reason_condition():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    assert "render_plan_step_choice_controls" in text

    bad_pattern = (
        "if reason:\n"
        "            st.caption(reason)\n"
        "\n"
        "            render_plan_step_choice_controls"
    )

    assert bad_pattern not in text

def test_app_v2_renders_pending_data_cleaning_block():
    text = Path("ui/app_v2.py").read_text(encoding="utf-8")

    assert "blocked_pending_data_cleaning" in text
    assert "waiting for data cleaning" in text