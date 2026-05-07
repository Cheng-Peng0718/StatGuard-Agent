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