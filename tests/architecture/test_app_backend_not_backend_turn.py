from pathlib import Path


APP_BACKEND_FILES = [
    "core/app_backend/session.py",
    "core/app_backend/dataset_upload.py",
    "core/app_backend/turn.py",
    "core/app_backend/plan_runner.py",
    "core/app_backend/snapshot.py",
]


def _read(path: str) -> str:
    return Path(path).read_text(encoding="utf-8")


def test_app_backend_does_not_import_workflow_nodes():
    forbidden = [
        "core.workflow.nodes",
        "build_context_node",
        "intent_router_node",
        "advisory_answer_node",
        "plan_only_node",
        "execute_pending_plan_node",
        "supervisor_node",
        "verify_node",
        "human_review_node",
        "execute_node",
        "summarize_node",
        "deliverable_gate_node",
        "final_response_node",
    ]

    for path in APP_BACKEND_FILES:
        text = _read(path)

        for phrase in forbidden:
            assert phrase not in text, f"{path} must not import/call {phrase}"


def test_app_backend_does_not_reimplement_graph_routing():
    forbidden = [
        "route_after_intent",
        "route_after_supervisor",
        "route_after_verify",
        "route_after_review",
        "route_after_summarize",
        "route_after_execute_pending_plan",
        "workflow.add_node",
        "workflow.add_edge",
        "workflow.add_conditional_edges",
        "StateGraph",
        "END",
    ]

    for path in APP_BACKEND_FILES:
        text = _read(path)

        for phrase in forbidden:
            assert phrase not in text, f"{path} must not reimplement graph routing"


def test_only_graph_runner_may_call_create_graph_app_inside_runtime_path():
    turn_text = _read("core/app_backend/turn.py")
    plan_runner_text = _read("core/app_backend/plan_runner.py")
    graph_runner_text = _read("core/runtime/graph_runner.py")

    assert "create_graph_app" not in turn_text
    assert "create_graph_app" not in plan_runner_text

    assert "from core.graph import create_graph_app" in graph_runner_text
    assert "app.invoke" in graph_runner_text


def test_turn_backend_is_thin_graph_runner_adapter():
    text = _read("core/app_backend/turn.py")

    assert "run_graph_once" in text
    assert "build_ui_snapshot" in text

    forbidden = [
        "execute_analysis_tool",
        "validate_plugin_action",
        "build_context(",
        "call_supervisor",
        "execute_pending_plan_node",
        "summarize_node",
        "verify_node",
    ]

    for phrase in forbidden:
        assert phrase not in text


def test_plan_runner_uses_turn_contract_not_graph_or_nodes():
    text = _read("core/app_backend/plan_runner.py")

    assert "run_user_turn" in text
    assert "build_ui_snapshot" in text

    forbidden = [
        "run_graph_once",
        "create_graph_app",
        "execute_pending_plan_node",
        "validate_plugin_action",
        "execute_analysis_tool",
        "summarize_node",
        "verify_node",
    ]

    for phrase in forbidden:
        assert phrase not in text