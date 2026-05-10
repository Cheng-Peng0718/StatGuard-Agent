from pathlib import Path

from core.runtime.graph_runner import run_graph_once


class FakeGraphApp:
    def __init__(self):
        self.seen_state = None
        self.seen_config = None

    def invoke(self, state, config=None):
        self.seen_state = state
        self.seen_config = config
        result = dict(state)
        result["ran_graph"] = True
        if config is not None:
            result["saw_config"] = True
        return result


def test_graph_runner_invokes_compiled_graph(monkeypatch):
    fake_app = FakeGraphApp()

    monkeypatch.setattr(
        "core.graph.create_graph_app",
        lambda: fake_app,
    )

    result = run_graph_once({
        "user_request": "What does the data look like?",
    })

    assert result["user_request"] == "What does the data look like?"
    assert result["ran_graph"] is True
    assert fake_app.seen_state["user_request"] == "What does the data look like?"


def test_graph_runner_passes_config(monkeypatch):
    fake_app = FakeGraphApp()

    monkeypatch.setattr(
        "core.graph.create_graph_app",
        lambda: fake_app,
    )

    result = run_graph_once(
        {"user_request": "Run the plan."},
        config={"configurable": {"thread_id": "test_thread"}},
    )

    assert result["ran_graph"] is True
    assert result["saw_config"] is True
    assert fake_app.seen_config == {
        "configurable": {
            "thread_id": "test_thread",
        }
    }


def test_graph_runner_does_not_import_workflow_nodes_or_ui():
    text = Path("core/runtime/graph_runner.py").read_text(encoding="utf-8")

    forbidden = [
        "core.workflow.nodes",
        "intent_router_node",
        "verify_node",
        "execute_node",
        "summarize_node",
        "build_ui_snapshot",
        "core.ui_adapter",
        "core.controller",
    ]

    for item in forbidden:
        assert item not in text