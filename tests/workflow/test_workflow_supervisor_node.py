from core.schema import ActionProposal
from core.workflow.nodes import supervisor as supervisor_module


class DummyContext:
    context_text = "dummy context"


def test_supervisor_node_sets_current_action(monkeypatch):
    def fake_build_context(**kwargs):
        return DummyContext()

    def fake_call_supervisor(context_pkg):
        return ActionProposal(
            action_id="act_1",
            action_type="tool_call",
            tool_name="get_summary_stats",
            arguments={"columns": ["GPA"]},
            reasoning_summary="Compute summary statistics.",
        )

    monkeypatch.setattr(supervisor_module, "build_context", fake_build_context)
    monkeypatch.setattr(supervisor_module, "call_supervisor", fake_call_supervisor)

    updates = supervisor_module.supervisor_node({
        "workspace_dir": "./tmp",
        "dataset_profile": {"columns": []},
        "user_request": "do summary stats",
        "observations": [],
        "current_step": 1,
        "max_steps": 12,
    })

    assert updates["current_action"].action_id == "act_1"
    assert updates["current_action"].tool_name == "get_summary_stats"
    assert "task_contract" not in updates


def test_supervisor_node_extracts_task_contract_from_dict_action(monkeypatch):
    def fake_build_context(**kwargs):
        return DummyContext()

    def fake_call_supervisor(context_pkg):
        return {
            "action_id": "act_final",
            "action_type": "final_answer",
            "tool_name": None,
            "arguments": {},
            "reasoning_summary": "Prepare final answer.",
            "task_contract": {
                "required_tools": ["get_summary_stats"],
                "required_deliverables": ["brief summary"],
            },
        }

    monkeypatch.setattr(supervisor_module, "build_context", fake_build_context)
    monkeypatch.setattr(supervisor_module, "call_supervisor", fake_call_supervisor)

    updates = supervisor_module.supervisor_node({
        "workspace_dir": "./tmp",
        "dataset_profile": {"columns": []},
        "user_request": "give me final answer",
        "observations": [],
        "current_step": 1,
        "max_steps": 12,
    })

    assert updates["current_action"]["action_id"] == "act_final"
    assert updates["task_contract"] == {
        "required_tools": ["get_summary_stats"],
        "required_deliverables": ["brief summary"],
    }