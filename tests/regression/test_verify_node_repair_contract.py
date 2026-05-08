from core.graph import verify_node
from core.schema import ActionProposal


def test_verify_node_attaches_repair_state_for_recoverable_verification_failure(monkeypatch):
    action = ActionProposal(
        action_id="act_clean_bad",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "drop rows",
            "strategy": "drop",
            "columns": ["GPA"],
        },
        reasoning_summary="Clean data with invalid aliases.",
    )

    def fake_verify(action, profile):
        return (
            "rejected_recoverable",
            "Invalid clean_data arguments.",
            {
                "status": "rejected_recoverable",
                "feedback": "Invalid clean_data arguments.",
                "error_code": "INVALID_TOOL_ARGUMENTS",
                "details": {
                    "tool_name": "clean_data",
                },
            },
        )

    monkeypatch.setattr("core.graph.verify", fake_verify)

    updates = verify_node({
        "current_action": action,
        "dataset_profile": {
            "columns": ["GPA"],
        },
        "observations": [],
        "repair_attempts": [],
    })

    assert updates["current_verification"]["status"] == "rejected_recoverable"
    assert updates["observations"][0]["status"] == "rejected"

    assert "repair_decision" in updates
    assert updates["repair_decision"]["tool_name"] == "clean_data"
    assert updates["repair_decision"]["error_code"] == "INVALID_TOOL_ARGUMENTS"
    assert updates["repair_decision"]["status"] in {"repairable", "needs_user"}

    assert "repair_proposal" in updates
    assert updates["repair_proposal"]["source_tool_name"] == "clean_data"
    assert updates["repair_proposal"]["source_action_id"] == "act_clean_bad"


def test_verify_node_attaches_terminal_repair_state_for_terminal_verification_failure(monkeypatch):
    action = ActionProposal(
        action_id="act_unknown",
        action_type="tool_call",
        tool_name="unknown_tool",
        arguments={},
        reasoning_summary="Try unknown tool.",
    )

    def fake_verify(action, profile):
        return (
            "rejected_terminal",
            "Unknown tool.",
            {
                "status": "rejected_terminal",
                "feedback": "Unknown tool.",
                "error_code": "UNKNOWN_TOOL",
                "details": {
                    "tool_name": "unknown_tool",
                },
            },
        )

    monkeypatch.setattr("core.graph.verify", fake_verify)

    updates = verify_node({
        "current_action": action,
        "dataset_profile": {},
        "observations": [],
        "repair_attempts": [],
    })

    assert updates["current_verification"]["status"] == "rejected_terminal"
    assert updates["observations"][0]["status"] == "rejected"

    assert "repair_decision" in updates
    assert updates["repair_decision"]["status"] == "terminal"
    assert updates["repair_decision"]["tool_name"] == "unknown_tool"

    assert "repair_proposal" in updates
    assert updates["repair_proposal"]["proposal_type"] == "no_op"