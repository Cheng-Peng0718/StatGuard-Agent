from types import SimpleNamespace

from core.graph import human_review_node


def make_action():
    return SimpleNamespace(
        action_id="act_clean",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA"],
        },
    )


def test_human_review_approval_converts_needs_review_to_allowed_without_observation():
    state = {
        "current_action": make_action(),
        "current_verification": {
            "status": "needs_review",
            "feedback": "Human confirmation required.",
            "details": {
                "action_hash": "abc123",
                "canonical_arguments": {
                    "action_type": "drop",
                    "strategy": "rows",
                    "columns": ["GPA"],
                },
                "requires_confirmation": True,
            },
        },
        "human_review_decision": "approved",
        "observations": [],
    }

    result = human_review_node(state)

    assert "observations" not in result

    verification = result["current_verification"]

    if isinstance(verification, dict):
        assert verification["status"] == "allowed"
        assert "approved" in verification["feedback"].lower()
    else:
        assert verification.status == "allowed"
        assert "approved" in verification.feedback.lower()

    assert result["current_action"] is not None
    assert result["human_review_required"] is False
    assert result["pending_action"] is None
    assert result["human_review_decision"] is None