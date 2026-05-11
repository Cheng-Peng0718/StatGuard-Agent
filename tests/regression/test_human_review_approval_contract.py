from types import SimpleNamespace

from core.workflow.nodes.human_review import human_review_node


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


def make_needs_review_state():
    return {
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
        "human_review_action_hash": "abc123",
        "observations": [],
    }


def test_human_review_approval_converts_needs_review_to_allowed_without_observation():
    state = make_needs_review_state()

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


def test_human_review_approval_with_mismatched_action_hash_does_not_allow_execution():
    state = make_needs_review_state()
    state["human_review_action_hash"] = "wrong_hash"

    result = human_review_node(state)

    assert result["human_review_required"] is True
    assert result["pending_action"] is not None
    assert result["current_action"] is not None
    assert result["current_verification"] is not None
    assert result["human_review_decision"] is None

    assert result["assistant_response"]["response_type"] == "error"
    assert (
        result["assistant_response"]["metadata"]["error_code"]
        == "HUMAN_REVIEW_ACTION_HASH_MISMATCH"
    )

    verification = result["current_verification"]

    if isinstance(verification, dict):
        assert verification["status"] == "needs_review"
    else:
        assert verification.status == "needs_review"

    # Mismatched approval must not create execution/rejection observations.
    assert "observations" not in result