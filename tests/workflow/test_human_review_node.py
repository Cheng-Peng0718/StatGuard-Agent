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
        "human_review_decision": None,
        "human_review_action_hash": None,
    }


def test_human_review_node_creates_review_state_without_observation():
    state = make_needs_review_state()

    updates = human_review_node(state)

    assert updates["human_review_required"] is True
    assert updates["pending_action"]["action_id"] == "act_clean"
    assert updates["current_action"] is not None
    assert updates["current_verification"] is not None
    assert "observations" not in updates


def test_human_review_node_blocks_mismatched_action_hash():
    state = make_needs_review_state()
    state["human_review_decision"] = "approved"
    state["human_review_action_hash"] = "wrong_hash"

    updates = human_review_node(state)

    assert updates["human_review_required"] is True
    assert updates["pending_action"] is not None
    assert updates["assistant_response"]["response_type"] == "error"
    assert (
        updates["assistant_response"]["metadata"]["error_code"]
        == "HUMAN_REVIEW_ACTION_HASH_MISMATCH"
    )
    assert "observations" not in updates