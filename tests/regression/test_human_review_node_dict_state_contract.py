from core.graph import human_review_node


def test_human_review_node_accepts_dict_action_and_dict_verification_for_needs_review():
    state = {
        "current_action": {
            "action_id": "act_1",
            "action_type": "tool_call",
            "tool_name": "clean_data",
            "arguments": {
                "action_type": "drop",
                "strategy": "rows",
                "columns": ["GPA"],
            },
            "reasoning_summary": "Drop missing GPA rows.",
        },
        "current_verification": {
            "action_id": "act_1",
            "status": "needs_review",
            "feedback": "Cleaning data requires confirmation.",
            "error_code": None,
            "details": {
                "action_hash": "hash_1",
                "canonical_arguments": {
                    "action_type": "drop",
                    "strategy": "rows",
                    "columns": ["GPA"],
                },
            },
        },
        "human_review_decision": None,
    }

    updates = human_review_node(state)

    assert updates["human_review_required"] is True
    assert isinstance(updates["pending_action"], dict)
    assert updates["pending_action"]["action_id"] == "act_1"
    assert updates["current_action"] is not None
    assert updates["current_verification"] is not None

    # Waiting for human confirmation is runtime review state,
    # not an observation/execution/rejection record.
    assert "observations" not in updates

def test_human_review_node_accepts_dict_action_and_dict_verification_for_rejection():
    state = {
        "current_action": {
            "action_id": "act_2",
            "action_type": "tool_call",
            "tool_name": "run_multiple_regression",
            "arguments": {},
            "reasoning_summary": "Run regression.",
        },
        "current_verification": {
            "action_id": "act_2",
            "status": "rejected_recoverable",
            "feedback": "Missing required arguments.",
            "error_code": "SCHEMA_VALIDATION_FAILED",
            "details": {},
        },
        "human_review_decision": None,
    }

    updates = human_review_node(state)

    obs = updates["observations"][0]
    assert obs["source_action_id"] == "act_2"
    assert obs["tool_name"] == "run_multiple_regression"
    assert obs["raw_data"]["verification"]["status"] == "rejected_recoverable"

def test_human_review_node_records_explicit_user_rejection():
    state = {
        "current_action": {
            "action_id": "act_reject",
            "action_type": "tool_call",
            "tool_name": "clean_data",
            "arguments": {
                "action_type": "drop",
                "strategy": "rows",
                "columns": ["GPA"],
            },
            "reasoning_summary": "Drop missing GPA rows.",
        },
        "current_verification": {
            "action_id": "act_reject",
            "status": "needs_review",
            "feedback": "Cleaning data requires confirmation.",
            "error_code": None,
            "details": {
                "action_hash": "hash_reject",
                "canonical_arguments": {
                    "action_type": "drop",
                    "strategy": "rows",
                    "columns": ["GPA"],
                },
            },
        },
        "human_review_decision": "rejected",
        "human_review_rejection_reason": "Do not mutate the data.",
    }

    updates = human_review_node(state)

    assert updates["human_review_required"] is False
    assert updates["pending_action"] is None
    assert updates["human_review_decision"] is None
    assert updates["human_review_rejection_reason"] is None
    assert updates["current_action"] is None
    assert updates["current_verification"] is None

    obs = updates["observations"][0]
    assert obs["source_action_id"] == "act_reject"
    assert obs["tool_name"] == "clean_data"
    assert obs["error_code"] == "HUMAN_REVIEW_REJECTED"
    assert obs["raw_data"]["human_review_decision"] == "rejected"