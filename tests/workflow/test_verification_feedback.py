from core.workflow.verification_feedback import attach_verification_blocked_response


def test_attach_verification_blocked_response_for_recoverable_failure():
    state = {
        "active_data_version_id": "raw_v1",
        "current_action": {
            "action_id": "act_clean",
            "action_type": "tool_call",
            "tool_name": "clean_data",
            "arguments": {},
        },
    }

    updates = {
        "current_verification": {
            "status": "rejected_recoverable",
            "feedback": "Missing required arguments.",
            "error_code": "INVALID_TOOL_ARGUMENTS",
            "details": {},
        },
        "repair_decision": {
            "status": "repairable",
            "tool_name": "clean_data",
            "error_code": "INVALID_TOOL_ARGUMENTS",
        },
        "repair_proposal": {
            "proposal_type": "argument_repair",
            "source_tool_name": "clean_data",
            "source_action_id": "act_clean",
            "reason": "Arguments can be normalized.",
        },
    }

    result = attach_verification_blocked_response(state, updates)

    assert result["assistant_response"]["response_type"] == "error"
    assert result["assistant_response"]["metadata"]["semantic_type"] == "verification_blocked"
    assert result["assistant_response"]["source_node"] == "verify"
    assert "clean_data" in result["assistant_response"]["content"]
    assert "INVALID_TOOL_ARGUMENTS" in result["assistant_response"]["content"]
    assert (
        result["assistant_response"]["metadata"]["verification_status"]
        == "rejected_recoverable"
    )
    assert (
        result["assistant_response"]["metadata"]["repair_decision"]["status"]
        == "repairable"
    )


def test_attach_verification_blocked_response_for_terminal_failure():
    state = {
        "current_action": {
            "action_id": "act_unknown",
            "action_type": "tool_call",
            "tool_name": "unknown_tool",
            "arguments": {},
        },
    }

    updates = {
        "current_verification": {
            "status": "rejected_terminal",
            "feedback": "Unknown tool.",
            "error_code": "TOOL_NOT_REGISTERED",
            "details": {},
        },
        "repair_decision": {
            "status": "terminal",
            "tool_name": "unknown_tool",
            "error_code": "TOOL_NOT_REGISTERED",
        },
        "repair_proposal": {
            "proposal_type": "no_op",
            "source_tool_name": "unknown_tool",
            "source_action_id": "act_unknown",
            "reason": "No repair is possible.",
        },
    }

    result = attach_verification_blocked_response(state, updates)

    assert result["assistant_response"]["response_type"] == "error"
    assert result["assistant_response"]["metadata"]["semantic_type"] == "verification_blocked"
    assert "terminal" in result["assistant_response"]["content"].lower()
    assert (
        result["assistant_response"]["metadata"]["verification_status"]
        == "rejected_terminal"
    )


def test_attach_verification_blocked_response_does_not_override_existing_response():
    state = {}

    updates = {
        "current_verification": {
            "status": "rejected_recoverable",
            "feedback": "Bad args.",
            "error_code": "INVALID_TOOL_ARGUMENTS",
            "details": {},
        },
        "assistant_response": {
            "response_type": "existing",
            "content": "Keep me.",
        },
    }

    result = attach_verification_blocked_response(state, updates)

    assert result["assistant_response"]["response_type"] == "existing"
    assert result["assistant_response"]["content"] == "Keep me."