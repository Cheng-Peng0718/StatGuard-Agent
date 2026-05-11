from core.workflow.routes import route_after_verify


def test_recoverable_verification_rejection_stops_turn_even_with_repair_proposal():
    state = {
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
        },
    }

    assert route_after_verify(state) == "end"


def test_terminal_verification_rejection_stops_turn():
    state = {
        "current_verification": {
            "status": "rejected_terminal",
            "feedback": "Unknown tool.",
            "error_code": "TOOL_NOT_REGISTERED",
            "details": {},
        },
        "repair_decision": {
            "status": "terminal",
            "tool_name": "unknown_tool",
        },
    }

    assert route_after_verify(state) == "end"