from core.controller.backend_turn import _action_to_graph_object
from core.schema import ActionProposal


def test_dict_current_action_is_rehydrated_as_action_proposal_contract():
    raw_action = {
        "action_id": "act_1",
        "action_type": "tool_call",
        "tool_name": "get_summary_stats",
        "arguments": {
            "columns": ["GPA"],
        },
        "reasoning_summary": "Compute summary statistics for GPA.",
    }

    action = _action_to_graph_object(raw_action)

    assert isinstance(action, ActionProposal)
    assert action.action_id == "act_1"
    assert action.action_type == "tool_call"
    assert action.tool_name == "get_summary_stats"
    assert action.arguments == {"columns": ["GPA"]}
    assert action.reasoning_summary == "Compute summary statistics for GPA."


def test_dict_current_action_with_legacy_summary_is_normalized_to_reasoning_summary():
    raw_action = {
        "action_id": "act_2",
        "action_type": "tool_call",
        "tool_name": "get_summary_stats",
        "arguments": {},
        "summary": "Legacy summary field.",
    }

    action = _action_to_graph_object(raw_action)

    assert isinstance(action, ActionProposal)
    assert action.reasoning_summary == "Legacy summary field."