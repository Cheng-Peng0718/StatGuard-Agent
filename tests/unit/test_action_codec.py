import json

from core.action_codec import action_from_state, action_to_state_dict
from core.schema import ActionProposal


def test_action_to_state_dict_serializes_action_proposal():
    action = ActionProposal(
        action_id="act_1",
        action_type="tool_call",
        tool_name="get_summary_stats",
        arguments={"columns": ["GPA"]},
        reasoning_summary="Compute summary statistics.",
    )

    payload = action_to_state_dict(action)

    assert isinstance(payload, dict)
    assert payload["action_id"] == "act_1"
    assert payload["tool_name"] == "get_summary_stats"
    assert payload["arguments"] == {"columns": ["GPA"]}

    json.dumps(payload)


def test_action_from_state_rehydrates_dict_to_action_proposal():
    payload = {
        "action_id": "act_2",
        "action_type": "tool_call",
        "tool_name": "run_multiple_regression",
        "arguments": {
            "target_col": "GPA",
            "feature_cols": ["SATM"],
        },
        "reasoning_summary": "Run regression.",
    }

    action = action_from_state(payload)

    assert isinstance(action, ActionProposal)
    assert action.action_id == "act_2"
    assert action.tool_name == "run_multiple_regression"
    assert action.arguments == {
        "target_col": "GPA",
        "feature_cols": ["SATM"],
    }


def test_action_codec_normalizes_legacy_summary_field():
    payload = {
        "action_id": "act_3",
        "action_type": "tool_call",
        "tool_name": "get_summary_stats",
        "arguments": {},
        "summary": "Legacy summary.",
    }

    action = action_from_state(payload)

    assert action.reasoning_summary == "Legacy summary."


def test_action_codec_handles_none():
    assert action_to_state_dict(None) is None
    assert action_from_state(None) is None