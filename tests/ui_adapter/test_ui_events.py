import json
import pytest

from core.ui_adapter.events import (
    apply_ui_event_to_state,
    make_approve_human_review_event,
    make_cancel_plan_event,
    make_reject_human_review_event,
    make_run_plan_event,
    make_user_message_event,
    normalize_ui_event,
)


def test_make_user_message_event_and_apply_to_state():
    event = make_user_message_event("What can I do with this dataset?")

    assert event["schema_version"] == "ui_event_v1"
    assert event["event_type"] == "user_message"
    assert event["payload"]["message"] == "What can I do with this dataset?"

    updates = apply_ui_event_to_state({}, event)

    assert updates["user_request"] == "What can I do with this dataset?"
    assert updates["latest_ui_event"]["event_type"] == "user_message"

    json.dumps(updates)


def test_user_message_requires_non_empty_message():
    event = {
        "event_type": "user_message",
        "payload": {
            "message": "   ",
        },
    }

    with pytest.raises(ValueError):
        apply_ui_event_to_state({}, event)


def test_run_plan_event_sets_user_request_to_run_plan():
    event = make_run_plan_event()

    updates = apply_ui_event_to_state({}, event)

    assert updates["user_request"] == "run the plan"
    assert updates["latest_ui_event"]["event_type"] == "run_plan"

    json.dumps(updates)


def test_approve_human_review_event_sets_approval_state():
    event = make_approve_human_review_event(
        action_hash="abc123",
    )

    updates = apply_ui_event_to_state({}, event)

    assert updates["human_review_decision"] == "approved"
    assert updates["human_review_action_hash"] == "abc123"
    assert updates["latest_ui_event"]["event_type"] == "approve_human_review"

    json.dumps(updates)


def test_reject_human_review_event_sets_rejection_state():
    event = make_reject_human_review_event(
        action_hash="abc123",
        reason="Too risky.",
    )

    updates = apply_ui_event_to_state({}, event)

    assert updates["human_review_decision"] == "rejected"
    assert updates["human_review_rejection_reason"] == "Too risky."
    assert updates["human_review_action_hash"] == "abc123"

    json.dumps(updates)


def test_cancel_plan_event_clears_runtime_state():
    event = make_cancel_plan_event(reason="User cancelled.")

    updates = apply_ui_event_to_state({}, event)

    assert updates["plan_status"] == "cancelled"
    assert updates["plan_execution_status"] == "cancelled"
    assert updates["current_action"] is None
    assert updates["current_execution"] is None
    assert updates["current_verification"] is None
    assert updates["current_plan_step_id"] is None
    assert updates["action_origin"] is None

    json.dumps(updates)


def test_select_plan_step_event_sets_selected_step_id():
    event = {
        "event_type": "select_plan_step",
        "payload": {
            "step_id": "s2",
        },
    }

    updates = apply_ui_event_to_state({}, event)

    assert updates["selected_plan_step_id"] == "s2"
    assert updates["latest_ui_event"]["event_type"] == "select_plan_step"

    json.dumps(updates)


def test_clear_runtime_event_clears_runtime_state():
    event = {
        "event_type": "clear_runtime",
        "payload": {},
    }

    updates = apply_ui_event_to_state({}, event)

    assert updates["current_action"] is None
    assert updates["current_execution"] is None
    assert updates["current_verification"] is None
    assert updates["current_plan_step_id"] is None
    assert updates["action_origin"] is None

    json.dumps(updates)


def test_normalize_ui_event_fills_missing_metadata_and_event_id():
    event = normalize_ui_event({
        "event_type": "run_plan",
    })

    assert event["event_id"].startswith("ui_evt_")
    assert event["schema_version"] == "ui_event_v1"
    assert event["payload"] == {}
    assert event["metadata"] == {}

    json.dumps(event)

def test_update_plan_step_choices_event_updates_pending_plan_step():
    state = {
        "pending_plan": {
            "plan_id": "plan_1",
            "steps": [
                {
                    "step_id": "s1",
                    "tool_name": "run_multiple_regression",
                    "status": "needs_user_choice",
                    "execution_ready": False,
                    "execution_status": "not_started",
                    "variables": {},
                    "arguments": {},
                    "required_user_choices": ["target_col", "feature_cols"],
                }
            ],
        }
    }

    event = {
        "event_type": "update_plan_step_choices",
        "payload": {
            "step_id": "s1",
            "choices": {
                "target_col": "GPA",
                "feature_cols": ["SATM"],
            },
        },
    }

    updates = apply_ui_event_to_state(state, event)

    step = updates["pending_plan"]["steps"][0]

    assert step["status"] == "ready"
    assert step["execution_ready"] is True
    assert step["required_user_choices"] == []

    assert step["variables"]["target_col"] == "GPA"
    assert step["variables"]["feature_cols"] == ["SATM"]
    assert step["arguments"]["target_col"] == "GPA"
    assert step["arguments"]["feature_cols"] == ["SATM"]

    assert updates["assistant_response"]["response_type"] == "plan_step_choices_updated"

    json.dumps(updates)

def test_update_plan_step_choices_supports_clean_data_choices():
    state = {
        "pending_plan": {
            "plan_id": "plan_clean",
            "steps": [
                {
                    "step_id": "s_clean",
                    "tool_name": "clean_data",
                    "status": "needs_user_choice",
                    "execution_ready": False,
                    "execution_status": "not_started",
                    "variables": {},
                    "arguments": {},
                    "required_user_choices": ["action_type", "strategy", "columns"],
                }
            ],
        }
    }

    event = {
        "event_type": "update_plan_step_choices",
        "payload": {
            "step_id": "s_clean",
            "choices": {
                "action_type": "drop",
                "strategy": "rows",
                "columns": ["GPA", "SATM"],
            },
        },
    }

    updates = apply_ui_event_to_state(state, event)

    step = updates["pending_plan"]["steps"][0]

    assert step["status"] == "ready"
    assert step["execution_ready"] is True
    assert step["required_user_choices"] == []
    assert step["arguments"]["action_type"] == "drop"
    assert step["arguments"]["strategy"] == "rows"
    assert step["arguments"]["columns"] == ["GPA", "SATM"]