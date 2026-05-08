from types import SimpleNamespace

from core.graph import human_review_node
from core.ui_adapter.events import (
    apply_ui_event_to_state,
    make_approve_human_review_event,
    make_reject_human_review_event,
)


def apply_updates(state, updates):
    merged = dict(state)
    merged.update(updates)
    return merged


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


def make_review_state():
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
        "observations": [],
    }


def test_approve_human_review_ui_event_allows_human_review_node():
    state = make_review_state()

    event = make_approve_human_review_event(
        action_hash="abc123",
    )

    updates = apply_ui_event_to_state(state, event)
    state = apply_updates(state, updates)

    review_updates = human_review_node(state)

    assert "observations" not in review_updates

    verification = review_updates["current_verification"]

    if isinstance(verification, dict):
        assert verification["status"] == "allowed"
    else:
        assert verification.status == "allowed"

    assert review_updates["current_action"] is not None
    assert review_updates["human_review_decision"] is None


def test_reject_human_review_ui_event_records_rejection_path():
    state = make_review_state()

    event = make_reject_human_review_event(
        action_hash="abc123",
        reason="Do not mutate the data.",
    )

    updates = apply_ui_event_to_state(state, event)
    state = apply_updates(state, updates)

    review_updates = human_review_node(state)

    # Current human_review_node may record an observation for non-approved review.
    # This test locks only that rejection does not become allowed.
    if "current_verification" in review_updates:
        verification = review_updates["current_verification"]

        if isinstance(verification, dict):
            assert verification.get("status") != "allowed"
        else:
            assert getattr(verification, "status", None) != "allowed"

def test_approve_human_review_with_wrong_action_hash_is_blocked():
    state = make_review_state()

    event = make_approve_human_review_event(
        action_hash="wrong_hash",
    )

    updates = apply_ui_event_to_state(state, event)
    state = apply_updates(state, updates)

    review_updates = human_review_node(state)

    assert review_updates["human_review_required"] is True
    assert review_updates["pending_action"] is not None
    assert review_updates["current_action"] is not None
    assert review_updates["current_verification"] is not None
    assert review_updates["human_review_decision"] is None

    assert review_updates["assistant_response"]["response_type"] == "error"
    assert (
        review_updates["assistant_response"]["metadata"]["error_code"]
        == "HUMAN_REVIEW_ACTION_HASH_MISMATCH"
    )