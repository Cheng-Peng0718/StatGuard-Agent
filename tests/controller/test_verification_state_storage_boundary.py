import json

from core.controller.backend_turn import (
    _finish,
    _normalize_state_verifications_for_storage,
    _verification_status,
)


class FakeVerification:
    def __init__(self):
        self.status = "needs_review"
        self.feedback = "Requires approval."
        self.error_code = None
        self.details = {"action_hash": "abc123"}


def test_finish_serializes_current_verification_before_returning_state():
    result = _finish(
        state={
            "current_verification": FakeVerification(),
            "user_request": "clean the data",
            "messages": [],
        },
        node_trace=[],
    )

    stored_verification = result.state["current_verification"]

    assert isinstance(stored_verification, dict)
    assert stored_verification["status"] == "needs_review"
    assert stored_verification["details"] == {"action_hash": "abc123"}

    json.dumps(result.state)


def test_storage_normalizer_serializes_current_verification():
    state = _normalize_state_verifications_for_storage({
        "current_verification": FakeVerification(),
    })

    assert isinstance(state["current_verification"], dict)
    assert state["current_verification"]["status"] == "needs_review"

    json.dumps(state)


def test_verification_status_reads_dict_and_object():
    assert _verification_status({
        "current_verification": FakeVerification(),
    }) == "needs_review"

    assert _verification_status({
        "current_verification": {
            "status": "allowed",
            "feedback": "ok",
            "details": {},
        },
    }) == "allowed"