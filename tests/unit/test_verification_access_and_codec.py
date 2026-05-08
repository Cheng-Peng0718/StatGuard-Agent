import json

from core.verification_access import (
    get_verification_details,
    get_verification_error_code,
    get_verification_feedback,
    get_verification_status,
    set_verification_fields,
)
from core.verification_codec import verification_to_state_dict


class FakeVerification:
    def __init__(self):
        self.status = "needs_review"
        self.feedback = "Requires approval."
        self.error_code = None
        self.details = {"action_hash": "abc123"}


class FakeModelDumpVerification:
    def model_dump(self):
        return {
            "status": "allowed",
            "feedback": "Looks good.",
            "error_code": None,
            "details": {"action_hash": "xyz789"},
        }


def test_verification_access_supports_object():
    verification = FakeVerification()

    assert get_verification_status(verification) == "needs_review"
    assert get_verification_feedback(verification) == "Requires approval."
    assert get_verification_error_code(verification) is None
    assert get_verification_details(verification) == {"action_hash": "abc123"}


def test_verification_access_supports_dict():
    verification = {
        "status": "rejected_recoverable",
        "feedback": "Missing required argument.",
        "error_code": "SCHEMA_VALIDATION_FAILED",
        "details": {"field": "target_col"},
    }

    assert get_verification_status(verification) == "rejected_recoverable"
    assert get_verification_feedback(verification) == "Missing required argument."
    assert get_verification_error_code(verification) == "SCHEMA_VALIDATION_FAILED"
    assert get_verification_details(verification) == {"field": "target_col"}


def test_set_verification_fields_updates_object():
    verification = FakeVerification()

    updated = set_verification_fields(
        verification,
        status="allowed",
        feedback="Approved by user.",
    )

    assert updated.status == "allowed"
    assert updated.feedback == "Approved by user."


def test_set_verification_fields_updates_dict_without_mutating_original():
    verification = {
        "status": "needs_review",
        "feedback": "Requires approval.",
        "details": {},
    }

    updated = set_verification_fields(
        verification,
        status="allowed",
        feedback="Approved.",
    )

    assert verification["status"] == "needs_review"
    assert updated["status"] == "allowed"
    assert updated["feedback"] == "Approved."


def test_verification_codec_serializes_model_dump_object():
    payload = verification_to_state_dict(FakeModelDumpVerification())

    assert payload["status"] == "allowed"
    assert payload["details"] == {"action_hash": "xyz789"}

    json.dumps(payload)


def test_verification_codec_serializes_plain_object():
    payload = verification_to_state_dict(FakeVerification())

    assert payload["status"] == "needs_review"
    assert payload["feedback"] == "Requires approval."
    assert payload["details"] == {"action_hash": "abc123"}

    json.dumps(payload)


def test_verification_codec_handles_none():
    assert verification_to_state_dict(None) is None