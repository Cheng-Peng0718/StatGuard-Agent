from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from core.verification_access import (
    get_verification_details,
    get_verification_error_code,
    get_verification_feedback,
    get_verification_status,
)


def verification_to_state_dict(verification: Any) -> dict | None:
    """
    Convert verification runtime object/dict into JSON-safe state payload.

    This intentionally accepts both VerificationResult-style objects and plain
    dict payloads so workflow and human-review state remain JSON-safe.
    """
    if verification is None:
        return None

    if isinstance(verification, Mapping):
        payload = dict(verification)
    elif hasattr(verification, "model_dump"):
        payload = verification.model_dump()
    elif hasattr(verification, "dict"):
        payload = verification.dict()
    else:
        payload = {
            "status": get_verification_status(verification),
            "feedback": get_verification_feedback(verification),
            "error_code": get_verification_error_code(verification),
            "details": get_verification_details(verification),
        }

    if payload.get("details") is None:
        payload["details"] = {}

    if not isinstance(payload.get("details"), dict):
        payload["details"] = {}

    return payload