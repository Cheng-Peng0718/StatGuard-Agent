from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def get_verification_field(
    verification: Any,
    field_name: str,
    default: Any = None,
) -> Any:
    if verification is None:
        return default

    if isinstance(verification, Mapping):
        return verification.get(field_name, default)

    return getattr(verification, field_name, default)


def get_verification_status(verification: Any, default: str | None = None) -> str | None:
    return get_verification_field(verification, "status", default)


def get_verification_feedback(verification: Any, default: str | None = None) -> str | None:
    return get_verification_field(verification, "feedback", default)


def get_verification_error_code(
    verification: Any,
    default: str | None = None,
) -> str | None:
    return get_verification_field(verification, "error_code", default)


def get_verification_details(verification: Any) -> dict:
    details = get_verification_field(verification, "details", {})

    if isinstance(details, dict):
        return dict(details)

    return {}


def set_verification_fields(verification: Any, **fields: Any) -> Any:
    """
    Return verification with updated fields.

    Supports both dict verification payloads and Pydantic/object verification
    instances during the migration period.
    """
    if verification is None:
        return None

    if isinstance(verification, Mapping):
        updated = dict(verification)
        updated.update(fields)
        return updated

    for field_name, value in fields.items():
        try:
            setattr(verification, field_name, value)
        except Exception:
            pass

    return verification