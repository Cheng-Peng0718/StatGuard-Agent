from __future__ import annotations

from typing import Any

from core.audit.execution_state import audit_execution_state
from core.audit.state_serialization import audit_state_serialization


def as_plain_dict(obj: Any) -> dict:
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if hasattr(obj, "model_dump"):
        return obj.model_dump()

    return {}


def merge_state_for_audit(state: dict, updates: dict) -> dict:
    """
    Build a best-effort post-node state snapshot for backend audit.

    This does not mutate graph state. It only gives audit_execution_state()
    a view of what state will look like after this node's updates.
    """
    merged = dict(state)

    for key, value in updates.items():
        if key in {"observations", "analysis_runs", "data_versions", "data_audit_log"}:
            old_value = merged.get(key, []) or []
            new_value = value or []

            if isinstance(old_value, list) and isinstance(new_value, list):
                merged[key] = old_value + new_value
            else:
                merged[key] = value
        else:
            merged[key] = value

    return merged


def compact_state_serialization_audit(audit_result) -> dict:
    """
    Store only compact serialization-audit metadata in GraphState.

    Do NOT store audit_result.safe_state inside GraphState, because that would
    duplicate the entire state and can create huge nested state snapshots.
    """
    return {
        "status": audit_result.status,
        "n_issues": len(audit_result.issues),
        "issues": [
            issue.model_dump()
            if hasattr(issue, "model_dump")
            else dict(issue)
            for issue in audit_result.issues
        ],
    }


def attach_state_serialization_audit(state: dict, updates: dict) -> dict:
    """
    Observe-only serialization audit.

    This does not mutate business state, does not block routing, and does not
    persist the full safe_state. It only records whether the post-update state
    contains objects that may be unsafe for checkpoint/UI serialization.
    """
    audit_state = dict(state)
    audit_state.update(updates)

    # Avoid recursively auditing/storing older audit snapshots.
    audit_state.pop("state_serialization_audit", None)

    audit_result = audit_state_serialization(audit_state)
    updates["state_serialization_audit"] = compact_state_serialization_audit(
        audit_result
    )

    if audit_result.status != "ok":
        print("\n" + "=" * 40)
        print("[STATE SERIALIZATION AUDIT]")
        print(updates["state_serialization_audit"])
        print("=" * 40 + "\n")

    return updates


def attach_execution_audit(state: dict, updates: dict) -> dict:
    """
    Run backend execution-state audit after a node update.

    S11B is observe-only: audit findings are recorded but do not alter routing.
    """
    audit_state = merge_state_for_audit(state, updates)
    audit_result = audit_execution_state(audit_state)

    updates["execution_audit"] = audit_result.model_dump()

    if audit_result.status != "ok":
        print("\n" + "=" * 40)
        print("[EXECUTION AUDIT]")
        print(audit_result.model_dump())
        print("=" * 40 + "\n")

    return attach_state_serialization_audit(state, updates)