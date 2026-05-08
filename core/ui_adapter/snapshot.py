from __future__ import annotations

import json
from typing import Any, Dict

from core.audit.state_serialization import make_checkpoint_safe_state
from core.execution_codec import normalize_execution_view


UI_SNAPSHOT_SCHEMA_VERSION = "ui_snapshot_v1"


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def _get_field(value: Any, field_name: str, default=None):
    if value is None:
        return default

    if isinstance(value, dict):
        return value.get(field_name, default)

    return getattr(value, field_name, default)


def _safe_list(value: Any) -> list:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def _build_action_snapshot(action: Any) -> Dict[str, Any] | None:
    """
    Convert current_action into a UI-safe action summary.

    This intentionally supports dict, Pydantic models, and simple objects.
    UI should not need the full internal object.
    """
    if action is None:
        return None

    action_dict = _as_dict(action)

    if not action_dict:
        action_dict = {
            "action_id": _get_field(action, "action_id"),
            "action_type": _get_field(action, "action_type"),
            "tool_name": _get_field(action, "tool_name"),
            "arguments": _get_field(action, "arguments", {}) or {},
            "reasoning_summary": _get_field(action, "reasoning_summary"),
        }

    return {
        "action_id": action_dict.get("action_id"),
        "action_type": action_dict.get("action_type"),
        "tool_name": action_dict.get("tool_name"),
        "arguments": action_dict.get("arguments") or {},
        "reasoning_summary": (
            action_dict.get("reasoning_summary")
            or action_dict.get("summary")
            or action_dict.get("message")
        ),
    }


def _build_verification_snapshot(verification: Any) -> Dict[str, Any] | None:
    if verification is None:
        return None

    verification_dict = _as_dict(verification)

    if not verification_dict:
        verification_dict = {
            "status": _get_field(verification, "status"),
            "feedback": _get_field(verification, "feedback"),
            "error_code": _get_field(verification, "error_code"),
            "details": _get_field(verification, "details", {}) or {},
        }

    return {
        "status": verification_dict.get("status"),
        "feedback": verification_dict.get("feedback"),
        "error_code": verification_dict.get("error_code"),
        "details": verification_dict.get("details") or {},
    }


def _build_execution_snapshot(execution: Any) -> Dict[str, Any] | None:
    execution_view = normalize_execution_view(execution)

    if execution_view is None:
        return None

    return {
        "execution_id": execution_view.get("execution_id"),
        "status": execution_view.get("status"),
        "success": execution_view.get("success"),
        "error_code": execution_view.get("error_code"),
        "message": execution_view.get("message"),
    }


def _build_human_review_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    verification = _build_verification_snapshot(
        state.get("current_verification")
    )
    action = _build_action_snapshot(
        state.get("current_action")
    )

    verification_status = (verification or {}).get("status")

    required = verification_status == "needs_review"

    return {
        "required": required,
        "status": verification_status,
        "feedback": (verification or {}).get("feedback"),
        "action": action if required else None,
        "action_hash": (
            (verification or {})
            .get("details", {})
            .get("action_hash")
        ),
        "requires_confirmation": (
            (verification or {})
            .get("details", {})
            .get("requires_confirmation")
        ),
    }


def _build_runtime_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    action = _build_action_snapshot(state.get("current_action"))
    execution = _build_execution_snapshot(state.get("current_execution"))
    verification = _build_verification_snapshot(state.get("current_verification"))

    return {
        "has_current_action": action is not None,
        "has_current_execution": execution is not None,
        "has_current_verification": verification is not None,
        "current_action": action,
        "current_execution": execution,
        "current_verification": verification,
        "current_plan_step_id": state.get("current_plan_step_id"),
        "action_origin": state.get("action_origin"),
        "plan_execution_status": state.get("plan_execution_status"),
    }


def _build_repair_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "decision": state.get("repair_decision"),
        "proposal": state.get("repair_proposal"),
        "attempts": _safe_list(state.get("repair_attempts")),
    }


def _build_audits_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "execution_audit": state.get("execution_audit"),
        "state_serialization_audit": state.get("state_serialization_audit"),
        "deliverable_check": state.get("deliverable_check"),
    }


def build_ui_snapshot(state: Any) -> Dict[str, Any]:
    """
    Build a stable, JSON-safe snapshot for UI consumption.

    UI should read this snapshot instead of inspecting raw GraphState internals.
    This function does not mutate state, execute tools, call LLMs, or change routing.
    """
    raw_state = state if isinstance(state, dict) else _as_dict(state)
    safe_state = make_checkpoint_safe_state(raw_state)

    snapshot = {
        "schema_version": UI_SNAPSHOT_SCHEMA_VERSION,

        "assistant_response": safe_state.get("assistant_response"),

        "plan": {
            "pending_plan": safe_state.get("pending_plan"),
            "plan_status": safe_state.get("plan_status"),
            "plan_execution_status": safe_state.get("plan_execution_status"),
        },

        "analysis": {
            "observations": _safe_list(safe_state.get("observations")),
            "analysis_runs": _safe_list(safe_state.get("analysis_runs")),
        },

        "data": {
            "active_data_version_id": safe_state.get("active_data_version_id"),
            "data_versions": _safe_list(safe_state.get("data_versions")),
            "data_audit_log": _safe_list(safe_state.get("data_audit_log")),
            "uploaded_dataset_info": safe_state.get("uploaded_dataset_info"),
            "dataset_summary": safe_state.get("dataset_summary"),
        },

        "human_review": _build_human_review_snapshot(raw_state),

        "runtime": _build_runtime_snapshot(raw_state),

        "repair": _build_repair_snapshot(safe_state),

        "audits": _build_audits_snapshot(safe_state),
    }

    # Final safety normalization.
    snapshot = make_checkpoint_safe_state(snapshot)

    # Fail early during tests/development if snapshot is not JSON-safe.
    json.dumps(snapshot)

    return snapshot