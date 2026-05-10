from __future__ import annotations

from typing import Any, Dict, List, Optional


def _as_dict(value: Any) -> Dict[str, Any]:
    if value is None:
        return {}

    if isinstance(value, dict):
        return dict(value)

    if hasattr(value, "model_dump"):
        dumped = value.model_dump()
        return dict(dumped) if isinstance(dumped, dict) else {}

    if hasattr(value, "dict"):
        dumped = value.dict()
        return dict(dumped) if isinstance(dumped, dict) else {}

    return {}


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def build_ui_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal graph state into a stable UI-facing snapshot.

    UI code should consume this snapshot instead of reading graph state directly.
    This module does not invoke LangGraph, execute tools, or mutate data.
    """
    state = dict(state or {})

    dataset_profile_v2 = _as_dict(state.get("dataset_profile_v2"))
    dataset_summary = _as_dict(state.get("dataset_summary"))
    dataset_context = _as_dict(state.get("dataset_context"))

    pending_plan = _as_dict(state.get("pending_plan"))
    assistant_response = _as_dict(state.get("assistant_response"))

    current_verification = _as_dict(state.get("current_verification"))
    current_execution = _as_dict(state.get("current_execution"))

    return {
        "schema_version": "ui_snapshot_v2",
        "assistant_response": assistant_response,
        "dataset": {
            "dataset_name": (
                dataset_profile_v2.get("dataset_name")
                or dataset_context.get("dataset_name")
                or state.get("dataset_name")
            ),
            "active_data_version_id": state.get("active_data_version_id"),
            "data_versions": _as_list(state.get("data_versions")),
            "profile": dataset_profile_v2,
            "summary": dataset_summary,
            "context": dataset_context,
        },
        "plan": {
            "pending_plan": pending_plan or None,
            "plan_id": pending_plan.get("plan_id"),
            "plan_status": state.get("plan_status") or pending_plan.get("status"),
            "plan_execution_status": state.get("plan_execution_status"),
            "current_plan_step_id": state.get("current_plan_step_id"),
        },
        "analysis": {
            "observations": _as_list(state.get("observations")),
            "analysis_runs": _as_list(state.get("analysis_runs")),
            "current_execution": current_execution or None,
            "execution_audit": _as_dict(state.get("execution_audit")),
        },
        "review": {
            "human_review_required": bool(state.get("human_review_required")),
            "current_verification": current_verification or None,
        },
        "metadata": {
            "current_step": state.get("current_step"),
            "interaction_intent": state.get("interaction_intent"),
            "intent_decision": _as_dict(state.get("intent_decision")),
            "active_data_version_id": state.get("active_data_version_id"),
        },
    }