from __future__ import annotations

from typing import Any, Dict

import streamlit as st


def _text(value: Any) -> str:
    if value is None:
        return "none"
    return str(value)


def _icon(value: Any) -> str:
    text = _text(value)

    if text in {"ok", "ready", "completed", "allowed"}:
        return "✅"

    if text in {
        "needs_review",
        "needs_user_choice",
        "blocked_no_ready_steps",
        "blocked_pending_data_cleaning",
    }:
        return "⚠️"

    if text in {"failed", "error", "blocked", "rejected"}:
        return "❌"

    return "ℹ️"


def render_status_pill(label: str, value: Any) -> None:
    st.markdown(f"{_icon(value)} **{label}:** `{_text(value)}`")


def render_system_status(snapshot: Dict[str, Any]) -> None:
    plan = snapshot.get("plan") or {}
    data = snapshot.get("data") or {}
    audits = snapshot.get("audits") or {}

    execution_audit = audits.get("execution_audit") or {}
    serialization_audit = audits.get("state_serialization_audit") or {}

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        render_status_pill("Data", data.get("active_data_version_id"))

    with col2:
        render_status_pill("Plan", plan.get("plan_status"))

    with col3:
        render_status_pill("Execution", plan.get("plan_execution_status"))

    with col4:
        render_status_pill("Audit", execution_audit.get("status"))

    if execution_audit.get("status") not in {None, "ok"}:
        st.error("Execution audit has issues.")

    if serialization_audit.get("status") not in {None, "ok"}:
        st.error("State serialization audit has issues.")