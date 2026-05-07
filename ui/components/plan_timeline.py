from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st


def _step_icon(step: Dict[str, Any]) -> str:
    execution_status = step.get("execution_status")
    status = step.get("status")

    if execution_status == "completed":
        return "●"

    if execution_status == "failed" or status == "failed":
        return "✕"

    if status == "needs_user_choice":
        return "◑"

    if status == "ready":
        return "◐"

    return "○"


def _compact_title(step: Dict[str, Any]) -> str:
    return step.get("title") or step.get("tool_name") or step.get("step_id") or "Step"


def render_plan_timeline(snapshot: Dict[str, Any]) -> None:
    st.subheader("Plan")

    plan_section = snapshot.get("plan") or {}
    pending_plan = plan_section.get("pending_plan")
    current_step_id = plan_section.get("current_plan_step_id")

    st.caption(f"Status: `{plan_section.get('plan_status')}`")
    st.caption(f"Execution: `{plan_section.get('plan_execution_status')}`")

    if not pending_plan:
        st.info("No plan yet.")
        return

    steps: List[Dict[str, Any]] = pending_plan.get("steps") or []

    with st.container(height=520):
        for step in steps:
            icon = _step_icon(step)
            title = _compact_title(step)
            tool = step.get("tool_name")
            status = step.get("status")
            step_id = step.get("step_id")

            active_marker = " ← current" if step_id == current_step_id else ""

            st.markdown(
                f"{icon} **{title}**{active_marker}  \n"
                f"<span class='app-v3-muted'>`{tool}` · `{status}`</span>",
                unsafe_allow_html=True,
            )