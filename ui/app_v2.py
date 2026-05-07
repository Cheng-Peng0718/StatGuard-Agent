from __future__ import annotations

from typing import Any, Dict

import streamlit as st

from core.controller.backend_turn import run_backend_turn
from core.ui_adapter.events import (
    apply_ui_event_to_state,
    make_approve_human_review_event,
    make_reject_human_review_event,
    make_run_plan_event,
    make_user_message_event,
)
from core.ui_adapter.snapshot import build_ui_snapshot


APP_TITLE = "Analysis Agent V2"


def make_initial_backend_state() -> Dict[str, Any]:
    """
    Create the initial backend state for UI v2.

    This is intentionally minimal. Dataset upload/loading will be wired in a
    later step. The UI must not construct actions, verifications, executions,
    or tool results directly.
    """
    return {
        "user_request": "",
        "current_step": 0,
        "max_steps": 5,
        "observations": [],
        "analysis_runs": [],
        "data_versions": [],
        "data_audit_log": [],
        "active_data_version_id": None,
        "pending_plan": None,
        "plan_status": None,
        "plan_execution_status": None,
        "current_plan_step_id": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,
        "repair_attempts": [],
    }


def init_session_state() -> None:
    if "backend_state" not in st.session_state:
        st.session_state.backend_state = make_initial_backend_state()

    if "ui_snapshot" not in st.session_state:
        st.session_state.ui_snapshot = build_ui_snapshot(
            st.session_state.backend_state
        )

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    if "uploaded_dataset_info" not in st.session_state:
        st.session_state.uploaded_dataset_info = None

    if "last_error" not in st.session_state:
        st.session_state.last_error = None


def refresh_snapshot() -> Dict[str, Any]:
    snapshot = build_ui_snapshot(st.session_state.backend_state)
    st.session_state.ui_snapshot = snapshot
    return snapshot


def submit_ui_event(event: Dict[str, Any]) -> None:
    """
    Apply a UI event and run one backend-controller turn.

    This is the only place where UI events enter the backend.
    """
    try:
        updates = apply_ui_event_to_state(
            st.session_state.backend_state,
            event,
        )
        st.session_state.backend_state.update(updates)

        result = run_backend_turn(st.session_state.backend_state)

        st.session_state.backend_state = result["state"]
        st.session_state.ui_snapshot = result["ui_snapshot"]
        st.session_state.last_error = None

    except Exception as exc:
        st.session_state.last_error = str(exc)
        refresh_snapshot()


def render_header() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    st.title(APP_TITLE)
    st.caption(
        "V2 UI skeleton. This UI talks to the backend only through "
        "UIEvent and UISnapshot adapters."
    )


def render_last_error() -> None:
    if st.session_state.last_error:
        st.error(st.session_state.last_error)


def render_chat_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Chat")

    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.write(content)

    user_text = st.chat_input("Ask about your dataset...")

    if user_text:
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": user_text,
            }
        )

        submit_ui_event(
            make_user_message_event(user_text)
        )

        response = st.session_state.ui_snapshot.get("assistant_response")

        if response and response.get("content"):
            st.session_state.chat_history.append(
                {
                    "role": "assistant",
                    "content": response["content"],
                }
            )

        st.rerun()


def render_assistant_response(snapshot: Dict[str, Any]) -> None:
    st.subheader("Assistant Response")

    response = snapshot.get("assistant_response")

    if not response:
        st.info("No assistant response yet.")
        return

    st.markdown(response.get("content") or "")

    with st.expander("Response metadata", expanded=False):
        st.json(response)


def render_plan_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Plan")

    plan_section = snapshot.get("plan") or {}
    pending_plan = plan_section.get("pending_plan")

    col1, col2 = st.columns([1, 3])

    with col1:
        if st.button("Run plan", use_container_width=True):
            submit_ui_event(make_run_plan_event())
            st.rerun()

    with col2:
        st.write(f"Plan status: `{plan_section.get('plan_status')}`")
        st.write(
            f"Execution status: `{plan_section.get('plan_execution_status')}`"
        )

    if not pending_plan:
        st.info("No pending plan.")
        return

    steps = pending_plan.get("steps") or []

    for step in steps:
        with st.container(border=True):
            st.write(f"**{step.get('step_id')} — {step.get('title', step.get('tool_name'))}**")
            st.write(f"Tool: `{step.get('tool_name')}`")
            st.write(f"Status: `{step.get('status')}`")
            st.write(f"Execution: `{step.get('execution_status')}`")

            reason = step.get("reason")
            if reason:
                st.caption(reason)


def render_human_review_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Human Review")

    human_review = snapshot.get("human_review") or {}

    if not human_review.get("required"):
        st.info("No human review required.")
        return

    st.warning(human_review.get("feedback") or "Action requires review.")

    action = human_review.get("action") or {}

    st.write(f"Tool: `{action.get('tool_name')}`")
    st.json(action.get("arguments") or {})

    action_hash = human_review.get("action_hash")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Approve", type="primary", use_container_width=True):
            submit_ui_event(
                make_approve_human_review_event(
                    action_hash=action_hash,
                )
            )
            st.rerun()

    with col2:
        if st.button("Reject", use_container_width=True):
            submit_ui_event(
                make_reject_human_review_event(
                    action_hash=action_hash,
                    reason="Rejected from UI.",
                )
            )
            st.rerun()


def render_analysis_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Analysis Results")

    analysis = snapshot.get("analysis") or {}
    runs = analysis.get("analysis_runs") or []

    if not runs:
        st.info("No analysis results yet.")
        return

    for idx, run in enumerate(runs, start=1):
        with st.container(border=True):
            st.write(f"**Run {idx}: {run.get('tool_name')}**")
            st.write(f"Status: `{run.get('status')}`")
            st.write(f"Success: `{run.get('success')}`")

            summary = run.get("summary") or run.get("message")
            if summary:
                st.write(summary)

            with st.expander("Run details", expanded=False):
                st.json(run)


def render_data_versions_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Data Versions")

    data = snapshot.get("data") or {}

    st.write(f"Active version: `{data.get('active_data_version_id')}`")

    versions = data.get("data_versions") or []

    if not versions:
        st.info("No data versions yet.")
        return

    st.json(versions)


def render_repair_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Repair / Debug")

    repair = snapshot.get("repair") or {}

    with st.expander("Repair state", expanded=False):
        st.json(repair)


def render_audit_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Audits")

    audits = snapshot.get("audits") or {}

    with st.expander("Audit state", expanded=False):
        st.json(audits)


def render_snapshot_debug(snapshot: Dict[str, Any]) -> None:
    with st.expander("Raw UI snapshot", expanded=False):
        st.json(snapshot)


def main() -> None:
    init_session_state()
    snapshot = refresh_snapshot()

    render_header()
    render_last_error()

    left, right = st.columns([2, 1])

    with left:
        render_chat_panel(snapshot)
        render_assistant_response(snapshot)
        render_plan_panel(snapshot)
        render_human_review_panel(snapshot)
        render_analysis_panel(snapshot)

    with right:
        render_data_versions_panel(snapshot)
        render_repair_panel(snapshot)
        render_audit_panel(snapshot)
        render_snapshot_debug(snapshot)


if __name__ == "__main__":
    main()