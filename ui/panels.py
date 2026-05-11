from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import streamlit as st

from core.app_backend import (
    approve_pending_review,
    initialize_dataset_session_from_file,
    reject_pending_review,
    run_pending_plan_until_pause,
    run_user_turn,
)
from ui.renderers import (
    render_json_expander,
    render_key_value_captions,
    rows_from_column_profile,
    render_analysis_run,
    render_data_version_timeline,
)
from ui.state import (
    add_message,
    has_dataset,
    sync_assistant_response_to_chat,
)
from ui.styles import panel_header, status_pill


def render_upload_panel() -> None:
    panel_header("Upload Dataset", "CSV, Excel, or Parquet.")

    uploaded = st.file_uploader(
        "CSV, XLSX, XLS, or Parquet",
        type=["csv", "xlsx", "xls", "parquet"],
    )

    if uploaded is None:
        st.caption("Upload a dataset to initialize the analysis session.")
        return

    if st.button("Load dataset", type="primary", width="stretch"):
        session = st.session_state["app_session"]
        suffix = Path(uploaded.name).suffix

        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.getbuffer())
            tmp_path = tmp.name

        result = initialize_dataset_session_from_file(
            tmp_path,
            workspace_dir=session.workspace_dir,
            dataset_name=Path(uploaded.name).stem,
        )

        st.session_state["graph_state"] = result["state"]
        st.session_state["snapshot"] = result["snapshot"]
        st.session_state["messages"] = []

        sync_assistant_response_to_chat(result["snapshot"])

        st.rerun()


def render_dataset_panel(snapshot: Dict[str, Any]) -> None:
    panel_header("Dataset", "Current active data version.")

    dataset = snapshot.get("dataset") or {}

    if not dataset.get("dataset_name"):
        st.info("No dataset loaded.")
        return

    st.markdown(f"**{dataset.get('dataset_name')}**")
    active_version_id = dataset.get("active_data_version_id")
    status_pill(f"Active: {active_version_id}", kind="ok")
    st.write("")

    summary = dataset.get("summary") or {}
    profile = dataset.get("profile") or {}
    columns = profile.get("columns") or {}

    metric_cols = st.columns(2)
    metric_cols[0].metric("Rows", summary.get("n_rows", "—"))
    metric_cols[1].metric("Columns", summary.get("n_cols", len(columns) or "—"))

    with st.expander("Column profile", expanded=True):
        if not columns:
            st.caption("No column profile available.")
            return

        st.dataframe(
            rows_from_column_profile(columns),
            width="stretch",
            hide_index=True,
        )


def render_data_versions(snapshot: Dict[str, Any]) -> None:
    dataset = snapshot.get("dataset") or {}
    versions = dataset.get("data_versions") or []
    active_version_id = dataset.get("active_data_version_id")

    panel_header("Data Versions", "Track mutations and lineage.")

    render_data_version_timeline(
        versions,
        active_version_id=active_version_id,
    )


def render_plan_panel(snapshot: Dict[str, Any]) -> None:
    panel_header("Plan", "Review planned analysis steps.")

    plan = snapshot.get("plan") or {}
    pending_plan = plan.get("pending_plan")

    if not pending_plan:
        st.info("No pending plan.")
        return

    render_key_value_captions([
        ("Plan ID", plan.get("plan_id")),
        ("Plan status", plan.get("plan_status")),
        ("Execution status", plan.get("plan_execution_status")),
    ])

    steps = pending_plan.get("steps") or []

    for step in steps:
        title = step.get("title") or step.get("tool_name") or step.get("step_id")
        status = step.get("status")
        ready = step.get("execution_ready")

        with st.expander(f"{title} · {status}", expanded=False):
            st.write(step.get("purpose") or step.get("rationale") or "")
            st.json({
                "step_id": step.get("step_id"),
                "tool_name": step.get("tool_name"),
                "status": status,
                "execution_ready": ready,
                "arguments": step.get("arguments"),
                "required_user_choices": step.get("required_user_choices"),
                "warnings": step.get("warnings"),
            })

    st.divider()

    if st.button("Run confirmed plan", type="primary", width="stretch"):
        session = st.session_state["app_session"]

        result = run_pending_plan_until_pause(
            st.session_state["graph_state"],
            config=session.graph_config,
        )

        st.session_state["graph_state"] = result["state"]
        st.session_state["snapshot"] = result["snapshot"]

        plan_run = result.get("plan_run") or {}
        add_message(
            "assistant",
            f"Plan run `{plan_run.get('status')}`: {plan_run.get('reason')}",
        )
        sync_assistant_response_to_chat(result["snapshot"])

        st.rerun()


def render_analysis_panel(snapshot: Dict[str, Any]) -> None:
    panel_header("Analysis Results", "Executed tools and generated outputs.")

    analysis = snapshot.get("analysis") or {}
    runs = analysis.get("analysis_runs") or []
    observations = analysis.get("observations") or []

    if not runs and not observations:
        st.info("No analysis results yet.")
        return

    for run in runs:
        render_analysis_run(run)

    if observations:
        render_json_expander("Observation history", observations, expanded=False)


def render_review_panel(snapshot: Dict[str, Any]) -> None:
    panel_header("Review", "Approve or reject high-risk actions.")

    review = snapshot.get("review") or {}

    if not review.get("human_review_required"):
        status_pill("No approval required", kind="neutral")
        return

    status_pill("Approval required", kind="danger")
    st.write("")

    st.warning("This action requires approval before execution.")

    pending_action = review.get("pending_action") or {}
    verification = review.get("current_verification") or {}

    tool_name = (
        pending_action.get("tool_name")
        or verification.get("details", {}).get("tool_name")
        or "unknown_tool"
    )

    st.markdown(f"**Pending tool:** `{tool_name}`")

    feedback = review.get("feedback") or verification.get("feedback")
    if feedback:
        st.caption(feedback)

    with st.expander("Pending action", expanded=True):
        st.json(pending_action)

    with st.expander("Verification details", expanded=False):
        st.json(verification)

    rejection_reason = st.text_area(
        "Rejection reason",
        placeholder="Optional: explain why this action should not run.",
        key="human_review_rejection_reason_input",
    )

    approve_col, reject_col = st.columns(2)

    with approve_col:
        if st.button(
            "Approve and run",
            type="primary",
            width="stretch",
            key="approve_pending_review",
        ):
            session = st.session_state["app_session"]

            result = approve_pending_review(
                st.session_state["graph_state"],
                config=session.graph_config,
            )

            st.session_state["graph_state"] = result["state"]
            st.session_state["snapshot"] = result["snapshot"]

            add_message("assistant", "Approved. Continuing execution.")
            sync_assistant_response_to_chat(result["snapshot"])

            st.rerun()

    with reject_col:
        if st.button(
            "Reject",
            type="secondary",
            width="stretch",
            key="reject_pending_review",
        ):
            session = st.session_state["app_session"]

            result = reject_pending_review(
                st.session_state["graph_state"],
                rejection_reason=rejection_reason.strip() or None,
                config=session.graph_config,
            )

            st.session_state["graph_state"] = result["state"]
            st.session_state["snapshot"] = result["snapshot"]

            add_message("assistant", "Rejected the pending action.")
            sync_assistant_response_to_chat(result["snapshot"])

            st.rerun()


def render_chat(snapshot: Dict[str, Any]) -> None:
    panel_header("Chat", "Ask questions or request analysis plans.")

    sync_assistant_response_to_chat(snapshot)

    for message in st.session_state["messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if not has_dataset(snapshot):
        st.chat_input("Upload a dataset first.", disabled=True)
        return

    user_message = st.chat_input("Ask about your data or request an analysis plan...")

    if user_message:
        add_message("user", user_message)

        session = st.session_state["app_session"]

        result = run_user_turn(
            st.session_state["graph_state"],
            user_message,
            config=session.graph_config,
        )

        st.session_state["graph_state"] = result["state"]
        st.session_state["snapshot"] = result["snapshot"]

        sync_assistant_response_to_chat(result["snapshot"])

        st.rerun()