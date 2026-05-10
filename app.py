from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, List

import streamlit as st

from core.app_backend import (
    create_app_session,
    initialize_dataset_session_from_file,
    run_pending_plan_until_pause,
    run_user_turn,
)


st.set_page_config(
    page_title="Analysis Agent",
    page_icon="📊",
    layout="wide",
)


def ensure_session_state() -> None:
    if "app_session" not in st.session_state:
        st.session_state["app_session"] = create_app_session()

    if "graph_state" not in st.session_state:
        st.session_state["graph_state"] = None

    if "snapshot" not in st.session_state:
        st.session_state["snapshot"] = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def current_snapshot() -> Dict[str, Any]:
    return st.session_state.get("snapshot") or {}


def has_dataset(snapshot: Dict[str, Any]) -> bool:
    dataset = snapshot.get("dataset") or {}
    return bool(dataset.get("dataset_name"))


def add_message(role: str, content: str) -> None:
    if not content:
        return

    messages: List[Dict[str, str]] = st.session_state["messages"]

    if messages and messages[-1].get("role") == role and messages[-1].get("content") == content:
        return

    messages.append({
        "role": role,
        "content": content,
    })


def sync_assistant_response_to_chat(snapshot: Dict[str, Any]) -> None:
    response = snapshot.get("assistant_response") or {}
    content = response.get("content")

    if content:
        add_message("assistant", content)


def render_upload_panel() -> None:
    st.subheader("Upload Dataset")

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
    st.subheader("Dataset")

    dataset = snapshot.get("dataset") or {}

    if not dataset.get("dataset_name"):
        st.info("No dataset loaded.")
        return

    st.markdown(f"**{dataset.get('dataset_name')}**")
    st.caption(f"Active data version: `{dataset.get('active_data_version_id')}`")

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

        rows = []
        for name, meta in columns.items():
            rows.append({
                "column": name,
                "semantic_type": meta.get("semantic_type", "unknown"),
                "dtype": meta.get("dtype", "unknown"),
                "missing_rate": meta.get("missing_rate", 0.0),
                "n_unique": meta.get("n_unique"),
            })

        st.dataframe(rows, width="stretch", hide_index=True)


def render_data_versions(snapshot: Dict[str, Any]) -> None:
    dataset = snapshot.get("dataset") or {}
    versions = dataset.get("data_versions") or []

    st.subheader("Data Versions")

    if not versions:
        st.info("No data versions yet.")
        return

    for version in versions:
        label = version.get("version_id") or "unknown_version"
        with st.expander(label, expanded=False):
            st.json(version)


def render_plan_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Plan")

    plan = snapshot.get("plan") or {}
    pending_plan = plan.get("pending_plan")

    if not pending_plan:
        st.info("No pending plan.")
        return

    st.caption(f"Plan ID: `{plan.get('plan_id')}`")
    st.caption(f"Plan status: `{plan.get('plan_status')}`")
    st.caption(f"Execution status: `{plan.get('plan_execution_status')}`")

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
    st.subheader("Analysis Results")

    analysis = snapshot.get("analysis") or {}
    runs = analysis.get("analysis_runs") or []
    observations = analysis.get("observations") or []

    if not runs and not observations:
        st.info("No analysis results yet.")
        return

    for run in runs:
        title = run.get("tool_name") or run.get("run_id") or "Analysis run"
        status = run.get("status")

        with st.expander(f"{title} · {status}", expanded=True):
            if run.get("summary"):
                st.write(run["summary"])

            report_blocks = run.get("report_blocks") or []
            if report_blocks:
                st.write("Report blocks")
                st.json(report_blocks)

            if run.get("metrics"):
                st.write("Metrics")
                st.json(run["metrics"])

            if run.get("tables"):
                st.write("Tables")
                st.json(run["tables"])

            if run.get("artifacts"):
                st.write("Artifacts")
                st.json(run["artifacts"])

    if observations:
        with st.expander("Observation history", expanded=False):
            st.json(observations)


def render_review_panel(snapshot: Dict[str, Any]) -> None:
    st.subheader("Review")

    review = snapshot.get("review") or {}

    if not review.get("human_review_required"):
        st.info("No human review required.")
        return

    st.warning("This action requires approval before execution.")
    st.json(review.get("current_verification") or {})

    st.caption("Approve / reject wiring will be added in the next UI phase.")


def render_chat(snapshot: Dict[str, Any]) -> None:
    st.subheader("Chat")

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


def render_layout() -> None:
    snapshot = current_snapshot()

    left, center, right = st.columns([0.95, 1.45, 1.1], gap="large")

    with left:
        render_upload_panel()
        st.divider()
        render_dataset_panel(snapshot)
        st.divider()
        render_data_versions(snapshot)

    with center:
        render_chat(snapshot)
        st.divider()
        render_plan_panel(snapshot)

    with right:
        render_analysis_panel(snapshot)
        st.divider()
        render_review_panel(snapshot)


def main() -> None:
    ensure_session_state()

    st.title("📊 Analysis Agent")
    st.caption("Plan-first, plugin-driven data analysis assistant.")

    render_layout()


if __name__ == "__main__":
    main()