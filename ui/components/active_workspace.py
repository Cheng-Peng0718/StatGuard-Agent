from __future__ import annotations

from typing import Any, Callable, Dict

import pandas as pd
import streamlit as st


def _latest_run(snapshot: Dict[str, Any]) -> Dict[str, Any] | None:
    analysis = snapshot.get("analysis") or {}
    runs = analysis.get("analysis_runs") or []

    if not runs:
        return None

    return runs[-1]


def _first_needs_choice_step(snapshot: Dict[str, Any]) -> Dict[str, Any] | None:
    plan = snapshot.get("plan") or {}
    pending_plan = plan.get("pending_plan") or {}

    for step in pending_plan.get("steps") or []:
        if step.get("status") == "needs_user_choice":
            return step

    return None


def determine_active_focus(snapshot: Dict[str, Any]) -> str:
    human_review = snapshot.get("human_review") or {}

    if human_review.get("required"):
        return "human_review"

    if _first_needs_choice_step(snapshot):
        return "user_choices"

    latest = _latest_run(snapshot)

    if latest and latest.get("success") is False:
        return "failed_result"

    if latest:
        return "latest_result"

    if (snapshot.get("plan") or {}).get("pending_plan"):
        return "pending_plan"

    return "dataset_upload"


def render_active_workspace(
    *,
    snapshot: Dict[str, Any],
    on_dataset_upload: Callable[[pd.DataFrame, str], None],
) -> None:
    focus = determine_active_focus(snapshot)

    st.subheader("Active Workspace")
    st.caption(f"Focus: `{focus}`")

    if focus == "human_review":
        human_review = snapshot.get("human_review") or {}
        action = human_review.get("action") or {}

        st.warning("Action requires human review.")
        st.write(f"Tool: `{action.get('tool_name')}`")
        st.json(action.get("arguments") or {})
        st.info("Use the bottom action bar to approve or reject.")
        return

    if focus == "user_choices":
        step = _first_needs_choice_step(snapshot) or {}
        st.warning("This step needs user choices.")
        st.write(f"Step: `{step.get('title') or step.get('tool_name')}`")
        st.write("Required choices:")
        st.json(step.get("required_user_choices") or [])
        st.info("Choice controls will be wired in the next App V3 step.")
        return

    if focus in {"latest_result", "failed_result"}:
        run = _latest_run(snapshot) or {}

        if focus == "failed_result":
            st.error("Latest analysis run failed.")
        else:
            st.success("Latest analysis run completed.")

        st.write(f"Tool: `{run.get('tool_name')}`")
        st.write(run.get("summary") or run.get("message") or "No summary available.")

        with st.expander("Run details", expanded=False):
            st.json(run)

        return

    if focus == "pending_plan":
        st.info("A plan is ready. Use the bottom action bar to run the next step.")
        return

    st.info("Upload a CSV dataset to begin.")

    uploaded_file = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
        key="app_v3_dataset_upload",
    )

    if uploaded_file is not None:
        if st.button("Load dataset", use_container_width=True):
            df = pd.read_csv(uploaded_file)
            on_dataset_upload(df, uploaded_file.name)
            st.rerun()