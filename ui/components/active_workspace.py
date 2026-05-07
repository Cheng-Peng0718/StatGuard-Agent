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


def _choice_label(choice_name: str) -> str:
    labels = {
        "target_col": "outcome / response column",
        "feature_cols": "predictor columns",
        "group_col": "group column",
        "columns": "columns",
        "analysis variables": "analysis variables",
        "action_type": "cleaning action",
        "strategy": "cleaning strategy",
    }

    return labels.get(choice_name, choice_name)


def _available_columns_for_choice(
    snapshot: Dict[str, Any],
    choice_name: str,
) -> list[str]:
    data = snapshot.get("data") or {}
    summary = data.get("dataset_summary") or {}
    uploaded_info = data.get("uploaded_dataset_info") or {}

    all_columns = uploaded_info.get("columns") or []
    numeric_columns = summary.get("numeric_columns") or []
    categorical_columns = summary.get("categorical_columns") or []

    if choice_name in {"target_col", "feature_cols", "analysis variables"}:
        return numeric_columns or all_columns

    if choice_name == "group_col":
        return categorical_columns or all_columns

    if choice_name == "columns":
        return all_columns

    return all_columns


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


def render_choice_form(
    *,
    snapshot: Dict[str, Any],
    step: Dict[str, Any],
    on_save_choices: Callable[[str, Dict[str, Any]], None],
) -> None:
    step_id = step.get("step_id")
    required_choices = step.get("required_user_choices") or []

    if not step_id:
        st.error("Cannot save choices because this step has no step_id.")
        return

    if not required_choices:
        st.info("This step does not require user choices.")
        return

    st.warning("This step needs user choices before it can run.")

    st.write(f"**Step:** `{step.get('title') or step.get('tool_name')}`")
    st.write(f"**Tool:** `{step.get('tool_name')}`")

    with st.form(f"app_v3_choice_form_{step_id}", clear_on_submit=False):
        choices: Dict[str, Any] = {}

        for choice_name in required_choices:
            widget_key = f"app_v3_{step_id}_{choice_name}"

            if choice_name == "action_type":
                choices[choice_name] = st.selectbox(
                    "Choose cleaning action",
                    options=["", "drop", "impute"],
                    key=widget_key,
                )
                continue

            if choice_name == "strategy":
                choices[choice_name] = st.selectbox(
                    "Choose cleaning strategy",
                    options=["", "rows", "columns", "mean", "median", "mode"],
                    key=widget_key,
                )
                continue

            options = _available_columns_for_choice(
                snapshot=snapshot,
                choice_name=choice_name,
            )

            if not options:
                st.info(f"No available columns for `{choice_name}`.")
                choices[choice_name] = [] if choice_name.endswith("s") else ""
                continue

            label = f"Choose {_choice_label(choice_name)}"

            if choice_name in {"feature_cols", "columns", "analysis variables"}:
                choices[choice_name] = st.multiselect(
                    label,
                    options=options,
                    key=widget_key,
                )
            else:
                choices[choice_name] = st.selectbox(
                    label,
                    options=[""] + options,
                    key=widget_key,
                )

        submitted = st.form_submit_button(
            "Save choices for this step",
            use_container_width=True,
        )

    if submitted:
        on_save_choices(step_id, choices)
        st.rerun()

    with st.expander("Step details", expanded=False):
        st.json(step)


def render_human_review_focus(snapshot: Dict[str, Any]) -> None:
    human_review = snapshot.get("human_review") or {}
    action = human_review.get("action") or {}

    st.error("Human review required before execution.")

    st.write(f"**Tool:** `{action.get('tool_name')}`")

    feedback = human_review.get("feedback")
    if feedback:
        st.warning(feedback)

    st.write("**Arguments:**")
    st.json(action.get("arguments") or {})

    st.info(
        "Use the bottom action bar to approve or reject this action. "
        "Approval is required before the backend can execute it."
    )


def render_latest_result(snapshot: Dict[str, Any], *, failed: bool = False) -> None:
    run = _latest_run(snapshot) or {}

    if failed:
        st.error("Latest analysis run failed.")
    else:
        st.success("Latest analysis run completed.")

    st.write(f"**Tool:** `{run.get('tool_name')}`")
    st.write(f"**Status:** `{run.get('status')}`")
    st.write(f"**Success:** `{run.get('success')}`")

    error_code = run.get("error_code")
    if error_code:
        st.error(f"Error code: `{error_code}`")

    summary = run.get("summary") or run.get("message")
    if summary:
        st.write(summary)
    else:
        st.info("No summary available.")

    with st.expander("Run details", expanded=False):
        st.json(run)


def render_active_workspace(
    *,
    snapshot: Dict[str, Any],
    on_dataset_upload: Callable[[pd.DataFrame, str], None],
    on_save_choices: Callable[[str, Dict[str, Any]], None],
) -> None:
    focus = determine_active_focus(snapshot)

    st.subheader("Active Workspace")
    st.caption(f"Focus: `{focus}`")

    with st.container(height=560):
        if focus == "human_review":
            render_human_review_focus(snapshot)
            return

        if focus == "user_choices":
            step = _first_needs_choice_step(snapshot) or {}
            render_choice_form(
                snapshot=snapshot,
                step=step,
                on_save_choices=on_save_choices,
            )
            return

        if focus == "failed_result":
            render_latest_result(snapshot, failed=True)
            return

        if focus == "latest_result":
            render_latest_result(snapshot, failed=False)
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