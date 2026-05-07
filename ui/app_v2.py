from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

# Ensure project root is importable when Streamlit runs this file from ui/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import streamlit as st

from core.controller.backend_turn import run_backend_turn
from core.ui_adapter.dataset_upload import prepare_uploaded_dataset_state
from core.ui_adapter.events import (
    apply_ui_event_to_state,
    make_approve_human_review_event,
    make_reject_human_review_event,
    make_run_plan_event,
    make_user_message_event,
    make_update_plan_step_choices_event,
)
from core.ui_adapter.snapshot import build_ui_snapshot

APP_TITLE = "Analysis Agent V2"

def render_dataset_upload_panel() -> None:
    st.subheader("Dataset Upload")

    uploaded_file = st.file_uploader(
        "Upload CSV dataset",
        type=["csv"],
    )

    if uploaded_file is None:
        info = st.session_state.get("uploaded_dataset_info")

        if info:
            st.success(
                f"Loaded `{info.get('filename')}` "
                f"({info.get('n_rows')} rows, {info.get('n_cols')} columns)."
            )
        else:
            st.info("No dataset uploaded yet.")

        return

    if st.button("Load dataset", use_container_width=True):
        try:
            df = pd.read_csv(uploaded_file)

            workspace_dir = "workspaces/ui_app_v2"

            updates = prepare_uploaded_dataset_state(
                df=df,
                workspace_dir=workspace_dir,
                filename=uploaded_file.name,
            )

            st.session_state.backend_state.update(updates)
            st.session_state.uploaded_dataset_info = updates.get(
                "uploaded_dataset_info"
            )

            refresh_snapshot()
            st.rerun()

        except Exception as exc:
            st.session_state.last_error = str(exc)
            refresh_snapshot()

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

    if choice_name in {"target_col", "feature_cols", "columns", "analysis variables"}:
        return numeric_columns or all_columns

    if choice_name in {"group_col"}:
        return categorical_columns or all_columns

    return all_columns

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

    if choice_name in {"target_col", "feature_cols", "columns", "analysis variables"}:
        return numeric_columns or all_columns

    if choice_name in {"group_col"}:
        return categorical_columns or all_columns

    return all_columns


def render_plan_step_choice_controls(
    *,
    step: Dict[str, Any],
    snapshot: Dict[str, Any],
) -> None:
    required_choices = step.get("required_user_choices") or []

    if not required_choices:
        return

    st.warning(
        "Needs user choices: "
        + ", ".join(required_choices)
    )

    step_id = step.get("step_id")
    choices: Dict[str, Any] = {}

    for choice_name in required_choices:

        widget_key = f"{step_id}_{choice_name}"

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

        options = _available_columns_for_choice(snapshot, choice_name)

        if not options:
            st.info(f"No available columns for `{choice_name}`.")
            continue

        widget_key = f"{step_id}_{choice_name}"

        if choice_name in {"feature_cols", "columns", "analysis variables"}:
            choices[choice_name] = st.multiselect(
                f"Choose {_choice_label(choice_name)}",
                options=options,
                key=widget_key,
            )
        else:
            choices[choice_name] = st.selectbox(
                f"Choose {_choice_label(choice_name)}",
                options=[""] + options,
                key=widget_key,
            )

    if st.button(
        "Save choices for this step",
        key=f"save_choices_{step_id}",
        use_container_width=True,
    ):
        submit_ui_event(
            make_update_plan_step_choices_event(
                step_id=step_id,
                choices=choices,
            )
        )
        st.rerun()

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

    execution_status = plan_section.get("plan_execution_status")

    if execution_status == "blocked_no_ready_steps":
        st.warning(
            "No executable plan step is currently ready. "
            "Some remaining steps need user-selected variables before they can run."
        )

    if execution_status == "blocked_pending_data_cleaning":
        st.warning(
            "A modeling step is waiting for data cleaning to complete "
            "because the current dataset has missing values."
        )

    if not pending_plan:
        st.info("No pending plan.")
        return

    execution_status = plan_section.get("plan_execution_status")

    if execution_status == "blocked_no_ready_steps":
        st.warning(
            "No executable plan step is currently ready. "
            "Some remaining steps need user-selected variables before they can run."
        )

    steps = pending_plan.get("steps") or []

    for step in steps:
        with st.container(border=True):
            st.write(
                f"**{step.get('step_id')} — "
                f"{step.get('title', step.get('tool_name'))}**"
            )
            st.write(f"Tool: `{step.get('tool_name')}`")
            st.write(f"Status: `{step.get('status')}`")
            st.write(f"Execution: `{step.get('execution_status')}`")

            st.write(f"Execution ready: `{step.get('execution_ready')}`")

            readiness = step.get("readiness") or step.get("metadata", {}).get("readiness") or {}
            if readiness:
                reason = readiness.get("reason")
                if reason:
                    st.caption(f"Readiness reason: {reason}")

            reason = (
                    step.get("reason")
                    or step.get("purpose")
                    or step.get("rationale")
            )

            if reason:
                st.caption(reason)

            # IMPORTANT:
            # This must NOT be inside `if reason:`.
            # Some plan steps have purpose/rationale but no reason field.
            render_plan_step_choice_controls(
                step=step,
                snapshot=snapshot,
            )

            candidate_variables = step.get("candidate_variables") or {}
            if candidate_variables:
                with st.expander("Candidate variables", expanded=False):
                    st.json(candidate_variables)

            arguments = step.get("arguments") or {}
            if arguments:
                with st.expander("Arguments", expanded=False):
                    st.json(arguments)




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
        render_dataset_upload_panel()
        render_data_versions_panel(snapshot)
        render_repair_panel(snapshot)
        render_audit_panel(snapshot)
        render_snapshot_debug(snapshot)


if __name__ == "__main__":
    main()