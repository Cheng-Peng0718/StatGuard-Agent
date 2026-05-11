from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.controller.backend_turn import run_backend_turn
from core.ui_adapter.dataset_upload import prepare_uploaded_dataset_state
from core.ui_adapter.events import (
    apply_ui_event_to_state,
    make_approve_human_review_event,
    make_reject_human_review_event,
    make_run_plan_event,
    make_update_plan_step_choices_event,
    make_user_message_event,
)
from core.ui_adapter.snapshot import build_ui_snapshot

from ui.components.action_bar import render_action_bar
from ui.components.active_workspace import render_active_workspace
from ui.components.chat_panel import render_chat_panel
from ui.components.debug_panel import render_debug_panel
from ui.components.plan_timeline import render_plan_timeline
from ui.components.system_status import render_system_status
from ui.components.report_panel import render_report_panel


APP_TITLE = "Analysis Agent V3"


def load_css() -> None:
    css_path = PROJECT_ROOT / "ui" / "styles" / "app_v3.css"

    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )


def make_initial_backend_state() -> Dict[str, Any]:
    return {
        "user_request": "",
        "current_step": 0,
        "max_steps": 8,
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
    if "backend_state_v3" not in st.session_state:
        st.session_state.backend_state_v3 = make_initial_backend_state()

    if "ui_snapshot_v3" not in st.session_state:
        st.session_state.ui_snapshot_v3 = build_ui_snapshot(
            st.session_state.backend_state_v3
        )

    if "chat_history_v3" not in st.session_state:
        st.session_state.chat_history_v3 = []

    if "last_error_v3" not in st.session_state:
        st.session_state.last_error_v3 = None


def refresh_snapshot() -> Dict[str, Any]:
    snapshot = build_ui_snapshot(st.session_state.backend_state_v3)
    st.session_state.ui_snapshot_v3 = snapshot
    return snapshot


def submit_ui_event(event: Dict[str, Any]) -> None:
    try:
        updates = apply_ui_event_to_state(
            st.session_state.backend_state_v3,
            event,
        )

        st.session_state.backend_state_v3.update(updates)

        result = run_backend_turn(st.session_state.backend_state_v3)

        st.session_state.backend_state_v3 = result["state"]
        st.session_state.ui_snapshot_v3 = result["ui_snapshot"]
        st.session_state.last_error_v3 = None

    except Exception as exc:
        st.session_state.last_error_v3 = str(exc)
        refresh_snapshot()


def on_user_message(message: str) -> None:
    st.session_state.chat_history_v3.append(
        {
            "role": "user",
            "content": message,
        }
    )

    submit_ui_event(make_user_message_event(message))

    response = st.session_state.ui_snapshot_v3.get("assistant_response") or {}
    content = response.get("content")

    if content:
        st.session_state.chat_history_v3.append(
            {
                "role": "assistant",
                "content": content,
            }
        )


def on_dataset_upload(df: pd.DataFrame, filename: str) -> None:
    try:
        updates = prepare_uploaded_dataset_state(
            df=df,
            workspace_dir="workspaces/ui_app_v3",
            filename=filename,
        )

        st.session_state.backend_state_v3.update(updates)
        refresh_snapshot()

        response = st.session_state.ui_snapshot_v3.get("assistant_response") or {}
        content = response.get("content")

        if content:
            st.session_state.chat_history_v3.append(
                {
                    "role": "assistant",
                    "content": content,
                }
            )

        st.session_state.last_error_v3 = None

    except Exception as exc:
        st.session_state.last_error_v3 = str(exc)
        refresh_snapshot()


def on_save_choices(step_id: str, choices: Dict[str, Any]) -> None:
    submit_ui_event(
        make_update_plan_step_choices_event(
            step_id=step_id,
            choices=choices,
        )
    )

def on_run_plan() -> None:
    submit_ui_event(make_run_plan_event())


def on_approve() -> None:
    snapshot = st.session_state.ui_snapshot_v3
    human_review = snapshot.get("human_review") or {}
    action_hash = human_review.get("action_hash")

    submit_ui_event(
        make_approve_human_review_event(
            action_hash=action_hash,
        )
    )


def on_reject() -> None:
    snapshot = st.session_state.ui_snapshot_v3
    human_review = snapshot.get("human_review") or {}
    action_hash = human_review.get("action_hash")

    submit_ui_event(
        make_reject_human_review_event(
            action_hash=action_hash,
            reason="Rejected from App V3.",
        )
    )


def render_header() -> None:
    st.markdown("<div class='app-v3-header'>", unsafe_allow_html=True)
    st.title(APP_TITLE)
    st.caption("One-screen UI skeleton. Backend behavior remains adapter/controller-driven.")
    st.markdown("</div>", unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE,
        layout="wide",
    )

    load_css()
    init_session_state()

    snapshot = refresh_snapshot()

    render_header()

    if st.session_state.last_error_v3:
        st.error(st.session_state.last_error_v3)

    render_system_status(snapshot)

    workspace, right = st.columns([3.25, 1.05])

    with workspace:
        left, center = st.columns([1.15, 2.1])

        with left:
            render_chat_panel(
                snapshot=snapshot,
                chat_history=st.session_state.chat_history_v3,
                on_user_message=on_user_message,
            )

        with center:
            render_active_workspace(
                snapshot=snapshot,
                on_dataset_upload=on_dataset_upload,
                on_save_choices=on_save_choices,
            )

        render_action_bar(
            snapshot=snapshot,
            on_run_plan=on_run_plan,
            on_approve=on_approve,
            on_reject=on_reject,
        )

    with right:
        render_plan_timeline(snapshot)
        render_report_panel(st.session_state.backend_state_v3)
        render_debug_panel(snapshot)


if __name__ == "__main__":
    main()