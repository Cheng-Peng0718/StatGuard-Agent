from __future__ import annotations

from typing import Callable, Dict

import streamlit as st


def render_action_bar(
    *,
    snapshot: Dict,
    on_run_plan: Callable[[], None],
    on_approve: Callable[[], None],
    on_reject: Callable[[], None],
) -> None:
    st.markdown("<div class='app-v3-action-bar'>", unsafe_allow_html=True)

    human_review = snapshot.get("human_review") or {}

    col1, col2, col3 = st.columns([2, 1, 1])

    with col1:
        if st.button("Run next step", use_container_width=True):
            on_run_plan()
            st.rerun()

    with col2:
        disabled = not human_review.get("required")
        if st.button("Approve", type="primary", disabled=disabled, use_container_width=True):
            on_approve()
            st.rerun()

    with col3:
        disabled = not human_review.get("required")
        if st.button("Reject", disabled=disabled, use_container_width=True):
            on_reject()
            st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)