from __future__ import annotations

import streamlit as st

from ui.panels import (
    render_analysis_panel,
    render_chat,
    render_data_versions,
    render_dataset_panel,
    render_plan_panel,
    render_review_panel,
    render_upload_panel,
)
from ui.state import current_snapshot, ensure_session_state


st.set_page_config(
    page_title="Analysis Agent",
    page_icon="📊",
    layout="wide",
)


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