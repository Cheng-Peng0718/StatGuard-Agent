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
from ui.styles import inject_app_styles

st.set_page_config(
    page_title="Analysis Agent",
    page_icon="📊",
    layout="wide",
)


def render_layout() -> None:
    snapshot = current_snapshot()

    left, center, right = st.columns([0.9, 1.55, 1.15], gap="large")

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
    inject_app_styles()

    st.markdown(
        """
        <div class="app-header">
            <div class="app-title">Analysis Agent</div>
            <div class="app-subtitle">
                Plan-first, plugin-driven data analysis assistant with review gates and data versioning.
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    render_layout()

if __name__ == "__main__":
    main()