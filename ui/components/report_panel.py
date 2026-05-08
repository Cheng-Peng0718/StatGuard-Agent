from __future__ import annotations

from typing import Dict

import streamlit as st

from core.ui_adapter.report_export import build_report_package_from_state


def render_report_panel(state: Dict) -> None:
    st.subheader("Report")

    analysis_runs = state.get("analysis_runs") or []

    if not analysis_runs:
        st.info("Run at least one analysis step before exporting a report.")
        return

    package = build_report_package_from_state(state)

    metadata = package.get("metadata") or {}

    st.caption(
        f"Runs: `{metadata.get('n_analysis_runs')}` · "
        f"Active data: `{metadata.get('active_data_version_id')}`"
    )

    col1, col2 = st.columns(2)

    with col1:
        st.download_button(
            label="Download Markdown",
            data=package["markdown"],
            file_name="analysis_report.md",
            mime="text/markdown",
            use_container_width=True,
            key="app_v3_download_markdown_report",
        )

    with col2:
        st.download_button(
            label="Download HTML",
            data=package["html"],
            file_name="analysis_report.html",
            mime="text/html",
            use_container_width=True,
            key="app_v3_download_html_report",
        )

    with st.expander("Plain-text summary", expanded=False):
        st.text(package["plain_text_summary"])