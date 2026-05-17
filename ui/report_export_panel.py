import streamlit as st
from core.report_builder import (
    build_markdown_report,
    build_html_report_from_state,
)


def render_report_export_panel():
    analysis_runs = st.session_state.get("analysis_runs", [])

    if not analysis_runs:
        return

    st.divider()
    st.subheader("Export Report")

    report_user_request = ""
    if st.session_state.get("messages"):
        user_messages = [
            m.get("content", "")
            for m in st.session_state.messages
            if m.get("role") == "user"
        ]
        report_user_request = user_messages[-1] if user_messages else ""

    markdown_report = build_markdown_report(
        user_request=report_user_request,
        active_data_version_id=st.session_state.get("active_data_version_id"),
        data_versions=st.session_state.get("data_versions", []),
        data_audit_log=st.session_state.get("data_audit_log", []),
        analysis_runs=st.session_state.get("analysis_runs", []),
        title="Data Analysis Report",
    )

    html_report = build_html_report_from_state(
        user_request=report_user_request,
        active_data_version_id=st.session_state.get("active_data_version_id"),
        data_versions=st.session_state.get("data_versions", []),
        data_audit_log=st.session_state.get("data_audit_log", []),
        analysis_runs=st.session_state.get("analysis_runs", []),
        title="Data Analysis Report",
    )

    st.download_button(
        label="Download Markdown Report",
        data=markdown_report,
        file_name="analysis_report.md",
        mime="text/markdown",
        key="download_markdown_report_main",
    )

    st.download_button(
        label="Download HTML Report",
        data=html_report,
        file_name="analysis_report.html",
        mime="text/html",
        key="download_html_report_main",
    )