from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.report_builder import (
    build_html_report_from_state,
    build_markdown_report,
)


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []

    if isinstance(value, list):
        return value

    return [value]


def _latest_user_request(state: Dict[str, Any]) -> str:
    user_request = state.get("user_request")

    if isinstance(user_request, str) and user_request.strip():
        return user_request.strip()

    latest_event = state.get("latest_ui_event") or {}
    payload = latest_event.get("payload") or {}

    text = (
        payload.get("text")
        or payload.get("message")
        or payload.get("user_request")
    )

    if isinstance(text, str) and text.strip():
        return text.strip()

    return ""


def build_report_package_from_state(
    state: Dict[str, Any],
    *,
    title: str = "Data Analysis Report",
    user_request: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Build downloadable report artifacts from backend state.

    This adapter is UI-safe:
    - it does not execute tools
    - it does not mutate state
    - it does not call graph nodes
    - it only adapts state into report_builder inputs
    """
    if not isinstance(state, dict):
        raise TypeError("build_report_package_from_state requires a state dictionary.")

    analysis_runs = _as_list(state.get("analysis_runs"))
    data_versions = _as_list(state.get("data_versions"))
    data_audit_log = _as_list(state.get("data_audit_log"))
    active_data_version_id = state.get("active_data_version_id")

    resolved_user_request = (
        user_request
        if user_request is not None
        else _latest_user_request(state)
    )

    markdown_report = build_markdown_report(
        user_request=resolved_user_request,
        active_data_version_id=active_data_version_id,
        data_versions=data_versions,
        data_audit_log=data_audit_log,
        analysis_runs=analysis_runs,
        title=title,
    )

    html_report = build_html_report_from_state(
        user_request=resolved_user_request,
        active_data_version_id=active_data_version_id,
        data_versions=data_versions,
        data_audit_log=data_audit_log,
        analysis_runs=analysis_runs,
        title=title,
    )

    plain_text_summary = "\n".join(
        [
            title,
            "",
            f"Analysis runs: {len(analysis_runs)}",
            f"Data versions: {len(data_versions)}",
            f"Audit events: {len(data_audit_log)}",
            f"Active data version: {active_data_version_id or 'none'}",
        ]
    )

    return {
        "title": title,
        "markdown": markdown_report,
        "html": html_report,
        "plain_text_summary": plain_text_summary,
        "metadata": {
            "n_analysis_runs": len(analysis_runs),
            "n_data_versions": len(data_versions),
            "n_data_audit_events": len(data_audit_log),
            "active_data_version_id": active_data_version_id,
            "has_analysis_runs": len(analysis_runs) > 0,
        },
    }