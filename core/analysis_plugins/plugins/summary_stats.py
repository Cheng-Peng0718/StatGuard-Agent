from typing import Any, Dict, Tuple

from core.analysis_plugins.base import AnalysisPlugin
from core.analysis_plugins.registry import register_plugin


def extract_summary_stats(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    title = "Summary Statistics"
    summary = "Computed descriptive summary statistics for the active dataset."

    metrics: Dict[str, Any] = {}
    tables: Dict[str, Any] = {}

    numeric_summary = payload.get("numeric_summary", {})
    categorical_summary = payload.get("categorical_summary", {})

    if numeric_summary:
        tables["numeric_summary"] = numeric_summary
    if categorical_summary:
        tables["categorical_summary"] = categorical_summary

    metadata: Dict[str, Any] = {}
    return title, summary, metrics, tables, metadata


PLUGIN = register_plugin(AnalysisPlugin(
    tool_name="get_summary_stats",
    display_name="Summary Statistics",
    extractor=extract_summary_stats,
    guardrail_evaluators=[],
))