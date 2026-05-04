from typing import Any, Dict, Tuple

from core.analysis_plugins.base import AnalysisPlugin, compact_dict
from core.analysis_plugins.registry import register_plugin
from core.guardrails import evaluate_residual_guardrails


def extract_residual_histogram(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    title = "Residual Histogram"

    diagnostic_flags = payload.get("diagnostic_flags", [])

    metrics = compact_dict({
        "n_residuals": payload.get("n_residuals"),
        "residual_mean": payload.get("residual_mean"),
        "residual_std": payload.get("residual_std"),
        "residual_skewness": payload.get("residual_skewness"),
        "residual_kurtosis_fisher": payload.get("residual_kurtosis_fisher"),
        "outliers_abs_2sd": payload.get("outliers_abs_2sd"),
        "outliers_abs_3sd": payload.get("outliers_abs_3sd"),
        "diagnostic_flags": diagnostic_flags if diagnostic_flags else None,
    })

    tables: Dict[str, Any] = {}

    summary = "Generated a residual histogram."
    if diagnostic_flags:
        summary += f" Diagnostic flags: {', '.join(str(x) for x in diagnostic_flags)}."

    return title, summary, metrics, tables


PLUGIN = register_plugin(AnalysisPlugin(
    tool_name="generate_residual_histogram",
    display_name="Residual Histogram",
    extractor=extract_residual_histogram,
    guardrail_evaluators=[
        evaluate_residual_guardrails,
    ],
))