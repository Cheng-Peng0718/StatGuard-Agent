from typing import Any, Dict, Tuple
from core.analysis_plugins.registry import register_plugin
from core.guardrails import evaluate_residual_guardrails

from core.analysis_plugins.base import (
    AnalysisPlugin,
    DisplayConfig,
    MetricDisplayConfig,
    compact_dict,
    format_list_semicolon,
    format_number,
)

def format_diagnostic_flags(value):
    mapping = {
        "left_skew_detected": "left skew detected",
        "right_skew_detected": "right skew detected",
        "heavy_tails_possible": "heavy tails possible",
        "possible_extreme_residual_outliers_abs_3sd": "possible extreme residual outliers beyond 3 SD",
        "possible_residual_outliers_abs_2sd": "possible residual outliers beyond 2 SD",
        "non_normal_residual_pattern_possible": "possible non-normal residual pattern",
    }

    if isinstance(value, list):
        return "; ".join(mapping.get(str(x), str(x).replace("_", " ")) for x in value)

    return mapping.get(str(value), str(value).replace("_", " "))

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
        summary += f" Diagnostic flags: {format_diagnostic_flags(diagnostic_flags)}."

    metadata: Dict[str, Any] = {}
    return title, summary, metrics, tables, metadata

RESIDUAL_HISTOGRAM_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "n_residuals": "Residual count",
            "residual_mean": "Residual mean",
            "residual_std": "Residual SD",
            "residual_skewness": "Residual skewness",
            "residual_kurtosis_fisher": "Residual kurtosis",
            "outliers_abs_2sd": "Residuals beyond 2 SD",
            "outliers_abs_3sd": "Residuals beyond 3 SD",
            "diagnostic_flags": "Diagnostic flags",
        },
        formatters={
            "residual_mean": lambda x: format_number(x, digits=4),
            "residual_std": lambda x: format_number(x, digits=4),
            "residual_skewness": lambda x: format_number(x, digits=4),
            "residual_kurtosis_fisher": lambda x: format_number(x, digits=4),
            "diagnostic_flags": format_diagnostic_flags,
        },
        order=[
            "n_residuals",
            "residual_mean",
            "residual_std",
            "residual_skewness",
            "residual_kurtosis_fisher",
            "outliers_abs_2sd",
            "outliers_abs_3sd",
            "diagnostic_flags",
        ],
    )
)

PLUGIN = register_plugin(AnalysisPlugin(
    tool_name="generate_residual_histogram",
    display_name="Residual Histogram",
    extractor=extract_residual_histogram,
    guardrail_evaluators=[
        evaluate_residual_guardrails,
    ],
    display_config=RESIDUAL_HISTOGRAM_DISPLAY,
))