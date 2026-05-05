from typing import Any, Dict, Tuple

from core.analysis_plugins.base import AnalysisPlugin, compact_dict, safe_join_list
from core.analysis_plugins.registry import register_plugin
from core.guardrails import evaluate_regression_guardrails


def extract_linear_model(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target = arguments.get("target_col")
    features = arguments.get("feature_cols", [])
    feature_text = safe_join_list(features)

    if target and feature_text:
        title = f"Linear Model: {target} ~ {feature_text}"
    else:
        title = "Linear Model"

    # User-facing metrics only.
    metrics = compact_dict({
        "nobs": payload.get("nobs"),
        "r_squared": payload.get("r_squared"),
        "adj_r_squared": payload.get("adj_r_squared"),
        "f_statistic": payload.get("f_statistic"),
        "f_p_value": payload.get("f_p_value"),
    })

    # Internal/system metrics. These should not show in the main report by default.
    metadata = compact_dict({
        "aic": payload.get("aic"),
        "bic": payload.get("bic"),
        "df_model": payload.get("df_model"),
        "df_resid": payload.get("df_resid"),
        "n_eff": payload.get("n_eff"),
        "p_eff": payload.get("p_eff"),
    })

    tables: Dict[str, Any] = {}

    coef_table = payload.get("coef_table", [])
    if coef_table:
        tables["coef_table"] = coef_table

    summary = "Fitted a linear model."
    if target:
        summary += f" Outcome: `{target}`."
    if feature_text:
        summary += f" Predictors: `{feature_text}`."
    if payload.get("nobs") is not None:
        summary += f" n={payload.get('nobs')}."
    if payload.get("r_squared") is not None:
        summary += f" R²={payload.get('r_squared')}."

    return title, summary, metrics, tables, metadata


PLUGIN = register_plugin(AnalysisPlugin(
    tool_name="run_multiple_regression",
    display_name="Linear Model",
    extractor=extract_linear_model,
    guardrail_evaluators=[
        evaluate_regression_guardrails,
    ],
))