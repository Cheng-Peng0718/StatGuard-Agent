from typing import Any, Dict, Tuple


from core.analysis_plugins.registry import register_plugin
from core.guardrails import evaluate_regression_guardrails

from core.analysis_plugins.base import (
    AnalysisPlugin,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_number,
    format_p_value,
    safe_join_list,
)

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


LINEAR_MODEL_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "nobs": "Observations used",
            "r_squared": "R-squared",
            "adj_r_squared": "Adjusted R-squared",
            "f_statistic": "F statistic",
            "f_p_value": "Model p-value",
        },
        formatters={
            "r_squared": lambda x: format_number(x, digits=4),
            "adj_r_squared": lambda x: format_number(x, digits=4),
            "f_statistic": lambda x: format_number(x, digits=4),
            "f_p_value": format_p_value,
        },
        order=[
            "nobs",
            "r_squared",
            "adj_r_squared",
            "f_statistic",
            "f_p_value",
        ],
    ),
    tables={
        "coef_table": TableDisplayConfig(
            column_labels={
                "term": "Term",
                "coef": "Estimate",
                "std_err": "Std. Error",
                "t": "t",
                "p_value": "p-value",
                "ci_lower": "95% CI lower",
                "ci_upper": "95% CI upper",
            },
            column_formatters={
                "coef": lambda x: format_number(x, digits=4),
                "std_err": lambda x: format_number(x, digits=4),
                "t": lambda x: format_number(x, digits=4),
                "p_value": format_p_value,
                "ci_lower": lambda x: format_number(x, digits=4),
                "ci_upper": lambda x: format_number(x, digits=4),
            },
            column_order=[
                "term",
                "coef",
                "std_err",
                "t",
                "p_value",
                "ci_lower",
                "ci_upper",
            ],
            value_mappers={
                "term": {
                    "const": "Intercept",
                    "intercept": "Intercept",
                    "Intercept": "Intercept",
                }
            },
        )
    },
)

PLUGIN = register_plugin(AnalysisPlugin(
    tool_name="run_multiple_regression",
    display_name="Linear Model",
    extractor=extract_linear_model,
    guardrail_evaluators=[
        evaluate_regression_guardrails,
    ],
    display_config=LINEAR_MODEL_DISPLAY,
))