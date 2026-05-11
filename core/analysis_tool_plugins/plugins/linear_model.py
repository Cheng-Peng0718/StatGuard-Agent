from typing import Any, Dict, Tuple
import math
import warnings

import statsmodels.api as sm

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_number,
    format_p_value,
    safe_join_list,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.regression_utils import prepare_regression_data
from core.guardrails import evaluate_regression_guardrails


def _ok(message: str, details: Dict[str, Any], artifacts=None):
    return {
        "status": "ok",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": artifacts or [],
    }


def _warning(message: str, details: Dict[str, Any], artifacts=None):
    return {
        "status": "warning",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": artifacts or [],
    }


def _failed(error_code: str, message: str, exc: Exception):
    return {
        "status": "failed",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        "artifacts": [],
    }


def _round_or_none(x: Any, digits: int = 6):
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return round(v, digits)
    except Exception:
        return None


def _get_arg(context, name: str, default: Any = None) -> Any:
    try:
        return context.get_arg(name, default)
    except TypeError:
        try:
            value = context.get_arg(name)
            return default if value is None else value
        except Exception:
            return default
    except Exception:
        return default


def execute_linear_model(context) -> Dict[str, Any]:
    """
    Fit an OLS multiple linear regression model.

    Args:
        target_col: numeric outcome column
        feature_cols: list of predictor columns
        max_missing_rate: optional, default 0.40
        max_categorical_levels: optional, default 10
        numeric_parse_threshold: optional, default 0.85
        min_n_per_parameter: optional, default 3
    """
    try:
        df = context.load_df()

        target_col = _get_arg(context, "target_col")
        feature_cols = _get_arg(context, "feature_cols", [])

        prep = prepare_regression_data(
            df,
            target_col,
            feature_cols,
            max_missing_rate=float(_get_arg(context, "max_missing_rate", 0.40)),
            max_categorical_levels=int(_get_arg(context, "max_categorical_levels", 10)),
            numeric_parse_threshold=float(_get_arg(context, "numeric_parse_threshold", 0.85)),
            min_n_per_parameter=int(_get_arg(context, "min_n_per_parameter", 3)),
        )

        if prep.get("status") != "ok":
            return prep

        y = prep["y"]
        X = prep["X"]

        X_const = sm.add_constant(X, has_constant="add")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = sm.OLS(y, X_const).fit()

        conf = model.conf_int()

        coef_table = []

        for term in model.params.index:
            coef_table.append({
                "term": str(term),
                "coef": _round_or_none(model.params[term]),
                "std_err": _round_or_none(model.bse[term]),
                "t": _round_or_none(model.tvalues[term]),
                "p_value": _round_or_none(model.pvalues[term]),
                "ci_lower": _round_or_none(conf.loc[term, 0]) if term in conf.index else None,
                "ci_upper": _round_or_none(conf.loc[term, 1]) if term in conf.index else None,
            })

        details = {
            **prep["details"],
            "model_type": "OLS multiple linear regression",
            "r_squared": _round_or_none(model.rsquared),
            "adj_r_squared": _round_or_none(model.rsquared_adj),
            "f_statistic": _round_or_none(model.fvalue),
            "f_p_value": _round_or_none(model.f_pvalue),
            "aic": _round_or_none(model.aic),
            "bic": _round_or_none(model.bic),
            "nobs": int(model.nobs),
            "df_model": _round_or_none(model.df_model),
            "df_resid": _round_or_none(model.df_resid),
            "coef_table": coef_table,
        }

        status = "ok"
        message = "Multiple regression completed successfully."

        if (
            int(model.nobs) < 30
            or prep["details"]["n_eff"] < 5 * (prep["details"]["p_eff"] + 1)
        ):
            status = "warning"
            message = (
                "Multiple regression completed, but sample size is small relative "
                "to the number of predictors. Interpret cautiously."
            )

        if status == "warning":
            return _warning(message, details)

        return _ok(message, details)

    except Exception as e:
        return _failed(
            "OLS_FIT_EXCEPTION",
            "OLS model fitting failed.",
            e,
        )


def extract_linear_model(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target = arguments.get("target_col") or payload.get("target")
    features = arguments.get("feature_cols") or []
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

    tables: Dict[str, Any] = {}

    coef_table = payload.get("coef_table", [])
    if coef_table:
        tables["coef_table"] = coef_table

    # Internal/system fields. These should not appear in the main report.
    metadata = compact_dict({
        "model_type": payload.get("model_type"),
        "aic": payload.get("aic"),
        "bic": payload.get("bic"),
        "df_model": payload.get("df_model"),
        "df_resid": payload.get("df_resid"),
        "n_eff": payload.get("n_eff"),
        "p_eff": payload.get("p_eff"),
        "target": payload.get("target"),
        "encoded_columns": payload.get("encoded_columns"),
        "used_features": payload.get("used_features"),
        "excluded_features": payload.get("excluded_features"),
        "raw_feature_count": payload.get("raw_feature_count"),
        "encoded_column_count": payload.get("encoded_column_count"),
        "min_required": payload.get("min_required"),
    })

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


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_multiple_regression",
    display_name="Linear Model",
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "feature_cols": list,
        },
        optional={
            "max_missing_rate": float,
            "max_categorical_levels": int,
            "numeric_parse_threshold": float,
            "min_n_per_parameter": int,
        },
        column_args=[
            "target_col",
        ],
        column_list_args=[
            "feature_cols",
        ],
        allow_all_columns=False,
    ),
    execute=execute_linear_model,
    extractor=extract_linear_model,
    guardrail_evaluators=[
        evaluate_regression_guardrails,
    ],
    display_config=LINEAR_MODEL_DISPLAY,
))