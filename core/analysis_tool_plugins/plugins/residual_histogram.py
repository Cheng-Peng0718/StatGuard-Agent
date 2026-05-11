from typing import Any, Dict, Tuple
import math
import os
import uuid
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.regression_utils import prepare_regression_data
from core.guardrails import evaluate_residual_guardrails


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


def _workspace_dir(context) -> str:
    return getattr(context, "workspace_dir", ".") or "."


def _artifact_path(context, output_path: Any = None) -> str:
    if output_path:
        return str(output_path)

    artifact_dir = os.path.join(_workspace_dir(context), "artifacts")
    os.makedirs(artifact_dir, exist_ok=True)

    return os.path.join(
        artifact_dir,
        f"residual_histogram_{uuid.uuid4().hex[:8]}.png",
    )


def _residual_flags(residuals: pd.Series) -> list[str]:
    flags = []

    residuals = pd.Series(residuals).dropna()

    if len(residuals) == 0:
        return flags

    skewness = residuals.skew()
    kurtosis = residuals.kurtosis()

    if skewness is not None and math.isfinite(float(skewness)):
        if skewness > 1:
            flags.append("right_skew_detected")
        elif skewness < -1:
            flags.append("left_skew_detected")

    if kurtosis is not None and math.isfinite(float(kurtosis)):
        if kurtosis > 3:
            flags.append("heavy_tails_possible")

    sd = residuals.std()

    if sd is not None and math.isfinite(float(sd)) and sd > 0:
        outliers_2sd = int((residuals.abs() > 2 * sd).sum())
        outliers_3sd = int((residuals.abs() > 3 * sd).sum())

        if outliers_2sd > 0:
            flags.append("possible_residual_outliers_abs_2sd")

        if outliers_3sd > 0:
            flags.append("possible_extreme_residual_outliers_abs_3sd")

    if flags:
        flags.append("non_normal_residual_pattern_possible")

    # Deduplicate while preserving order.
    seen = set()
    deduped = []
    for flag in flags:
        if flag not in seen:
            deduped.append(flag)
            seen.add(flag)

    return deduped


def _plot_residual_histogram(residuals: pd.Series, path: str) -> None:
    plt.figure(figsize=(8, 5))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.axvline(0, linestyle="--", linewidth=1)
    plt.title("Residual Histogram")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()


def execute_residual_histogram(context) -> Dict[str, Any]:
    """
    Fit OLS with the shared regression design matrix and generate a residual histogram.

    Args:
        target_col: numeric outcome column
        feature_cols: list of predictor columns
        output_path: optional explicit output path
        max_missing_rate: optional, default 0.40
        max_categorical_levels: optional, default 10
        numeric_parse_threshold: optional, default 0.85
        min_n_per_parameter: optional, default 3
    """
    try:
        df = context.load_df()

        prep = prepare_regression_data(
            df,
            _get_arg(context, "target_col"),
            _get_arg(context, "feature_cols", []),
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

        residuals = pd.Series(model.resid).dropna()

        if len(residuals) == 0:
            return {
                "status": "blocked",
                "error_code": "NO_RESIDUALS",
                "message": "No residuals were available after model fitting.",
                "recoverable": True,
                "details": prep["details"],
                "artifacts": [],
            }

        output_path = _artifact_path(context, _get_arg(context, "output_path", None))
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        _plot_residual_histogram(residuals, output_path)

        residual_mean = _round_or_none(residuals.mean())
        residual_std = _round_or_none(residuals.std())
        residual_skewness = _round_or_none(residuals.skew())
        residual_kurtosis_fisher = _round_or_none(residuals.kurtosis())

        sd = residuals.std()
        if sd is not None and math.isfinite(float(sd)) and sd > 0:
            outliers_abs_2sd = int((residuals.abs() > 2 * sd).sum())
            outliers_abs_3sd = int((residuals.abs() > 3 * sd).sum())
        else:
            outliers_abs_2sd = 0
            outliers_abs_3sd = 0

        diagnostic_flags = _residual_flags(residuals)

        details = {
            **prep["details"],
            "n_residuals": int(len(residuals)),
            "residual_mean": residual_mean,
            "residual_std": residual_std,
            "residual_skewness": residual_skewness,
            "residual_kurtosis_fisher": residual_kurtosis_fisher,
            "outliers_abs_2sd": outliers_abs_2sd,
            "outliers_abs_3sd": outliers_abs_3sd,
            "diagnostic_flags": diagnostic_flags,
            "plot_path": output_path,

            # Keep this explicit for deliverable / downstream checks.
            "residual_summary": {
                "n_residuals": int(len(residuals)),
                "residual_mean": residual_mean,
                "residual_std": residual_std,
                "residual_skewness": residual_skewness,
                "residual_kurtosis_fisher": residual_kurtosis_fisher,
                "outliers_abs_2sd": outliers_abs_2sd,
                "outliers_abs_3sd": outliers_abs_3sd,
                "diagnostic_flags": diagnostic_flags,
            },
        }

        artifacts = [
            {
                "type": "png",
                "name": "Residual Histogram",
                "path": output_path,
            }
        ]

        if diagnostic_flags:
            return _warning(
                "Residual histogram generated with diagnostic flags.",
                details,
                artifacts=artifacts,
            )

        return _ok(
            "Residual histogram generated successfully.",
            details,
            artifacts=artifacts,
        )

    except Exception as e:
        return _failed(
            "RESIDUAL_HISTOGRAM_EXCEPTION",
            "Residual histogram generation failed.",
            e,
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
        return "; ".join(
            mapping.get(str(x), str(x).replace("_", " "))
            for x in value
        )

    return mapping.get(str(value), str(value).replace("_", " "))


def extract_residual_histogram(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Residual Histogram"

    diagnostic_flags = payload.get("diagnostic_flags", []) or []

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

    metadata = compact_dict({
        "plot_path": payload.get("plot_path"),
        "residual_summary": payload.get("residual_summary"),
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

    summary = "Generated a residual histogram."

    if diagnostic_flags:
        summary += f" Diagnostic flags: {format_diagnostic_flags(diagnostic_flags)}."

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
    ),
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="generate_residual_histogram",
    display_name="Residual Histogram",
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "feature_cols": list,
        },
        optional={
            "output_path": str,
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
    execute=execute_residual_histogram,
    extractor=extract_residual_histogram,
    guardrail_evaluators=[
        evaluate_residual_guardrails,
    ],
    display_config=RESIDUAL_HISTOGRAM_DISPLAY,
))