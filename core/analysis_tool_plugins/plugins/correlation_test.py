from typing import Any, Dict, Tuple
import math

import numpy as np
import pandas as pd
from scipy import stats

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_bool_yes_no,
    format_number,
    format_p_value,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.guardrails import _new_finding

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


def _blocked(error_code: str, message: str, details=None, suggested_next_actions=None):
    result = {
        "status": "blocked",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": details or {},
        "artifacts": [],
    }

    if suggested_next_actions:
        result["suggested_next_actions"] = suggested_next_actions

    return result


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


def _fisher_z_ci_for_pearson(r: float, n: int, alpha: float) -> tuple[float | None, float | None]:
    """
    Fisher z-transform confidence interval for Pearson correlation.

    z = 0.5 * ln((1 + r) / (1 - r))     (variance-stabilizing transform)
    se = 1 / sqrt(n - 3)
    CI in z-space: z +/- z_crit * se
    Back-transform: r = (exp(2z) - 1) / (exp(2z) + 1) = tanh(z)
    """
    if n < 4 or not math.isfinite(float(r)) or abs(r) >= 1.0:
        return None, None

    try:
        z = math.atanh(float(r))
        se = 1.0 / math.sqrt(n - 3)
        z_crit = float(stats.norm.ppf(1 - alpha / 2))

        lo = math.tanh(z - z_crit * se)
        hi = math.tanh(z + z_crit * se)

        return lo, hi
    except Exception:
        return None, None


def _spearman_ci(r: float, n: int, alpha: float) -> tuple[float | None, float | None]:
    """
    Approximate confidence interval for Spearman correlation using the Fieller
    standard error 1 / sqrt(n - 3) on the Fisher z scale. This is a common
    practical approximation reported by SAS and many texts when n is moderate.
    """
    return _fisher_z_ci_for_pearson(r, n, alpha)


def _interpret_correlation_magnitude(r: float | None) -> str | None:
    """Common rules of thumb for |r|."""
    if r is None:
        return None

    a = abs(float(r))

    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "moderate"
    if a < 0.70:
        return "large"

    return "very large"


def execute_correlation_test(context) -> Dict[str, Any]:
    """
    Pairwise correlation test (Pearson / Spearman / Kendall) with confidence
    intervals and magnitude interpretation.

    Args:
        x_col: first numeric column
        y_col: second numeric column
        method: 'pearson', 'spearman', or 'kendall', default 'pearson'
        alpha: optional, default 0.05
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        x_col = _get_arg(context, "x_col")
        y_col = _get_arg(context, "y_col")
        method = str(_get_arg(context, "method", "pearson")).lower().strip()

        try:
            alpha = float(_get_arg(context, "alpha", 0.05))
        except Exception:
            alpha = 0.05

        if not (0 < alpha < 1):
            alpha = 0.05

        if not x_col or not y_col:
            return _blocked(
                "MISSING_CORRELATION_ARGS",
                "x_col and y_col are required.",
                suggested_next_actions=[
                    "Specify two numeric columns for the correlation test."
                ],
            )

        missing_cols = [col for col in [x_col, y_col] if col not in df.columns]

        if missing_cols:
            return _blocked(
                "COLUMNS_NOT_FOUND",
                f"Columns not found: {missing_cols}",
                details={
                    "missing_cols": missing_cols,
                    "available_columns": list(df.columns),
                },
                suggested_next_actions=[
                    "Inspect dataset columns and retry with valid column names."
                ],
            )

        if method not in {"pearson", "spearman", "kendall"}:
            return _blocked(
                "UNSUPPORTED_CORRELATION_METHOD",
                f"Unsupported correlation method: {method}",
                details={"method": method},
                suggested_next_actions=[
                    "Use method='pearson', method='spearman', or method='kendall'."
                ],
            )

        x = pd.to_numeric(df[x_col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        y = pd.to_numeric(df[y_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        work = pd.DataFrame({"x": x, "y": y}).dropna()
        nobs = int(len(work))

        if nobs < 3:
            return _blocked(
                "INSUFFICIENT_VALID_OBSERVATIONS",
                "Correlation test requires at least 3 complete numeric observations.",
                details={
                    "x_col": x_col,
                    "y_col": y_col,
                    "nobs": nobs,
                },
                suggested_next_actions=[
                    "Choose columns with more complete numeric observations."
                ],
            )

        if work["x"].nunique() <= 1 or work["y"].nunique() <= 1:
            return _blocked(
                "CONSTANT_INPUT",
                "Correlation is undefined when one selected column is constant.",
                details={
                    "x_col": x_col,
                    "y_col": y_col,
                    "x_unique": int(work["x"].nunique()),
                    "y_unique": int(work["y"].nunique()),
                },
                suggested_next_actions=[
                    "Choose two non-constant numeric columns."
                ],
            )

        ci_lower = None
        ci_upper = None
        ci_method_label = None

        if method == "pearson":
            correlation, p_value = stats.pearsonr(work["x"], work["y"])
            ci_lower, ci_upper = _fisher_z_ci_for_pearson(
                float(correlation), nobs, alpha
            )
            ci_method_label = "Fisher z-transform"
        elif method == "spearman":
            correlation, p_value = stats.spearmanr(work["x"], work["y"])
            ci_lower, ci_upper = _spearman_ci(float(correlation), nobs, alpha)
            ci_method_label = "Fisher z-transform (approximate for Spearman)"
        else:
            # Kendall's tau-b (tied-rank corrected, scipy default)
            correlation, p_value = stats.kendalltau(work["x"], work["y"])
            ci_lower, ci_upper = None, None
            ci_method_label = "Not provided for Kendall's tau"

        method_label_map = {
            "pearson": "Pearson product-moment correlation",
            "spearman": "Spearman rank correlation",
            "kendall": "Kendall's tau-b rank correlation",
        }

        magnitude = _interpret_correlation_magnitude(
            float(correlation) if math.isfinite(float(correlation)) else None
        )

        assumptions: list[str] = [
            "Reported p-value tests the null hypothesis of zero correlation.",
            "Observations are assumed to be independent.",
        ]

        if method == "pearson":
            assumptions.append(
                "Pearson correlation measures linear association and is sensitive to outliers and extreme non-normality. "
                "Consider Spearman or Kendall when the relationship is monotonic but not linear, or when outliers are present."
            )
            assumptions.append(
                "The reported CI uses the Fisher z-transform, which assumes approximate bivariate normality."
            )
        elif method == "spearman":
            assumptions.append(
                "Spearman correlation measures monotonic association on ranks; it is robust to outliers and non-normality."
            )
            assumptions.append(
                "The reported CI is an approximate Fisher z-transform interval (no closed-form parametric CI exists for Spearman)."
            )
        else:
            assumptions.append(
                "Kendall's tau-b measures monotonic association on ranks with a tied-rank correction. It is robust but more conservative than Spearman."
            )
            assumptions.append(
                "No closed-form parametric CI is provided for Kendall's tau; rely on the p-value for inference."
            )

        if nobs < 30:
            assumptions.append(
                f"Sample size is small (n={nobs}); confidence intervals and p-values should be interpreted cautiously."
            )

        details = {
            "x_col": x_col,
            "y_col": y_col,
            "method": method,
            "method_label": method_label_map.get(method, method),
            "alpha": alpha,
            "nobs": nobs,
            "correlation": _round_or_none(correlation),
            "correlation_magnitude": magnitude,
            "p_value": _round_or_none(p_value),
            "significant_at_alpha": (
                bool(p_value < alpha) if math.isfinite(float(p_value)) else None
            ),
            "significant_at_0_05": (
                bool(p_value < 0.05) if math.isfinite(float(p_value)) else None
            ),
            "ci_lower": _round_or_none(ci_lower),
            "ci_upper": _round_or_none(ci_upper),
            "ci_method": ci_method_label,
            "assumptions_and_limitations": assumptions,
        }

        status = "ok"
        message = f"{method_label_map.get(method, method)} test completed."

        if nobs < 30:
            status = "warning"
            message = (
                f"{method_label_map.get(method, method)} test completed, but sample size "
                f"is small (n={nobs}). Interpret with caution."
            )

        if status == "warning":
            return _warning(message, details)

        return _ok(message, details)

    except Exception as e:
        return _failed(
            "CORRELATION_TEST_EXCEPTION",
            "Correlation test failed.",
            e,
        )


def extract_correlation_test(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    x_col = payload.get("x_col") or arguments.get("x_col")
    y_col = payload.get("y_col") or arguments.get("y_col")
    method = payload.get("method")
    method_label = payload.get("method_label") or method

    if x_col and y_col:
        title = f"Correlation Test: {x_col} vs {y_col}"
    else:
        title = "Correlation Test"

    metrics = compact_dict({
        "method": method,
        "method_label": method_label,
        "alpha": payload.get("alpha"),
        "nobs": payload.get("nobs"),
        "correlation": payload.get("correlation"),
        "correlation_magnitude": payload.get("correlation_magnitude"),
        "p_value": payload.get("p_value"),
        "significant_at_alpha": payload.get("significant_at_alpha"),
        "ci_lower": payload.get("ci_lower"),
        "ci_upper": payload.get("ci_upper"),
        "ci_method": payload.get("ci_method"),
    })

    tables: Dict[str, Any] = {}

    if payload.get("assumptions_and_limitations"):
        tables["assumptions_and_limitations"] = [
            {"item": item} for item in payload.get("assumptions_and_limitations", [])
        ]

    metadata = compact_dict({
        "x_col": x_col,
        "y_col": y_col,
        "significant_at_0_05": payload.get("significant_at_0_05"),
    })

    summary = "Completed correlation test."

    if method_label:
        summary += f" Method: `{method_label}`."

    if x_col and y_col:
        summary += f" Variables: `{x_col}` and `{y_col}`."

    if payload.get("correlation") is not None:
        summary += (
            f" r={payload.get('correlation')} "
            f"({payload.get('correlation_magnitude') or 'magnitude n/a'})."
        )

    if payload.get("ci_lower") is not None and payload.get("ci_upper") is not None:
        summary += f" CI=[{payload.get('ci_lower')}, {payload.get('ci_upper')}]."

    return title, summary, metrics, tables, metadata


CORRELATION_TEST_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method (key)",
            "method_label": "Method",
            "alpha": "Alpha",
            "nobs": "Observations used",
            "correlation": "Correlation coefficient",
            "correlation_magnitude": "Correlation magnitude",
            "p_value": "p-value",
            "significant_at_alpha": "Significant",
            "ci_lower": "CI lower",
            "ci_upper": "CI upper",
            "ci_method": "CI method",
        },
        formatters={
            "correlation": lambda x: format_number(x, digits=4),
            "p_value": format_p_value,
            "significant_at_alpha": format_bool_yes_no,
            "ci_lower": lambda x: format_number(x, digits=4),
            "ci_upper": lambda x: format_number(x, digits=4),
        },
        order=[
            "method",
            "method_label",
            "alpha",
            "nobs",
            "correlation",
            "correlation_magnitude",
            "p_value",
            "significant_at_alpha",
            "ci_lower",
            "ci_upper",
            "ci_method",
        ],
    ),
    tables={
        "assumptions_and_limitations": TableDisplayConfig(
            column_labels={
                "item": "Assumption / limitation",
            },
            column_order=["item"],
        ),
    },
)
# ==========================================================
# Guardrails
# ==========================================================

def evaluate_correlation_guardrails(run: Dict[str, Any]) -> list[Dict[str, Any]]:
    findings: list[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}

    r = metrics.get("correlation")
    magnitude = metrics.get("correlation_magnitude")
    p_value = metrics.get("p_value")
    sig = metrics.get("significant_at_alpha")
    method = metrics.get("method")
    nobs = metrics.get("nobs")
    ci_lower = metrics.get("ci_lower")
    ci_upper = metrics.get("ci_upper")

    if r is not None and magnitude is not None:
        findings.append(_new_finding(
            category="effect_size",
            severity="info",
            title=f"Correlation magnitude: {magnitude}",
            message=(
                f"Estimated correlation r = {r} (interpretation: {magnitude}). "
                "Statistical significance does not by itself indicate a practically large effect."
            ),
            evidence={
                "method": method,
                "correlation": r,
                "correlation_magnitude": magnitude,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            },
        ))

    if sig is True and magnitude in {"negligible", "small"}:
        findings.append(_new_finding(
            category="interpretation",
            severity="warning",
            title="Significant but small correlation",
            message=(
                "The correlation is statistically significant but the magnitude is small. "
                "With large samples even tiny correlations can reach statistical significance; "
                "judge practical relevance via the effect size and the confidence interval."
            ),
            evidence={
                "p_value": p_value,
                "correlation": r,
                "correlation_magnitude": magnitude,
                "ci_lower": ci_lower,
                "ci_upper": ci_upper,
            },
        ))

    if nobs is not None:
        try:
            if int(nobs) < 30:
                findings.append(_new_finding(
                    category="sample_size",
                    severity="warning",
                    title="Small sample for correlation",
                    message=(
                        f"Only {nobs} complete observations were available. Correlation "
                        f"estimates and confidence intervals are unstable for small samples."
                    ),
                    evidence={"nobs": nobs},
                    recommendation=(
                        "Treat the point estimate cautiously and rely on the confidence interval."
                    ),
                ))
        except Exception:
            pass

    if method == "pearson":
        findings.append(_new_finding(
            category="method_guidance",
            severity="info",
            title="Pearson assumes a linear relationship and is sensitive to outliers",
            message=(
                "Pearson's correlation measures linear association. If the relationship is "
                "monotonic but nonlinear, or outliers are present, Spearman's rho or Kendall's "
                "tau may better describe the dependency."
            ),
            evidence={"method": method},
        ))

    return findings


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_correlation_test",
    display_name="Correlation Test",
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={
            "x_col": str,
            "y_col": str,
        },
        optional={
            "method": str,
            "alpha": float,
        },
        column_args=[
            "x_col",
            "y_col",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_correlation_test,
    extractor=extract_correlation_test,
    guardrail_evaluators=[],
    display_config=CORRELATION_TEST_DISPLAY,
))