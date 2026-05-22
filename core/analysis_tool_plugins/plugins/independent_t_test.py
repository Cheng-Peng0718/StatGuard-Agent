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
from core.analysis_tool_plugins.shared.group_comparison_guardrails import (
    evaluate_group_comparison_guardrails,
)
from core.analysis_tool_plugins.shared.effect_size_ci import (
    cohens_d_independent_ci,
    hedges_g_independent_ci,
)
from core.analysis_tool_plugins.shared.apa_writers import write_apa_independent_t_test


MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    "NA", "N/A", "NaN", "NULL", "None", "Missing", "Unknown",
}


def _ok(message: str, details: Dict[str, Any], artifacts=None):
    return {
        "status": "ok",
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


def _safe_divide(numerator: float, denominator: float) -> float | None:
    try:
        numerator = float(numerator)
        denominator = float(denominator)
        if not math.isfinite(numerator) or not math.isfinite(denominator) or abs(denominator) < 1e-12:
            return None
        return numerator / denominator
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


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_missing = {str(x).strip().lower() for x in MISSING_TOKENS}

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            def norm(x):
                if isinstance(x, str):
                    lx = x.strip().lower()
                    if lx in lower_missing:
                        return np.nan
                    return x.strip()
                return x

            df[col] = df[col].map(norm)

    return df.replace([np.inf, -np.inf], np.nan)


def _welch_df(x1: np.ndarray, x2: np.ndarray) -> float | None:
    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return None

    s1 = float(np.var(x1, ddof=1))
    s2 = float(np.var(x2, ddof=1))

    a = s1 / n1
    b = s2 / n2

    denominator = (a ** 2) / (n1 - 1) + (b ** 2) / (n2 - 1)

    return _safe_divide((a + b) ** 2, denominator)


def _cohens_d_and_hedges_g(x1: np.ndarray, x2: np.ndarray) -> tuple[float | None, float | None]:
    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return None, None

    s1 = float(np.var(x1, ddof=1))
    s2 = float(np.var(x2, ddof=1))

    pooled_var = _safe_divide((n1 - 1) * s1 + (n2 - 1) * s2, n1 + n2 - 2)

    if pooled_var is None or pooled_var < 0:
        return None, None

    pooled_sd = math.sqrt(pooled_var)
    d = _safe_divide(float(np.mean(x1) - np.mean(x2)), pooled_sd)

    if d is None:
        return None, None

    correction = 1 - 3 / (4 * (n1 + n2) - 9) if (n1 + n2) > 2 else 1

    return d, d * correction


def _mean_diff_ci(x1: np.ndarray, x2: np.ndarray, alpha: float) -> tuple[float | None, float | None]:
    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return None, None

    diff = float(np.mean(x1) - np.mean(x2))
    se = math.sqrt(float(np.var(x1, ddof=1)) / n1 + float(np.var(x2, ddof=1)) / n2)
    df = _welch_df(x1, x2)

    if df is None or se <= 0:
        return None, None

    t_crit = float(stats.t.ppf(1 - alpha / 2, df))

    return diff - t_crit * se, diff + t_crit * se


def _interpret_d(d: float | None) -> str | None:
    if d is None:
        return None

    a = abs(float(d))

    if a < 0.2:
        return "negligible"
    if a < 0.5:
        return "small"
    if a < 0.8:
        return "medium"

    return "large"


def execute_independent_t_test(context) -> Dict[str, Any]:
    """
    Welch independent two-sample t-test with effect size and confidence interval.

    Args:
        target_col: numeric outcome
        group_col: grouping column
        group1_val: label for group 1
        group2_val: label for group 2
        alpha: optional, default 0.05
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        df = _standardize_dataframe(df)

        target_col = _get_arg(context, "target_col")
        group_col = _get_arg(context, "group_col")
        group1_val = _get_arg(context, "group1_val")
        group2_val = _get_arg(context, "group2_val")

        try:
            alpha = float(_get_arg(context, "alpha", 0.05))
        except Exception:
            alpha = 0.05

        if not (0 < alpha < 1):
            alpha = 0.05

        if not target_col or not group_col or group1_val is None or group2_val is None:
            return _blocked(
                "MISSING_T_TEST_ARGS",
                "target_col, group_col, group1_val, and group2_val are required.",
                details={
                    "target_col": target_col,
                    "group_col": group_col,
                    "group1_val": group1_val,
                    "group2_val": group2_val,
                },
                suggested_next_actions=[
                    "Specify a numeric target column, a grouping column, and two group values."
                ],
            )

        missing_cols = [c for c in [target_col, group_col] if c not in df.columns]

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

        y = pd.to_numeric(df[target_col], errors="coerce").replace([np.inf, -np.inf], np.nan)

        g1 = y[df[group_col] == group1_val].dropna().astype(float)
        g2 = y[df[group_col] == group2_val].dropna().astype(float)

        if len(g1) < 2 or len(g2) < 2:
            return _blocked(
                "INSUFFICIENT_GROUP_SIZE",
                "Each group needs at least 2 valid numeric observations.",
                details={
                    "group1": str(group1_val),
                    "group1_n": int(len(g1)),
                    "group2": str(group2_val),
                    "group2_n": int(len(g2)),
                },
                suggested_next_actions=[
                    "Choose groups with at least 2 valid numeric observations each."
                ],
            )

        x1 = g1.to_numpy(dtype=float)
        x2 = g2.to_numpy(dtype=float)

        t_stat, p_value = stats.ttest_ind(
            x1,
            x2,
            equal_var=False,
            nan_policy="omit",
        )

        df_welch = _welch_df(x1, x2)
        d, g = _cohens_d_and_hedges_g(x1, x2)
        d_ci_low, d_ci_high = cohens_d_independent_ci(d, len(x1), len(x2), alpha) if d is not None else (None, None)
        g_ci_low, g_ci_high = hedges_g_independent_ci(g, len(x1), len(x2), alpha) if g is not None else (None, None)
        ci_low, ci_high = _mean_diff_ci(x1, x2, alpha)
        mean_diff = float(np.mean(x1) - np.mean(x2))

        # Auxiliary normality check
        shapiro_rows = []

        for name, arr in [(str(group1_val), x1), (str(group2_val), x2)]:
            n = len(arr)

            if n < 3 or n > 5000:
                shapiro_rows.append({
                    "group": name,
                    "n": int(n),
                    "statistic": None,
                    "p_value": None,
                    "normal_at_0_05": None,
                    "note": (
                        "n < 3 (not enough)" if n < 3
                        else "n > 5000 (use Anderson-Darling instead)"
                    ),
                })
                continue

            try:
                stat_s, p_s = stats.shapiro(arr)
                shapiro_rows.append({
                    "group": name,
                    "n": int(n),
                    "statistic": _round_or_none(stat_s),
                    "p_value": _round_or_none(p_s),
                    "normal_at_0_05": bool(p_s >= 0.05) if math.isfinite(float(p_s)) else None,
                    "note": None,
                })
            except Exception:
                shapiro_rows.append({
                    "group": name,
                    "n": int(n),
                    "statistic": None,
                    "p_value": None,
                    "normal_at_0_05": None,
                    "note": "Shapiro-Wilk failed",
                })

        any_non_normal = any(row.get("normal_at_0_05") is False for row in shapiro_rows)

        assumptions = [
            "Welch's t-test does not assume equal variances.",
            "Observations within and across groups are assumed to be independent.",
            f"This is an observational comparison; with alpha={alpha}, the reported significance is association-based, not causal.",
        ]

        if any_non_normal:
            assumptions.append(
                "Shapiro-Wilk flagged possible non-normality; with small groups consider Mann-Whitney U as a non-parametric alternative."
            )

        details = {
            "method": "Welch two-sample t-test",
            "target_col": target_col,
            "group_col": group_col,
            "alpha": alpha,
            "group1": str(group1_val),
            "group1_n": int(len(x1)),
            "group1_mean": _round_or_none(np.mean(x1)),
            "group1_std": _round_or_none(np.std(x1, ddof=1)),
            "group2": str(group2_val),
            "group2_n": int(len(x2)),
            "group2_mean": _round_or_none(np.mean(x2)),
            "group2_std": _round_or_none(np.std(x2, ddof=1)),
            "mean_difference_group1_minus_group2": _round_or_none(mean_diff),
            "mean_difference_ci_low": _round_or_none(ci_low),
            "mean_difference_ci_high": _round_or_none(ci_high),
            "t_statistic": _round_or_none(t_stat),
            "degrees_of_freedom": _round_or_none(df_welch),
            "p_value": _round_or_none(p_value),
            "effect_size_name": "Hedges g",
            "effect_size": _round_or_none(g),
            "effect_size_ci_low": _round_or_none(g_ci_low),
            "effect_size_ci_high": _round_or_none(g_ci_high),
            "effect_size_magnitude": _interpret_d(g),
            "cohens_d": _round_or_none(d),
            "cohens_d_ci_low": _round_or_none(d_ci_low),
            "cohens_d_ci_high": _round_or_none(d_ci_high),
            "significant_at_alpha": (
                bool(p_value < alpha) if math.isfinite(float(p_value)) else None
            ),
            "significant_at_0_05": (
                bool(p_value < 0.05) if math.isfinite(float(p_value)) else None
            ),
            "shapiro_per_group": shapiro_rows,
            "assumptions_and_limitations": assumptions,
        }

        return _ok("Welch independent t-test completed.", details)

    except Exception as e:
        return _failed(
            "T_TEST_EXCEPTION",
            "T-test failed.",
            e,
        )


def extract_independent_t_test(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target_col = payload.get("target_col") or arguments.get("target_col")
    group_col = payload.get("group_col") or arguments.get("group_col")

    title = "Independent t-test"
    if target_col and group_col:
        title = f"Independent t-test: {target_col} by {group_col}"

    metrics = compact_dict({
        "method": payload.get("method"),
        "alpha": payload.get("alpha"),
        "group1": payload.get("group1"),
        "group1_n": payload.get("group1_n"),
        "group1_mean": payload.get("group1_mean"),
        "group2": payload.get("group2"),
        "group2_n": payload.get("group2_n"),
        "group2_mean": payload.get("group2_mean"),
        "mean_difference_group1_minus_group2": payload.get("mean_difference_group1_minus_group2"),
        "mean_difference_ci_low": payload.get("mean_difference_ci_low"),
        "mean_difference_ci_high": payload.get("mean_difference_ci_high"),
        "t_statistic": payload.get("t_statistic"),
        "degrees_of_freedom": payload.get("degrees_of_freedom"),
        "p_value": payload.get("p_value"),
        "significant_at_alpha": payload.get("significant_at_alpha"),
        "significant_at_0_05": payload.get("significant_at_0_05"),
        "effect_size_name": payload.get("effect_size_name"),
        "effect_size": payload.get("effect_size"),
        "effect_size_ci_low": payload.get("effect_size_ci_low"),
        "effect_size_ci_high": payload.get("effect_size_ci_high"),
        "effect_size_magnitude": payload.get("effect_size_magnitude"),
        "cohens_d": payload.get("cohens_d"),
        "cohens_d_ci_low": payload.get("cohens_d_ci_low"),
        "cohens_d_ci_high": payload.get("cohens_d_ci_high"),
    })

    tables: Dict[str, Any] = {}

    shapiro_rows = payload.get("shapiro_per_group") or []
    if any(row.get("p_value") is not None for row in shapiro_rows):
        tables["shapiro_per_group"] = shapiro_rows

    if payload.get("assumptions_and_limitations"):
        tables["assumptions_and_limitations"] = [
            {"item": item} for item in payload.get("assumptions_and_limitations", [])
        ]

    metadata = compact_dict({
        "target_col": target_col,
        "group_col": group_col,
        "group1_std": payload.get("group1_std"),
        "group2_std": payload.get("group2_std"),
    })

    summary = "Completed Welch independent two-sample t-test."
    if target_col and group_col:
        summary += f" Compared `{target_col}` across groups of `{group_col}`."

    if payload.get("effect_size") is not None:
        summary += (
            f" Hedges' g = {payload.get('effect_size')} "
            f"({payload.get('effect_size_magnitude') or 'magnitude n/a'})."
        )

    return title, summary, metrics, tables, metadata


INDEPENDENT_T_TEST_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "alpha": "Alpha",
            "group1": "Group 1",
            "group1_n": "Group 1 n",
            "group1_mean": "Group 1 mean",
            "group2": "Group 2",
            "group2_n": "Group 2 n",
            "group2_mean": "Group 2 mean",
            "mean_difference_group1_minus_group2": "Mean difference (g1 - g2)",
            "mean_difference_ci_low": "Mean diff CI lower",
            "mean_difference_ci_high": "Mean diff CI upper",
            "t_statistic": "t statistic",
            "degrees_of_freedom": "Degrees of freedom (Welch)",
            "p_value": "p-value",
            "significant_at_alpha": "Significant",
            "significant_at_0_05": "Significant at 0.05",
            "effect_size_name": "Effect size",
            "effect_size": "Effect size value",
            "effect_size_ci_low": "Effect size 95% CI lower",
            "effect_size_ci_high": "Effect size 95% CI upper",
            "effect_size_magnitude": "Effect size magnitude",
            "cohens_d": "Cohen's d",
            "cohens_d_ci_low": "Cohen's d 95% CI lower",
            "cohens_d_ci_high": "Cohen's d 95% CI upper",
        },
        formatters={
            "group1_mean": format_number,
            "group2_mean": format_number,
            "mean_difference_group1_minus_group2": format_number,
            "mean_difference_ci_low": format_number,
            "mean_difference_ci_high": format_number,
            "t_statistic": format_number,
            "degrees_of_freedom": format_number,
            "p_value": format_p_value,
            "significant_at_alpha": format_bool_yes_no,
            "significant_at_0_05": format_bool_yes_no,
            "effect_size": format_number,
            "effect_size_ci_low": format_number,
            "effect_size_ci_high": format_number,
            "cohens_d": format_number,
            "cohens_d_ci_low": format_number,
            "cohens_d_ci_high": format_number,
        },
        order=[
            "method",
            "alpha",
            "group1",
            "group1_n",
            "group1_mean",
            "group2",
            "group2_n",
            "group2_mean",
            "mean_difference_group1_minus_group2",
            "mean_difference_ci_low",
            "mean_difference_ci_high",
            "t_statistic",
            "degrees_of_freedom",
            "p_value",
            "significant_at_alpha",
            "significant_at_0_05",
            "effect_size_name",
            "effect_size",
            "effect_size_ci_low",
            "effect_size_ci_high",
            "effect_size_magnitude",
            "cohens_d",
            "cohens_d_ci_low",
            "cohens_d_ci_high",
        ],
    ),
    tables={
        "shapiro_per_group": TableDisplayConfig(
            column_labels={
                "group": "Group",
                "n": "N",
                "statistic": "Shapiro W",
                "p_value": "p-value",
                "normal_at_0_05": "Normal at 0.05",
                "note": "Note",
            },
            column_order=[
                "group",
                "n",
                "statistic",
                "p_value",
                "normal_at_0_05",
                "note",
            ],
            column_formatters={
                "statistic": format_number,
                "p_value": format_p_value,
                "normal_at_0_05": format_bool_yes_no,
            },
        ),
        "assumptions_and_limitations": TableDisplayConfig(
            column_labels={
                "item": "Assumption / limitation",
            },
            column_order=["item"],
        ),
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_independent_t_test",
    evidence_categories=["group_comparison", "statistical_inference"],
    display_name="Independent t-test",
    requires_confirmation=False,
    is_inferential=True,
    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "group_col": str,
            "group1_val": object,
            "group2_val": object,
        },
        optional={
            "alpha": float,
        },
        column_args=[
            "target_col",
            "group_col",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_independent_t_test,
    extractor=extract_independent_t_test,
    guardrail_evaluators=[
        evaluate_group_comparison_guardrails
    ],
    apa_methods_writer=write_apa_independent_t_test,
    display_config=INDEPENDENT_T_TEST_DISPLAY,
))