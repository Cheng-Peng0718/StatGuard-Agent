from __future__ import annotations

from pathlib import Path
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


MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    "NA", "N/A", "NaN", "NULL", "None", "Missing", "Unknown",
}


def _get_arguments(context) -> dict[str, Any]:
    return getattr(context, "arguments", None) or getattr(context, "args", None) or {}


def _find_active_data_version(context) -> dict[str, Any] | None:
    active_id = getattr(context, "active_data_version_id", None)
    data_versions = getattr(context, "data_versions", []) or []

    if not active_id:
        return None

    for version in data_versions:
        if isinstance(version, dict) and version.get("version_id") == active_id:
            return version

    return None


def _load_active_dataframe(context) -> pd.DataFrame:
    if hasattr(context, "load_df"):
        df = context.load_df()
        if isinstance(df, pd.DataFrame):
            return df

    version = _find_active_data_version(context)

    if version is None:
        raise FileNotFoundError(
            "No active DataFrame data version is available. "
            "Upload a dataset or materialize a SQL query result first."
        )

    path = version.get("path")

    if not path:
        raise FileNotFoundError(
            f"Active data version `{version.get('version_id')}` has no path."
        )

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Active data file does not exist: {path}")

    suffix = path_obj.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path_obj)

    if suffix == ".csv":
        return pd.read_csv(path_obj)

    raise ValueError(f"Unsupported active data file type: {suffix}")


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


def _group_summary_rows(work: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []

    for group_value, values in work.groupby("group", dropna=True)["target"]:
        clean_values = values.dropna().astype(float)
        n = int(clean_values.shape[0])

        rows.append({
            "group": str(group_value),
            "n": n,
            "mean": _round_or_none(clean_values.mean()) if n else None,
            "std": _round_or_none(clean_values.std(ddof=1)) if n > 1 else None,
            "median": _round_or_none(clean_values.median()) if n else None,
            "min": _round_or_none(clean_values.min()) if n else None,
            "max": _round_or_none(clean_values.max()) if n else None,
        })

    return sorted(rows, key=lambda row: (row["mean"] is None, -(row["mean"] or 0)))


def _welch_degrees_of_freedom(x1: np.ndarray, x2: np.ndarray) -> float | None:
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


def _mean_difference_ci(x1: np.ndarray, x2: np.ndarray, alpha: float) -> tuple[float | None, float | None]:
    n1 = len(x1)
    n2 = len(x2)

    if n1 < 2 or n2 < 2:
        return None, None

    diff = float(np.mean(x1) - np.mean(x2))
    se = math.sqrt(float(np.var(x1, ddof=1)) / n1 + float(np.var(x2, ddof=1)) / n2)
    df = _welch_degrees_of_freedom(x1, x2)

    if df is None or se <= 0:
        return None, None

    t_crit = float(stats.t.ppf(1 - alpha / 2, df))

    return diff - t_crit * se, diff + t_crit * se


def _run_welch_t_test(groups: dict[str, np.ndarray], alpha: float) -> dict[str, Any]:
    labels = sorted(groups.keys())

    group1, group2 = labels[0], labels[1]
    x1, x2 = groups[group1], groups[group2]

    t_stat, p_value = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
    df = _welch_degrees_of_freedom(x1, x2)

    d, g = _cohens_d_and_hedges_g(x1, x2)
    ci_low, ci_high = _mean_difference_ci(x1, x2, alpha)

    mean_diff = float(np.mean(x1) - np.mean(x2))

    return {
        "method": "Welch independent two-sample t-test",
        "test_family": "two_group_numeric_comparison",
        "group1": group1,
        "group2": group2,
        "group1_mean": _round_or_none(np.mean(x1)),
        "group2_mean": _round_or_none(np.mean(x2)),
        "mean_difference_group1_minus_group2": _round_or_none(mean_diff),
        "mean_difference_ci_low": _round_or_none(ci_low),
        "mean_difference_ci_high": _round_or_none(ci_high),
        "t_statistic": _round_or_none(t_stat),
        "degrees_of_freedom": _round_or_none(df),
        "p_value": _round_or_none(p_value),
        "effect_size_name": "Hedges g",
        "effect_size": _round_or_none(g),
        "cohens_d": _round_or_none(d),
        "significant_at_alpha": bool(p_value < alpha) if math.isfinite(float(p_value)) else None,
    }


def _run_one_way_anova(groups: dict[str, np.ndarray], alpha: float) -> dict[str, Any]:
    arrays = list(groups.values())

    f_stat, p_value = stats.f_oneway(*arrays)

    all_values = np.concatenate(arrays)
    grand_mean = float(np.mean(all_values))

    ss_between = sum(len(x) * (float(np.mean(x)) - grand_mean) ** 2 for x in arrays)
    ss_within = sum(float(np.sum((x - float(np.mean(x))) ** 2)) for x in arrays)
    ss_total = ss_between + ss_within

    k = len(arrays)
    n = len(all_values)

    df_between = k - 1
    df_within = n - k

    ms_within = _safe_divide(ss_within, df_within)

    eta_squared = _safe_divide(ss_between, ss_total)

    omega_squared = None

    if ms_within is not None:
        omega_numerator = ss_between - df_between * ms_within
        omega_denominator = ss_total + ms_within
        omega_squared = _safe_divide(omega_numerator, omega_denominator)

        if omega_squared is not None:
            omega_squared = max(0.0, omega_squared)

    return {
        "method": "One-way ANOVA",
        "test_family": "multi_group_numeric_comparison",
        "F_statistic": _round_or_none(f_stat),
        "degrees_of_freedom_between": int(df_between),
        "degrees_of_freedom_within": int(df_within),
        "p_value": _round_or_none(p_value),
        "effect_size_name": "eta squared",
        "effect_size": _round_or_none(eta_squared),
        "eta_squared": _round_or_none(eta_squared),
        "omega_squared": _round_or_none(omega_squared),
        "significant_at_alpha": bool(p_value < alpha) if math.isfinite(float(p_value)) else None,
    }


def execute_statistical_group_comparison(context) -> Dict[str, Any]:
    arguments = _get_arguments(context)

    target_col = arguments.get("target_col")
    group_col = arguments.get("group_col")

    try:
        alpha = float(arguments.get("alpha", 0.05))
    except Exception:
        alpha = 0.05

    if not (0 < alpha < 1):
        alpha = 0.05

    try:
        df = _load_active_dataframe(context)

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "The active data source did not return a valid pandas DataFrame.",
            )

        df = _standardize_dataframe(df)

        if not target_col or not group_col:
            return _blocked(
                "MISSING_ARGUMENTS",
                "target_col and group_col are required for statistical_group_comparison.",
                details={
                    "target_col": target_col,
                    "group_col": group_col,
                },
                suggested_next_actions=[
                    "Specify a numeric outcome column as target_col and a categorical grouping column as group_col."
                ],
            )

        missing_cols = [col for col in [target_col, group_col] if col not in df.columns]

        if missing_cols:
            return _blocked(
                "COLUMN_NOT_FOUND",
                "One or more requested columns do not exist in the active dataset.",
                details={
                    "missing_columns": missing_cols,
                    "available_columns": list(df.columns),
                    "received_arguments": arguments,
                },
            )

        target = pd.to_numeric(df[target_col], errors="coerce")

        if target.notna().sum() == 0:
            return _blocked(
                "TARGET_NOT_NUMERIC",
                "target_col must contain numeric values for group comparison.",
                details={
                    "target_col": target_col,
                    "dtype": str(df[target_col].dtype),
                },
            )

        work = pd.DataFrame({
            "target": target,
            "group": df[group_col],
        }).dropna()

        if work.empty:
            return _blocked(
                "NO_COMPLETE_CASES",
                "No complete cases are available for the selected target and group columns.",
                details={
                    "target_col": target_col,
                    "group_col": group_col,
                },
            )

        group_summaries = _group_summary_rows(work)

        groups: dict[str, np.ndarray] = {}
        insufficient_groups = []

        for group_value, values in work.groupby("group", dropna=True)["target"]:
            clean = values.dropna().astype(float).to_numpy(dtype=float)

            if len(clean) >= 2:
                groups[str(group_value)] = clean
            else:
                insufficient_groups.append(str(group_value))

        if len(groups) < 2:
            return _blocked(
                "INSUFFICIENT_GROUPS",
                "Group comparison requires at least two groups with at least two valid numeric observations each.",
                details={
                    "target_col": target_col,
                    "group_col": group_col,
                    "group_summaries": group_summaries,
                    "insufficient_groups": insufficient_groups,
                },
            )

        if len(groups) == 2:
            test_details = _run_welch_t_test(groups, alpha)
        else:
            test_details = _run_one_way_anova(groups, alpha)

        valid_group_summaries = [
            row for row in group_summaries
            if row["group"] in groups
        ]

        top = max(valid_group_summaries, key=lambda row: row["mean"])
        bottom = min(valid_group_summaries, key=lambda row: row["mean"])

        top_bottom_diff = (
            top["mean"] - bottom["mean"]
            if top["mean"] is not None and bottom["mean"] is not None
            else None
        )

        relative_lift = (
            _safe_divide(top_bottom_diff, abs(bottom["mean"]))
            if top_bottom_diff is not None
            else None
        )

        assumptions_and_limitations = [
            "This is an observational comparison unless the data came from a randomized experiment.",
            "The test assumes independent observations within and across groups.",
            "The numeric outcome is compared across observed groups after dropping missing target/group values.",
        ]

        if len(groups) == 2:
            assumptions_and_limitations.append(
                "Welch's t-test does not assume equal variances, but very small or highly skewed groups still require caution."
            )
        else:
            assumptions_and_limitations.append(
                "One-way ANOVA compares group means; if significant, follow-up post-hoc comparisons are needed to identify which pairs differ."
            )

        details = {
            "target_col": target_col,
            "group_col": group_col,
            "alpha": alpha,
            "nobs": int(sum(len(x) for x in groups.values())),
            "valid_group_count": int(len(groups)),
            "excluded_group_count": int(len(insufficient_groups)),
            "excluded_groups": insufficient_groups,
            "top_group": top["group"],
            "top_group_mean": top["mean"],
            "lowest_group": bottom["group"],
            "lowest_group_mean": bottom["mean"],
            "top_minus_lowest_mean_difference": _round_or_none(top_bottom_diff),
            "top_vs_lowest_relative_lift": _round_or_none(relative_lift),
            "group_summaries": group_summaries,
            "assumptions_and_limitations": assumptions_and_limitations,
            **test_details,
        }

        return {
            "status": "ok",
            "message": f"{test_details['method']} completed for {target_col} by {group_col}.",
            "recoverable": False,
            "details": details,
            "artifacts": [],
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_code": "STATISTICAL_GROUP_COMPARISON_FAILED",
            "message": f"statistical_group_comparison failed: {exc}",
            "recoverable": True,
            "details": {
                "exception_type": type(exc).__name__,
                "error_message": str(exc),
                "received_arguments": arguments,
            },
            "artifacts": [],
        }


def extract_statistical_group_comparison(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target_col = payload.get("target_col") or arguments.get("target_col")
    group_col = payload.get("group_col") or arguments.get("group_col")
    method = payload.get("method") or "Statistical group comparison"

    title = "Statistical Group Comparison"

    if target_col and group_col:
        title = f"Statistical Group Comparison: {target_col} by {group_col}"

    significant = payload.get("significant_at_alpha")

    significance_phrase = ""

    if significant is True:
        significance_phrase = " The group difference is statistically significant at the selected alpha level."
    elif significant is False:
        significance_phrase = " The group difference is not statistically significant at the selected alpha level."

    summary = (
        f"Used {method} to compare numeric outcome `{target_col}` across groups in `{group_col}`."
        f" Top group by mean: `{payload.get('top_group')}`. Lowest group by mean: `{payload.get('lowest_group')}`."
        f"{significance_phrase}"
    )

    metrics = compact_dict({
        "method": method,
        "nobs": payload.get("nobs"),
        "valid_group_count": payload.get("valid_group_count"),
        "alpha": payload.get("alpha"),
        "p_value": payload.get("p_value"),
        "significant_at_alpha": significant,
        "effect_size_name": payload.get("effect_size_name"),
        "effect_size": payload.get("effect_size"),
        "top_group": payload.get("top_group"),
        "top_group_mean": payload.get("top_group_mean"),
        "lowest_group": payload.get("lowest_group"),
        "lowest_group_mean": payload.get("lowest_group_mean"),
        "top_minus_lowest_mean_difference": payload.get("top_minus_lowest_mean_difference"),
        "top_vs_lowest_relative_lift": payload.get("top_vs_lowest_relative_lift"),
        "F_statistic": payload.get("F_statistic"),
        "t_statistic": payload.get("t_statistic"),
        "degrees_of_freedom": payload.get("degrees_of_freedom"),
        "degrees_of_freedom_between": payload.get("degrees_of_freedom_between"),
        "degrees_of_freedom_within": payload.get("degrees_of_freedom_within"),
    })

    tables: Dict[str, Any] = {}

    if payload.get("group_summaries"):
        tables["group_summaries"] = payload.get("group_summaries")

    if payload.get("assumptions_and_limitations"):
        tables["assumptions_and_limitations"] = [
            {"item": item}
            for item in payload.get("assumptions_and_limitations", [])
        ]

    metadata = compact_dict({
        "target_col": target_col,
        "group_col": group_col,
        "test_family": payload.get("test_family"),
        "excluded_groups": payload.get("excluded_groups"),
    })

    return title, summary, metrics, tables, metadata


STATISTICAL_GROUP_COMPARISON_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "nobs": "Observations",
            "valid_group_count": "Valid groups",
            "alpha": "Alpha",
            "p_value": "p-value",
            "significant_at_alpha": "Significant",
            "effect_size_name": "Effect size",
            "effect_size": "Effect size value",
            "top_group": "Top group",
            "top_group_mean": "Top group mean",
            "lowest_group": "Lowest group",
            "lowest_group_mean": "Lowest group mean",
            "top_minus_lowest_mean_difference": "Top - lowest mean difference",
            "top_vs_lowest_relative_lift": "Relative lift",
            "F_statistic": "F statistic",
            "t_statistic": "t statistic",
            "degrees_of_freedom": "Degrees of freedom",
            "degrees_of_freedom_between": "DF between",
            "degrees_of_freedom_within": "DF within",
        },
        formatters={
            "p_value": format_p_value,
            "significant_at_alpha": format_bool_yes_no,
            "effect_size": format_number,
            "top_group_mean": format_number,
            "lowest_group_mean": format_number,
            "top_minus_lowest_mean_difference": format_number,
            "top_vs_lowest_relative_lift": format_number,
            "F_statistic": format_number,
            "t_statistic": format_number,
            "degrees_of_freedom": format_number,
        },
        order=[
            "method",
            "nobs",
            "valid_group_count",
            "alpha",
            "p_value",
            "significant_at_alpha",
            "effect_size_name",
            "effect_size",
            "top_group",
            "top_group_mean",
            "lowest_group",
            "lowest_group_mean",
            "top_minus_lowest_mean_difference",
            "top_vs_lowest_relative_lift",
            "F_statistic",
            "t_statistic",
            "degrees_of_freedom",
            "degrees_of_freedom_between",
            "degrees_of_freedom_within",
        ],
    ),
    tables={
        "group_summaries": TableDisplayConfig(
            column_labels={
                "group": "Group",
                "n": "N",
                "mean": "Mean",
                "std": "Std. dev.",
                "median": "Median",
                "min": "Min",
                "max": "Max",
            },
            column_order=[
                "group",
                "n",
                "mean",
                "std",
                "median",
                "min",
                "max",
            ],
            column_formatters={
                "mean": format_number,
                "std": format_number,
                "median": format_number,
                "min": format_number,
                "max": format_number,
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
    tool_name="statistical_group_comparison",
    display_name="Statistical Group Comparison",
    evidence_categories=["group_comparison", "statistical_inference"],
    description=(
        "Compare a numeric outcome across a categorical grouping variable using Welch's t-test "
        "for two groups or one-way ANOVA for three or more groups."
    ),
    usage_guidance=(
        "Use this when the user asks whether a numeric metric differs across groups, segments, regions, "
        "classes, treatments, or categories. This is a statistical comparison tool, not just a grouped summary. "
        "It reports p-values, effect sizes, group means, and assumptions/limitations. "
        "For inferential statistics, the active dataset must be observation-level, not one row per group. "
        "If the source is SQL, first materialize the natural observational unit, such as customer-level, "
        "order-level, subject-level, student-level, patient-level, transaction-level, or experimental-unit-level data. "
        "For example, to test whether total_revenue differs by region, use one row per customer or order with "
        "both region and total_revenue, not SELECT region, SUM(revenue) GROUP BY region."
    ),
    use_when=[
        "The user asks if a numeric outcome differs by a categorical group.",
        "The user asks whether revenue, sales, GPA, score, cost, or another numeric metric differs across segments, regions, treatments, or categories.",
        "The active DataFrame has a numeric target column and a categorical grouping column.",
    ],
    do_not_use_when=[
        "No active DataFrame dataset exists.",
        "The user only wants a descriptive grouped table; use groupby_summary instead.",
        "The target variable is categorical; use chi_square for two categorical variables.",
        "The user wants a multivariable model; use run_multiple_regression or a regression tool instead.",
        "The active dataset has already been aggregated to one row per group; materialize an observation-level dataset first.",
    ],
    requires_data_source="dataframe",
    produces_active_dataset=False,
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "group_col": str,
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
    execute=execute_statistical_group_comparison,
    extractor=extract_statistical_group_comparison,
    guardrail_evaluators=[],
    display_config=STATISTICAL_GROUP_COMPARISON_DISPLAY,
    examples=[
        {
            "user_request": "Does total revenue differ by region?",
            "arguments": {
                "target_col": "total_revenue",
                "group_col": "region",
            },
        },
        {
            "user_request": "Compare GPA across sex groups statistically.",
            "arguments": {
                "target_col": "GPA",
                "group_col": "Sex",
            },
        },
        {
            "user_request": "Is average revenue different across customer segments?",
            "arguments": {
                "target_col": "total_revenue",
                "group_col": "segment",
                "alpha": 0.05,
            },
        },
    ],
))