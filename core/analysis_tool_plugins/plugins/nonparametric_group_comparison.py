from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import math
import itertools

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
from core.analysis_tool_plugins.shared.apa_writers import write_apa_nonparametric_group_comparison


MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    "NA", "N/A", "NaN", "NULL", "None", "Missing", "Unknown",
}


# ==========================================================
# Helpers (mirrored from statistical_group_comparison.py)
# ==========================================================

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


# ==========================================================
# Effect-size and magnitude helpers
# ==========================================================

def _rank_biserial_correlation(
    u_statistic: float,
    n1: int,
    n2: int,
) -> float | None:
    """
    Rank-biserial correlation as an effect size for Mann-Whitney U.

    r_rb = 1 - 2U / (n1 * n2)

    Ranges in [-1, 1]. Interpreted with the same magnitude conventions as
    Pearson r.
    """
    if n1 < 1 or n2 < 1:
        return None

    return 1.0 - (2.0 * float(u_statistic)) / (n1 * n2)


def _interpret_rank_biserial(r: float | None) -> str | None:
    if r is None:
        return None

    a = abs(float(r))

    if a < 0.10:
        return "negligible"
    if a < 0.30:
        return "small"
    if a < 0.50:
        return "moderate"

    return "large"


def _epsilon_squared(h_statistic: float, n_total: int) -> float | None:
    """
    Epsilon-squared effect size for Kruskal-Wallis H.

    eps^2 = H / (n - 1)

    Bounded in [0, 1]. Magnitude conventions parallel eta-squared.
    """
    if n_total <= 1:
        return None

    value = float(h_statistic) / (n_total - 1)

    if not math.isfinite(value):
        return None

    return max(0.0, min(1.0, value))


def _interpret_epsilon_squared(eps2: float | None) -> str | None:
    if eps2 is None:
        return None

    a = float(eps2)

    if a < 0.01:
        return "negligible"
    if a < 0.06:
        return "small"
    if a < 0.14:
        return "medium"

    return "large"


def _hodges_lehmann_estimate_and_ci(
    x1: np.ndarray,
    x2: np.ndarray,
    alpha: float,
) -> tuple[float | None, float | None, float | None]:
    """
    Hodges-Lehmann estimator of the location shift between two groups, plus a
    distribution-free confidence interval based on the Mann-Whitney rank sum.

    HL = median of all pairwise differences x1_i - x2_j.

    CI: with K = n1 * n2 ordered pairwise differences D_(1) <= ... <= D_(K),
    the (1 - alpha) CI uses the (k_low)-th and (K - k_low + 1)-th order
    statistics, where k_low = floor(K / 2 - z * sqrt(n1 * n2 * (n1 + n2 + 1) / 12)).
    """
    n1 = int(len(x1))
    n2 = int(len(x2))

    if n1 < 1 or n2 < 1:
        return None, None, None

    # All pairwise differences.
    diffs = np.subtract.outer(x1, x2).ravel()
    diffs = diffs[np.isfinite(diffs)]

    if diffs.size == 0:
        return None, None, None

    hl = float(np.median(diffs))

    K = diffs.size

    if K < 4:
        # Too few pairs for a meaningful CI; return the point estimate only.
        return hl, None, None

    diffs_sorted = np.sort(diffs)

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    spread = z_crit * math.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)

    k_low = int(math.floor(K / 2.0 - spread))

    # Clamp into the valid index range.
    if k_low < 1:
        k_low = 1
    if k_low > K:
        k_low = K

    k_high = K - k_low + 1

    ci_low = float(diffs_sorted[k_low - 1])
    ci_high = float(diffs_sorted[k_high - 1])

    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low

    return hl, ci_low, ci_high


# ==========================================================
# Dunn's post-hoc with multiple-comparison correction
# ==========================================================

def _bh_fdr_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg step-up FDR adjustment."""
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.size

    if m == 0:
        return p_values

    order = np.argsort(p_values)
    ranked = p_values[order]

    raw = ranked * m / np.arange(1, m + 1)
    # Enforce monotonicity (step-up).
    adjusted_sorted = np.minimum.accumulate(raw[::-1])[::-1]

    result = np.empty(m, dtype=float)
    result[order] = np.clip(adjusted_sorted, 0.0, 1.0)

    return result


def _bonferroni_adjust(p_values: np.ndarray) -> np.ndarray:
    p_values = np.asarray(p_values, dtype=float)
    return np.clip(p_values * p_values.size, 0.0, 1.0)


def _holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm step-down adjustment."""
    p_values = np.asarray(p_values, dtype=float)
    m = p_values.size

    if m == 0:
        return p_values

    order = np.argsort(p_values)
    ranked = p_values[order]

    raw = ranked * (m - np.arange(m))
    adjusted_sorted = np.maximum.accumulate(raw)

    result = np.empty(m, dtype=float)
    result[order] = np.clip(adjusted_sorted, 0.0, 1.0)

    return result


def _adjust_p_values(p_values: np.ndarray, method: str) -> np.ndarray:
    method = (method or "bh").lower().strip()

    if method in {"bh", "fdr", "fdr_bh", "benjamini-hochberg"}:
        return _bh_fdr_adjust(p_values)

    if method in {"bonferroni", "bonf"}:
        return _bonferroni_adjust(p_values)

    if method in {"holm", "holm-bonferroni"}:
        return _holm_adjust(p_values)

    return p_values


def _run_dunn_test(
    groups: dict[str, np.ndarray],
    alpha: float,
    adjustment: str = "bh",
) -> tuple[list[dict[str, Any]], str]:
    """
    Dunn's post-hoc test following Kruskal-Wallis.

    For each pair of groups (i, j):
      z_ij = (R_i_bar - R_j_bar) / sqrt(sigma2 * (1/n_i + 1/n_j))

    where R_i_bar is the mean rank in group i across the pooled ranking, and
    sigma2 = N(N+1)/12 * (1 - sum(t^3 - t) / (N^3 - N)) is the variance under
    H0 with tie correction.

    Returns the raw p-values along with adjusted p-values controlling either
    FWER (Bonferroni, Holm) or FDR (Benjamini-Hochberg).
    """
    labels = sorted(groups.keys())
    k = len(labels)

    if k < 2:
        return [], "Dunn's test (no pairs)"

    # Pool values with group membership labels.
    all_values_parts = []
    membership = []

    for label in labels:
        arr = groups[label]
        all_values_parts.append(arr)
        membership.extend([label] * len(arr))

    all_values = np.concatenate(all_values_parts)
    membership = np.asarray(membership)
    n_total = int(all_values.size)

    if n_total < 2:
        return [], "Dunn's test (insufficient observations)"

    ranks = stats.rankdata(all_values, method="average")

    # Per-group mean rank and size.
    mean_ranks = {}
    sizes = {}

    for label in labels:
        mask = membership == label
        sizes[label] = int(mask.sum())
        if sizes[label] > 0:
            mean_ranks[label] = float(ranks[mask].mean())
        else:
            mean_ranks[label] = float("nan")

    # Tie correction term.
    _, counts = np.unique(all_values, return_counts=True)
    tied_counts = counts[counts > 1].astype(float)
    tie_sum = float(np.sum(tied_counts ** 3 - tied_counts))

    denom = n_total ** 3 - n_total
    tie_correction = 1.0 - (tie_sum / denom) if denom > 0 else 1.0

    sigma2 = (n_total * (n_total + 1) / 12.0) * tie_correction

    pairs = list(itertools.combinations(labels, 2))

    raw_p_values = []
    pair_info = []

    for g1, g2 in pairs:
        n1, n2 = sizes[g1], sizes[g2]

        if n1 < 1 or n2 < 1 or sigma2 <= 0:
            raw_p_values.append(float("nan"))
            pair_info.append({
                "group1": str(g1),
                "group2": str(g2),
                "n1": n1,
                "n2": n2,
                "mean_rank_1": mean_ranks[g1],
                "mean_rank_2": mean_ranks[g2],
                "mean_rank_difference": None,
                "z_statistic": None,
            })
            continue

        se = math.sqrt(sigma2 * (1.0 / n1 + 1.0 / n2))

        if se == 0:
            raw_p_values.append(float("nan"))
            pair_info.append({
                "group1": str(g1),
                "group2": str(g2),
                "n1": n1,
                "n2": n2,
                "mean_rank_1": mean_ranks[g1],
                "mean_rank_2": mean_ranks[g2],
                "mean_rank_difference": float(mean_ranks[g1] - mean_ranks[g2]),
                "z_statistic": None,
            })
            continue

        z = (mean_ranks[g1] - mean_ranks[g2]) / se
        p_two_sided = 2.0 * (1.0 - stats.norm.cdf(abs(z)))

        raw_p_values.append(p_two_sided)
        pair_info.append({
            "group1": str(g1),
            "group2": str(g2),
            "n1": n1,
            "n2": n2,
            "mean_rank_1": mean_ranks[g1],
            "mean_rank_2": mean_ranks[g2],
            "mean_rank_difference": float(mean_ranks[g1] - mean_ranks[g2]),
            "z_statistic": float(z),
        })

    raw_p_array = np.asarray(raw_p_values, dtype=float)

    finite_mask = np.isfinite(raw_p_array)

    adjusted = np.full_like(raw_p_array, np.nan)

    if finite_mask.any():
        adjusted_finite = _adjust_p_values(raw_p_array[finite_mask], adjustment)
        adjusted[finite_mask] = adjusted_finite

    method_label_map = {
        "bh": "Dunn's test (Benjamini-Hochberg FDR)",
        "fdr": "Dunn's test (Benjamini-Hochberg FDR)",
        "fdr_bh": "Dunn's test (Benjamini-Hochberg FDR)",
        "benjamini-hochberg": "Dunn's test (Benjamini-Hochberg FDR)",
        "bonferroni": "Dunn's test (Bonferroni FWER)",
        "bonf": "Dunn's test (Bonferroni FWER)",
        "holm": "Dunn's test (Holm-Bonferroni FWER)",
        "holm-bonferroni": "Dunn's test (Holm-Bonferroni FWER)",
    }
    method_label = method_label_map.get(
        (adjustment or "bh").lower().strip(),
        "Dunn's test",
    )

    rows = []

    for info, p_raw, p_adj in zip(pair_info, raw_p_array, adjusted):
        sig = None
        if math.isfinite(p_adj):
            sig = bool(p_adj < alpha)

        rows.append({
            "group1": info["group1"],
            "group2": info["group2"],
            "mean_rank_difference": _round_or_none(info["mean_rank_difference"]),
            "z_statistic": _round_or_none(info["z_statistic"]),
            "p_value_raw": _round_or_none(p_raw) if math.isfinite(p_raw) else None,
            "p_value_adjusted": _round_or_none(p_adj) if math.isfinite(p_adj) else None,
            "significant_at_alpha": sig,
            "adjustment_method": method_label,
        })

    return rows, method_label


# ==========================================================
# Per-group descriptive summary (median + IQR primary)
# ==========================================================

def _group_summary_rows(work: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []

    for group_value, values in work.groupby("group", dropna=True)["target"]:
        clean = values.dropna().astype(float)
        n = int(clean.shape[0])

        if n == 0:
            continue

        q1 = float(clean.quantile(0.25)) if n >= 1 else None
        q3 = float(clean.quantile(0.75)) if n >= 1 else None

        rows.append({
            "group": str(group_value),
            "n": n,
            "median": _round_or_none(clean.median()),
            "q1": _round_or_none(q1),
            "q3": _round_or_none(q3),
            "iqr": _round_or_none((q3 - q1) if (q1 is not None and q3 is not None) else None),
            "mean": _round_or_none(clean.mean()),
            "min": _round_or_none(clean.min()),
            "max": _round_or_none(clean.max()),
        })

    return sorted(rows, key=lambda row: (row["median"] is None, -(row["median"] or 0)))


# ==========================================================
# Main test routines
# ==========================================================

def _run_mann_whitney(
    groups: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, Any]:
    labels = sorted(groups.keys())
    group1, group2 = labels[0], labels[1]
    x1, x2 = groups[group1], groups[group2]

    n1, n2 = len(x1), len(x2)

    # Two-sided Mann-Whitney U with tie correction. scipy returns U for the
    # first sample (x1).
    try:
        result = stats.mannwhitneyu(
            x1, x2,
            alternative="two-sided",
            method="auto",
        )
        u_statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception:
        u_statistic = float("nan")
        p_value = float("nan")

    rb = _rank_biserial_correlation(u_statistic, n1, n2) if math.isfinite(u_statistic) else None
    rb_magnitude = _interpret_rank_biserial(rb)

    hl, hl_low, hl_high = _hodges_lehmann_estimate_and_ci(x1, x2, alpha)

    # Detect ties (informational; scipy does tie correction by default).
    combined = np.concatenate([x1, x2])
    _, counts = np.unique(combined, return_counts=True)
    has_ties = bool(np.any(counts > 1))

    return {
        "method": "Mann-Whitney U test (two-sided)",
        "test_family": "two_group_numeric_comparison_nonparametric",
        "group1": str(group1),
        "group2": str(group2),
        "group1_n": int(n1),
        "group2_n": int(n2),
        "group1_median": _round_or_none(np.median(x1)),
        "group2_median": _round_or_none(np.median(x2)),
        "hodges_lehmann_location_shift": _round_or_none(hl),
        "hodges_lehmann_ci_low": _round_or_none(hl_low),
        "hodges_lehmann_ci_high": _round_or_none(hl_high),
        "U_statistic": _round_or_none(u_statistic),
        "p_value": _round_or_none(p_value),
        "effect_size_name": "rank-biserial correlation",
        "effect_size": _round_or_none(rb),
        "effect_size_magnitude": rb_magnitude,
        "tie_correction_applied": has_ties,
        "significant_at_alpha": (
            bool(p_value < alpha) if math.isfinite(float(p_value)) else None
        ),
        "significant_at_0_05": (
            bool(p_value < 0.05) if math.isfinite(float(p_value)) else None
        ),
    }


def _run_kruskal_wallis(
    groups: dict[str, np.ndarray],
    alpha: float,
) -> dict[str, Any]:
    arrays = [groups[label] for label in sorted(groups.keys())]
    n_total = int(sum(len(a) for a in arrays))

    try:
        h_statistic, p_value = stats.kruskal(*arrays)
        h_statistic = float(h_statistic)
        p_value = float(p_value)
    except Exception:
        h_statistic = float("nan")
        p_value = float("nan")

    eps2 = _epsilon_squared(h_statistic, n_total) if math.isfinite(h_statistic) else None
    eps2_magnitude = _interpret_epsilon_squared(eps2)

    return {
        "method": "Kruskal-Wallis H test",
        "test_family": "multi_group_numeric_comparison_nonparametric",
        "H_statistic": _round_or_none(h_statistic),
        "degrees_of_freedom_between": int(len(arrays) - 1),
        "p_value": _round_or_none(p_value),
        "effect_size_name": "epsilon squared",
        "effect_size": _round_or_none(eps2),
        "effect_size_magnitude": eps2_magnitude,
        "epsilon_squared": _round_or_none(eps2),
        "significant_at_alpha": (
            bool(p_value < alpha) if math.isfinite(float(p_value)) else None
        ),
        "significant_at_0_05": (
            bool(p_value < 0.05) if math.isfinite(float(p_value)) else None
        ),
    }


# ==========================================================
# Main execute
# ==========================================================

def execute_nonparametric_group_comparison(context) -> Dict[str, Any]:
    arguments = _get_arguments(context)

    target_col = arguments.get("target_col")
    group_col = arguments.get("group_col")

    try:
        alpha = float(arguments.get("alpha", 0.05))
    except Exception:
        alpha = 0.05

    if not (0 < alpha < 1):
        alpha = 0.05

    post_hoc_adjustment = str(arguments.get("post_hoc_adjustment", "bh")).lower().strip()

    if post_hoc_adjustment not in {
        "bh", "fdr", "fdr_bh", "benjamini-hochberg",
        "bonferroni", "bonf",
        "holm", "holm-bonferroni",
    }:
        post_hoc_adjustment = "bh"

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
                "target_col and group_col are required for nonparametric_group_comparison.",
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
                "target_col must contain numeric values for nonparametric group comparison.",
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
                "Nonparametric group comparison requires at least two groups with at least two valid observations each.",
                details={
                    "target_col": target_col,
                    "group_col": group_col,
                    "group_summaries": group_summaries,
                    "insufficient_groups": insufficient_groups,
                },
            )

        # ------------------------------------------------------
        # Pick test
        # ------------------------------------------------------
        post_hoc_rows: list[dict[str, Any]] = []
        post_hoc_method = None

        if len(groups) == 2:
            test_details = _run_mann_whitney(groups, alpha)
        else:
            test_details = _run_kruskal_wallis(groups, alpha)

            if test_details.get("significant_at_alpha"):
                post_hoc_rows, post_hoc_method = _run_dunn_test(
                    groups,
                    alpha,
                    adjustment=post_hoc_adjustment,
                )

        valid_group_summaries = [row for row in group_summaries if row["group"] in groups]

        top = max(valid_group_summaries, key=lambda row: row["median"])
        bottom = min(valid_group_summaries, key=lambda row: row["median"])

        top_bottom_diff = (
            top["median"] - bottom["median"]
            if top["median"] is not None and bottom["median"] is not None
            else None
        )

        # ------------------------------------------------------
        # Assumptions / limitations
        # ------------------------------------------------------
        assumptions_and_limitations = [
            "Nonparametric rank-based tests assume independent observations within and across groups.",
            "Mann-Whitney U and Kruskal-Wallis test stochastic dominance; they reduce to tests of medians only when the group distributions have similar shape and spread.",
            "These tests are robust to non-normality and to outliers in the outcome.",
        ]

        if len(groups) == 2:
            assumptions_and_limitations.append(
                "The Hodges-Lehmann estimator and its distribution-free CI describe the median shift between the two groups."
            )

            if test_details.get("tie_correction_applied"):
                assumptions_and_limitations.append(
                    "Ties were present; scipy's Mann-Whitney implementation applies the standard tie correction."
                )
        else:
            if test_details.get("significant_at_alpha"):
                assumptions_and_limitations.append(
                    f"Significant Kruskal-Wallis result; pairwise comparisons reported using {post_hoc_method}."
                )
            else:
                assumptions_and_limitations.append(
                    "Kruskal-Wallis was not significant; no pairwise post-hoc comparisons were performed."
                )

        # ------------------------------------------------------
        # Compose result
        # ------------------------------------------------------
        details = {
            "target_col": target_col,
            "group_col": group_col,
            "alpha": alpha,
            "nobs": int(sum(len(x) for x in groups.values())),
            "valid_group_count": int(len(groups)),
            "excluded_group_count": int(len(insufficient_groups)),
            "excluded_groups": insufficient_groups,
            "top_group": top["group"],
            "top_group_median": top["median"],
            "lowest_group": bottom["group"],
            "lowest_group_median": bottom["median"],
            "top_minus_lowest_median_difference": _round_or_none(top_bottom_diff),
            "group_summaries": group_summaries,
            "post_hoc_method": post_hoc_method,
            "post_hoc_pairwise": post_hoc_rows,
            "post_hoc_adjustment_requested": post_hoc_adjustment,
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
            "error_code": "NONPARAMETRIC_GROUP_COMPARISON_FAILED",
            "message": f"nonparametric_group_comparison failed: {exc}",
            "recoverable": True,
            "details": {
                "exception_type": type(exc).__name__,
                "error_message": str(exc),
                "received_arguments": arguments,
            },
            "artifacts": [],
        }


# ==========================================================
# Extractor / Display
# ==========================================================

def extract_nonparametric_group_comparison(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target_col = payload.get("target_col") or arguments.get("target_col")
    group_col = payload.get("group_col") or arguments.get("group_col")
    method = payload.get("method") or "Nonparametric group comparison"

    title = "Nonparametric Group Comparison"

    if target_col and group_col:
        title = f"Nonparametric Group Comparison: {target_col} by {group_col}"

    significant = payload.get("significant_at_alpha")

    significance_phrase = ""

    if significant is True:
        significance_phrase = " The group difference is statistically significant at the selected alpha level."
    elif significant is False:
        significance_phrase = " The group difference is not statistically significant at the selected alpha level."

    summary = (
        f"Used {method} to compare numeric outcome `{target_col}` across groups in `{group_col}`."
        f" Top group by median: `{payload.get('top_group')}`. Lowest group by median: `{payload.get('lowest_group')}`."
        f"{significance_phrase}"
    )

    if payload.get("post_hoc_pairwise"):
        n_sig_pairs = sum(
            1 for row in payload.get("post_hoc_pairwise", [])
            if row.get("significant_at_alpha")
        )
        summary += f" Post-hoc ({payload.get('post_hoc_method')}): {n_sig_pairs} significant pair(s)."

    metrics = compact_dict({
        "method": method,
        "nobs": payload.get("nobs"),
        "valid_group_count": payload.get("valid_group_count"),
        "alpha": payload.get("alpha"),
        "p_value": payload.get("p_value"),
        "significant_at_alpha": significant,
        "significant_at_0_05": payload.get("significant_at_0_05"),
        "effect_size_name": payload.get("effect_size_name"),
        "effect_size": payload.get("effect_size"),
        "effect_size_magnitude": payload.get("effect_size_magnitude"),
        "top_group": payload.get("top_group"),
        "top_group_median": payload.get("top_group_median"),
        "lowest_group": payload.get("lowest_group"),
        "lowest_group_median": payload.get("lowest_group_median"),
        "top_minus_lowest_median_difference": payload.get("top_minus_lowest_median_difference"),
        "U_statistic": payload.get("U_statistic"),
        "H_statistic": payload.get("H_statistic"),
        "degrees_of_freedom_between": payload.get("degrees_of_freedom_between"),
        "hodges_lehmann_location_shift": payload.get("hodges_lehmann_location_shift"),
        "hodges_lehmann_ci_low": payload.get("hodges_lehmann_ci_low"),
        "hodges_lehmann_ci_high": payload.get("hodges_lehmann_ci_high"),
        "epsilon_squared": payload.get("epsilon_squared"),
    })

    tables: Dict[str, Any] = {}

    if payload.get("group_summaries"):
        tables["group_summaries"] = payload.get("group_summaries")

    if payload.get("post_hoc_pairwise"):
        tables["post_hoc_pairwise"] = payload.get("post_hoc_pairwise")

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
        "post_hoc_method": payload.get("post_hoc_method"),
        "post_hoc_adjustment_requested": payload.get("post_hoc_adjustment_requested"),
        "tie_correction_applied": payload.get("tie_correction_applied"),
    })

    return title, summary, metrics, tables, metadata


NONPARAMETRIC_GROUP_COMPARISON_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "nobs": "Observations",
            "valid_group_count": "Valid groups",
            "alpha": "Alpha",
            "p_value": "p-value",
            "significant_at_alpha": "Significant",
            "significant_at_0_05": "Significant at 0.05",
            "effect_size_name": "Effect size",
            "effect_size": "Effect size value",
            "effect_size_magnitude": "Effect size magnitude",
            "top_group": "Top group (by median)",
            "top_group_median": "Top group median",
            "lowest_group": "Lowest group (by median)",
            "lowest_group_median": "Lowest group median",
            "top_minus_lowest_median_difference": "Top - lowest median difference",
            "U_statistic": "Mann-Whitney U",
            "H_statistic": "Kruskal-Wallis H",
            "degrees_of_freedom_between": "Degrees of freedom",
            "hodges_lehmann_location_shift": "Hodges-Lehmann shift",
            "hodges_lehmann_ci_low": "HL CI lower",
            "hodges_lehmann_ci_high": "HL CI upper",
            "epsilon_squared": "Epsilon squared",
        },
        formatters={
            "p_value": format_p_value,
            "significant_at_alpha": format_bool_yes_no,
            "significant_at_0_05": format_bool_yes_no,
            "effect_size": format_number,
            "top_group_median": format_number,
            "lowest_group_median": format_number,
            "top_minus_lowest_median_difference": format_number,
            "U_statistic": format_number,
            "H_statistic": format_number,
            "hodges_lehmann_location_shift": format_number,
            "hodges_lehmann_ci_low": format_number,
            "hodges_lehmann_ci_high": format_number,
            "epsilon_squared": format_number,
        },
        order=[
            "method",
            "nobs",
            "valid_group_count",
            "alpha",
            "p_value",
            "significant_at_alpha",
            "significant_at_0_05",
            "effect_size_name",
            "effect_size",
            "effect_size_magnitude",
            "top_group",
            "top_group_median",
            "lowest_group",
            "lowest_group_median",
            "top_minus_lowest_median_difference",
            "U_statistic",
            "H_statistic",
            "degrees_of_freedom_between",
            "hodges_lehmann_location_shift",
            "hodges_lehmann_ci_low",
            "hodges_lehmann_ci_high",
            "epsilon_squared",
        ],
    ),
    tables={
        "group_summaries": TableDisplayConfig(
            column_labels={
                "group": "Group",
                "n": "N",
                "median": "Median",
                "q1": "Q1",
                "q3": "Q3",
                "iqr": "IQR",
                "mean": "Mean",
                "min": "Min",
                "max": "Max",
            },
            column_order=[
                "group",
                "n",
                "median",
                "q1",
                "q3",
                "iqr",
                "mean",
                "min",
                "max",
            ],
            column_formatters={
                "median": format_number,
                "q1": format_number,
                "q3": format_number,
                "iqr": format_number,
                "mean": format_number,
                "min": format_number,
                "max": format_number,
            },
        ),
        "post_hoc_pairwise": TableDisplayConfig(
            column_labels={
                "group1": "Group 1",
                "group2": "Group 2",
                "mean_rank_difference": "Mean rank diff (g1 - g2)",
                "z_statistic": "z statistic",
                "p_value_raw": "Raw p-value",
                "p_value_adjusted": "Adjusted p-value",
                "significant_at_alpha": "Significant",
                "adjustment_method": "Adjustment",
            },
            column_order=[
                "group1",
                "group2",
                "mean_rank_difference",
                "z_statistic",
                "p_value_raw",
                "p_value_adjusted",
                "significant_at_alpha",
                "adjustment_method",
            ],
            column_formatters={
                "mean_rank_difference": format_number,
                "z_statistic": format_number,
                "p_value_raw": format_p_value,
                "p_value_adjusted": format_p_value,
                "significant_at_alpha": format_bool_yes_no,
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
    tool_name="nonparametric_group_comparison",
    display_name="Nonparametric Group Comparison",
    is_inferential=True,
    evidence_categories=["group_comparison", "statistical_inference"],
    description=(
        "Rank-based group comparison without normality assumptions. "
        "Two groups: Mann-Whitney U with rank-biserial correlation and "
        "Hodges-Lehmann location-shift estimate plus distribution-free CI. "
        "Three or more groups: Kruskal-Wallis H test with epsilon-squared "
        "effect size; when significant, Dunn's pairwise post-hoc with "
        "Benjamini-Hochberg FDR control by default (Bonferroni and Holm "
        "available)."
    ),
    usage_guidance=(
        "Use this when normality is violated, when the outcome is ordinal, or "
        "when outliers in the numeric outcome are influential. Statistical "
        "rigor for group comparisons starts with the parametric tools "
        "(`statistical_group_comparison`, `run_independent_t_test`, `run_anova`); "
        "this nonparametric path is the appropriate fallback when their "
        "per-group Shapiro-Wilk reports flag non-normality, especially with "
        "small samples. Like the parametric tools, the active dataset must be "
        "observation-level (one row per subject / order / customer), not "
        "pre-aggregated to one row per group."
    ),
    use_when=[
        "Per-group Shapiro-Wilk normality was rejected in a parametric comparison.",
        "The outcome is ordinal rather than interval-scaled.",
        "Outliers in the numeric outcome appear to drive parametric results.",
        "The user explicitly asks for Mann-Whitney, Kruskal-Wallis, or 'non-parametric'.",
    ],
    do_not_use_when=[
        "No active DataFrame dataset exists.",
        "The user only wants a descriptive grouped table; use groupby_summary instead.",
        "Group sizes are large and normality is not in question; prefer the parametric path for power.",
        "The target variable is categorical; use chi_square for two categorical variables.",
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
            "post_hoc_adjustment": str,
        },
        column_args=[
            "target_col",
            "group_col",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_nonparametric_group_comparison,
    extractor=extract_nonparametric_group_comparison,
    guardrail_evaluators=[
        evaluate_group_comparison_guardrails,
    ],
    apa_methods_writer=write_apa_nonparametric_group_comparison,
    display_config=NONPARAMETRIC_GROUP_COMPARISON_DISPLAY,
    examples=[
        {
            "user_request": "Compare GPA across diet groups using a non-parametric test.",
            "arguments": {
                "target_col": "GPA",
                "group_col": "diet",
            },
        },
        {
            "user_request": "Run Kruskal-Wallis on revenue by region.",
            "arguments": {
                "target_col": "revenue",
                "group_col": "region",
            },
        },
        {
            "user_request": "Mann-Whitney on score between two groups, Bonferroni post-hoc not relevant for 2 groups but use Bonferroni for >2 if needed.",
            "arguments": {
                "target_col": "score",
                "group_col": "group",
                "post_hoc_adjustment": "bonferroni",
            },
        },
    ],
))