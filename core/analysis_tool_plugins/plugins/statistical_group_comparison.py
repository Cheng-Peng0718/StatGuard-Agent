from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple
import math
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from statsmodels.stats.multicomp import pairwise_tukeyhsd

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
    eta_squared_ci,
    omega_squared_ci,
)
from core.analysis_tool_plugins.shared.apa_writers import write_apa_statistical_group_comparison


MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    "NA", "N/A", "NaN", "NULL", "None", "Missing", "Unknown",
}


# ==========================================================
# Helpers
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


def _interpret_cohens_d(d: float | None) -> str | None:
    """Cohen 1988 rules of thumb."""
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


def _interpret_eta_squared(eta2: float | None) -> str | None:
    """Cohen 1988 rules of thumb for eta-squared."""
    if eta2 is None:
        return None

    a = float(eta2)

    if a < 0.01:
        return "negligible"
    if a < 0.06:
        return "small"
    if a < 0.14:
        return "medium"

    return "large"


# ==========================================================
# Assumption checks
# ==========================================================

def _run_levene(groups: dict[str, np.ndarray], center: str = "median") -> dict[str, Any]:
    """
    Levene's test for homogeneity of variances.

    Uses the Brown-Forsythe variant (center='median') by default, which is more
    robust to non-normality than the classic Levene's test (center='mean').
    """
    arrays = list(groups.values())

    if len(arrays) < 2 or any(len(arr) < 2 for arr in arrays):
        return {
            "statistic": None,
            "p_value": None,
            "method": f"Levene's test (center={center})",
            "variances_equal_at_0_05": None,
            "max_to_min_variance_ratio": None,
        }

    try:
        statistic, p_value = stats.levene(*arrays, center=center)
    except Exception:
        statistic, p_value = (float("nan"), float("nan"))

    variances = [float(np.var(arr, ddof=1)) for arr in arrays if len(arr) >= 2]
    var_ratio = None

    if variances and min(variances) > 0:
        var_ratio = max(variances) / min(variances)

    variances_equal = None

    if math.isfinite(float(p_value)):
        variances_equal = bool(p_value >= 0.05)

    return {
        "statistic": _round_or_none(statistic),
        "p_value": _round_or_none(p_value),
        "method": f"Levene's test (center={center}, Brown-Forsythe variant)",
        "variances_equal_at_0_05": variances_equal,
        "max_to_min_variance_ratio": _round_or_none(var_ratio),
    }


def _run_shapiro_per_group(groups: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    """
    Shapiro-Wilk normality test per group.

    Returns None p-value when n < 3 or n > 5000 (Shapiro-Wilk valid range).
    """
    rows = []

    for name, arr in groups.items():
        n = int(len(arr))

        if n < 3 or n > 5000:
            rows.append({
                "group": str(name),
                "n": n,
                "statistic": None,
                "p_value": None,
                "normal_at_0_05": None,
                "note": (
                    "n < 3 (not enough)" if n < 3
                    else "n > 5000 (Shapiro-Wilk not valid; consider Anderson-Darling)"
                ),
            })
            continue

        try:
            stat, p = stats.shapiro(arr)
        except Exception:
            rows.append({
                "group": str(name),
                "n": n,
                "statistic": None,
                "p_value": None,
                "normal_at_0_05": None,
                "note": "Shapiro-Wilk failed",
            })
            continue

        rows.append({
            "group": str(name),
            "n": n,
            "statistic": _round_or_none(stat),
            "p_value": _round_or_none(p),
            "normal_at_0_05": bool(p >= 0.05) if math.isfinite(float(p)) else None,
            "note": None,
        })

    return rows


# Thresholds for the nonparametric-switch decision (see _decide_nonparametric).
# - Below this per-group n, the central limit theorem cannot be relied on, so
#   any non-normality is a reason to prefer a rank-based test.
_SMALL_GROUP_N = 30
# - At or above this absolute skew, the distribution is severely skewed and the
#   CLT protection from n >= 30 is no longer adequate (Fagerland 2012; Fagerland
#   & Sandvik 2009 show Welch-type tests stay acceptable under moderate skew ~1
#   but degrade under severe skew). Non-normality then warrants a rank-based
#   test even for moderately large samples.
_HIGH_SKEW = 1.5


def _max_abs_skew(groups: dict[str, np.ndarray]) -> float | None:
    """Largest absolute sample skewness across groups (None if uncomputable)."""
    skews = []
    for arr in groups.values():
        if len(arr) >= 3:
            try:
                s = float(stats.skew(arr, bias=False))
                if math.isfinite(s):
                    skews.append(abs(s))
            except Exception:
                continue
    return max(skews) if skews else None


def _decide_nonparametric(
    groups: dict[str, np.ndarray],
    any_group_non_normal: bool,
) -> dict[str, Any]:
    """
    Deterministic decision: should the primary test be rank-based?

    Rule (two-factor):
        switch  <=>  any group is non-normal (Shapiro p < .05)
                     AND ( smallest group n < 30          # CLT unreliable
                           OR  max |skew| >= 1.0 )         # CLT inadequate

    Rationale: a bare "n < 30" rule is a folk simplification. With strong skew,
    n = 30 (or more) is not enough for the t-test's sampling distribution to be
    approximately normal, so we also switch on high skew. Conversely, with a
    large sample and only mild skew, Shapiro will reject normality yet the
    t-test remains valid and more powerful -- so we do NOT switch there.

    Returns a structured record (always), including why the decision was made,
    so the choice is fully auditable.
    """
    min_n = min((len(a) for a in groups.values()), default=0)
    max_skew = _max_abs_skew(groups)

    small_sample = min_n < _SMALL_GROUP_N
    high_skew = (max_skew is not None) and (max_skew >= _HIGH_SKEW)

    switch = bool(any_group_non_normal and (small_sample or high_skew))

    reasons = []
    if switch:
        if small_sample:
            reasons.append(f"smallest group n={min_n} < {_SMALL_GROUP_N}")
        if high_skew:
            reasons.append(f"max |skew|={max_skew:.2f} >= {_HIGH_SKEW}")

    return {
        "switch_to_nonparametric": switch,
        "any_group_non_normal": bool(any_group_non_normal),
        "min_group_n": int(min_n),
        "max_abs_skew": _round_or_none(max_skew),
        "small_sample": bool(small_sample),
        "high_skew": bool(high_skew),
        "reasons": reasons,
    }


# ==========================================================
# Pairwise post-hoc tests
# ==========================================================

def _run_tukey_hsd(work: pd.DataFrame, alpha: float) -> list[dict[str, Any]]:
    """
    Tukey HSD post-hoc test for one-way ANOVA when variances are equal.

    Uses statsmodels pairwise_tukeyhsd which controls family-wise error rate.
    """
    try:
        result = pairwise_tukeyhsd(
            endog=work["target"].astype(float).to_numpy(),
            groups=work["group"].astype(str).to_numpy(),
            alpha=alpha,
        )
    except Exception:
        return []

    rows = []

    # result._results_table.data has header + data rows
    # columns: group1, group2, meandiff, p-adj, lower, upper, reject
    try:
        data = result._results_table.data
        header = [str(h) for h in data[0]]
        # Map header to index
        idx = {h: i for i, h in enumerate(header)}

        for raw_row in data[1:]:
            g1 = str(raw_row[idx.get("group1", 0)])
            g2 = str(raw_row[idx.get("group2", 1)])
            mean_diff = float(raw_row[idx.get("meandiff", 2)])
            p_adj = float(raw_row[idx.get("p-adj", 3)])
            ci_lower = float(raw_row[idx.get("lower", 4)])
            ci_upper = float(raw_row[idx.get("upper", 5)])
            reject = bool(raw_row[idx.get("reject", 6)])

            rows.append({
                "group1": g1,
                "group2": g2,
                "mean_difference_g1_minus_g2": _round_or_none(mean_diff),
                "p_value_adjusted": _round_or_none(p_adj),
                "ci_lower": _round_or_none(ci_lower),
                "ci_upper": _round_or_none(ci_upper),
                "significant_at_alpha": reject,
                "adjustment_method": "Tukey HSD",
            })
    except Exception:
        return []

    return rows


def _run_games_howell(groups: dict[str, np.ndarray], alpha: float) -> list[dict[str, Any]]:
    """
    Games-Howell post-hoc test for one-way ANOVA when variances are unequal.

    Uses Welch-Satterthwaite degrees of freedom per pair and the studentized
    range distribution for p-value adjustment. Family-wise error rate is
    controlled at alpha.
    """
    rows = []

    labels = sorted(groups.keys())
    k = len(labels)

    if k < 2:
        return rows

    for g1, g2 in itertools.combinations(labels, 2):
        x1 = groups[g1]
        x2 = groups[g2]

        n1 = len(x1)
        n2 = len(x2)

        if n1 < 2 or n2 < 2:
            rows.append({
                "group1": str(g1),
                "group2": str(g2),
                "mean_difference_g1_minus_g2": None,
                "t_statistic": None,
                "degrees_of_freedom": None,
                "p_value_adjusted": None,
                "ci_lower": None,
                "ci_upper": None,
                "significant_at_alpha": None,
                "adjustment_method": "Games-Howell",
            })
            continue

        mean_diff = float(np.mean(x1) - np.mean(x2))
        s1 = float(np.var(x1, ddof=1))
        s2 = float(np.var(x2, ddof=1))

        se = math.sqrt(s1 / n1 + s2 / n2)

        if se <= 0:
            rows.append({
                "group1": str(g1),
                "group2": str(g2),
                "mean_difference_g1_minus_g2": _round_or_none(mean_diff),
                "t_statistic": None,
                "degrees_of_freedom": None,
                "p_value_adjusted": None,
                "ci_lower": None,
                "ci_upper": None,
                "significant_at_alpha": None,
                "adjustment_method": "Games-Howell",
            })
            continue

        # Welch-Satterthwaite df
        df_num = (s1 / n1 + s2 / n2) ** 2
        df_den = (s1 / n1) ** 2 / (n1 - 1) + (s2 / n2) ** 2 / (n2 - 1)
        df = _safe_divide(df_num, df_den)

        if df is None or df <= 0:
            rows.append({
                "group1": str(g1),
                "group2": str(g2),
                "mean_difference_g1_minus_g2": _round_or_none(mean_diff),
                "t_statistic": None,
                "degrees_of_freedom": None,
                "p_value_adjusted": None,
                "ci_lower": None,
                "ci_upper": None,
                "significant_at_alpha": None,
                "adjustment_method": "Games-Howell",
            })
            continue

        # Games-Howell uses the studentized range distribution with statistic
        # q = |mean_diff| / (se / sqrt(2))
        q_stat = abs(mean_diff) / (se / math.sqrt(2))

        try:
            # studentized_range.sf gives upper-tail probability
            p_adj = float(stats.studentized_range.sf(q_stat, k, df))
            q_crit = float(stats.studentized_range.ppf(1 - alpha, k, df))
            margin = q_crit * (se / math.sqrt(2))
        except Exception:
            p_adj = float("nan")
            margin = float("nan")

        ci_lower = mean_diff - margin if math.isfinite(margin) else None
        ci_upper = mean_diff + margin if math.isfinite(margin) else None

        rows.append({
            "group1": str(g1),
            "group2": str(g2),
            "mean_difference_g1_minus_g2": _round_or_none(mean_diff),
            "t_statistic": _round_or_none(mean_diff / se) if se > 0 else None,
            "degrees_of_freedom": _round_or_none(df),
            "p_value_adjusted": _round_or_none(p_adj) if math.isfinite(p_adj) else None,
            "ci_lower": _round_or_none(ci_lower),
            "ci_upper": _round_or_none(ci_upper),
            "significant_at_alpha": (
                bool(p_adj < alpha) if math.isfinite(p_adj) else None
            ),
            "adjustment_method": "Games-Howell",
        })

    return rows


# ==========================================================
# Main test routines
# ==========================================================

def _run_welch_t_test(groups: dict[str, np.ndarray], alpha: float) -> dict[str, Any]:
    labels = sorted(groups.keys())

    group1, group2 = labels[0], labels[1]
    x1, x2 = groups[group1], groups[group2]

    t_stat, p_value = stats.ttest_ind(x1, x2, equal_var=False, nan_policy="omit")
    df = _welch_degrees_of_freedom(x1, x2)

    d, g = _cohens_d_and_hedges_g(x1, x2)
    n1, n2 = len(x1), len(x2)
    d_ci_low, d_ci_high = cohens_d_independent_ci(d, n1, n2, alpha) if d is not None else (None, None)
    g_ci_low, g_ci_high = hedges_g_independent_ci(g, n1, n2, alpha) if g is not None else (None, None)
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
        "effect_size_ci_low": _round_or_none(g_ci_low),
        "effect_size_ci_high": _round_or_none(g_ci_high),
        "effect_size_magnitude": _interpret_cohens_d(g),
        "cohens_d": _round_or_none(d),
        "cohens_d_ci_low": _round_or_none(d_ci_low),
        "cohens_d_ci_high": _round_or_none(d_ci_high),
        "significant_at_alpha": bool(p_value < alpha) if math.isfinite(float(p_value)) else None,
        "significant_at_0_05": bool(p_value < 0.05) if math.isfinite(float(p_value)) else None,
    }


def _run_one_way_anova(groups: dict[str, np.ndarray], alpha: float) -> dict[str, Any]:
    """
    Classic Fisher one-way ANOVA. Assumes equal variances.
    """
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

    # 95% CI for eta-squared (Smithson 2003, via noncentral-F inversion)
    eta_ci_low, eta_ci_high = (
        eta_squared_ci(float(f_stat), df_between, df_within, alpha)
        if math.isfinite(float(f_stat))
        else (None, None)
    )
    omega_ci_low, omega_ci_high = (
        omega_squared_ci(float(f_stat), df_between, df_within, n, alpha)
        if math.isfinite(float(f_stat))
        else (None, None)
    )

    return {
        "method": "One-way ANOVA",
        "test_family": "multi_group_numeric_comparison",
        "F_statistic": _round_or_none(f_stat),
        "degrees_of_freedom_between": int(df_between),
        "degrees_of_freedom_within": int(df_within),
        "p_value": _round_or_none(p_value),
        "effect_size_name": "eta squared",
        "effect_size": _round_or_none(eta_squared),
        "effect_size_ci_low": _round_or_none(eta_ci_low),
        "effect_size_ci_high": _round_or_none(eta_ci_high),
        "effect_size_magnitude": _interpret_eta_squared(eta_squared),
        "eta_squared": _round_or_none(eta_squared),
        "eta_squared_ci_low": _round_or_none(eta_ci_low),
        "eta_squared_ci_high": _round_or_none(eta_ci_high),
        "omega_squared": _round_or_none(omega_squared),
        "omega_squared_ci_low": _round_or_none(omega_ci_low),
        "omega_squared_ci_high": _round_or_none(omega_ci_high),
        "significant_at_alpha": bool(p_value < alpha) if math.isfinite(float(p_value)) else None,
        "significant_at_0_05": bool(p_value < 0.05) if math.isfinite(float(p_value)) else None,
    }


def _run_welch_anova(groups: dict[str, np.ndarray], alpha: float) -> dict[str, Any]:
    """
    Welch's ANOVA (Alexander-Govern variant) for unequal variances.

    Falls back to a manual Welch ANOVA computation if scipy.stats.alexandergovern
    is not available. Returns same eta-squared computed from the underlying ANOVA
    decomposition so effect size remains comparable.
    """
    arrays = list(groups.values())
    k = len(arrays)

    method = "Welch's ANOVA (does not assume equal variances)"

    statistic = None
    p_value = None

    # Try scipy 1.10+ alexandergovern
    try:
        ag_result = stats.alexandergovern(*arrays)
        statistic = float(ag_result.statistic)
        p_value = float(ag_result.pvalue)
        method = "Alexander-Govern ANOVA (does not assume equal variances)"
    except Exception:
        # Manual Welch's ANOVA fallback
        try:
            n_i = np.array([len(a) for a in arrays], dtype=float)
            mean_i = np.array([np.mean(a) for a in arrays], dtype=float)
            var_i = np.array([np.var(a, ddof=1) for a in arrays], dtype=float)

            w_i = n_i / var_i
            w_sum = float(np.sum(w_i))
            grand_mean_w = float(np.sum(w_i * mean_i) / w_sum)

            numerator = float(np.sum(w_i * (mean_i - grand_mean_w) ** 2)) / (k - 1)

            denom_inner = float(np.sum(((1 - w_i / w_sum) ** 2) / (n_i - 1)))
            denominator = 1 + (2 * (k - 2) / (k ** 2 - 1)) * denom_inner

            f_stat = numerator / denominator

            df_num = k - 1
            df_den = (k ** 2 - 1) / (3 * denom_inner)

            statistic = float(f_stat)
            p_value = float(stats.f.sf(f_stat, df_num, df_den))
        except Exception:
            statistic = None
            p_value = None

    # Effect size: compute eta-squared from the overall data partition for
    # comparability with classic ANOVA.
    all_values = np.concatenate(arrays)
    grand_mean = float(np.mean(all_values))

    ss_between = sum(len(x) * (float(np.mean(x)) - grand_mean) ** 2 for x in arrays)
    ss_within = sum(float(np.sum((x - float(np.mean(x))) ** 2)) for x in arrays)
    ss_total = ss_between + ss_within

    eta_squared = _safe_divide(ss_between, ss_total)

    return {
        "method": method,
        "test_family": "multi_group_numeric_comparison",
        "F_statistic": _round_or_none(statistic),
        "degrees_of_freedom_between": int(k - 1),
        "degrees_of_freedom_within": None,
        "p_value": _round_or_none(p_value),
        "effect_size_name": "eta squared",
        "effect_size": _round_or_none(eta_squared),
        "effect_size_magnitude": _interpret_eta_squared(eta_squared),
        "eta_squared": _round_or_none(eta_squared),
        "omega_squared": None,
        "significant_at_alpha": (
            bool(p_value < alpha)
            if p_value is not None and math.isfinite(p_value)
            else None
        ),
        "significant_at_0_05": (
            bool(p_value < 0.05)
            if p_value is not None and math.isfinite(p_value)
            else None
        ),
    }


# ==========================================================
# Main execute
# ==========================================================

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

        # ----------------------------------------------------
        # Assumption check: equality of variances
        # ----------------------------------------------------
        levene = _run_levene(groups, center="median")

        # ----------------------------------------------------
        # Assumption check: normality per group (auxiliary)
        # ----------------------------------------------------
        shapiro_rows = _run_shapiro_per_group(groups)

        any_group_non_normal = any(
            row.get("normal_at_0_05") is False for row in shapiro_rows
        )

        # ----------------------------------------------------
        # Deterministic decision: rank-based primary test?
        # ----------------------------------------------------
        np_decision = _decide_nonparametric(groups, any_group_non_normal)

        # ----------------------------------------------------
        # Pick primary test
        # ----------------------------------------------------
        post_hoc_rows: list[dict[str, Any]] = []
        post_hoc_method = None
        secondary_test = None  # the parametric result, kept for transparency

        if np_decision["switch_to_nonparametric"]:
            # Strong non-normality (small sample or high skew): use a rank-based
            # primary test, but ALSO compute the parametric test and keep it as
            # a secondary result so expert users can compare.
            from core.analysis_tool_plugins.plugins.nonparametric_group_comparison import (
                _run_mann_whitney,
                _run_kruskal_wallis,
            )
            if len(groups) == 2:
                test_details = _run_mann_whitney(groups, alpha)
                secondary_test = _run_welch_t_test(groups, alpha)
            else:
                test_details = _run_kruskal_wallis(groups, alpha)
                variances_equal = levene.get("variances_equal_at_0_05")
                if variances_equal is False:
                    secondary_test = _run_welch_anova(groups, alpha)
                else:
                    secondary_test = _run_one_way_anova(groups, alpha)

        elif len(groups) == 2:
            # Two-group case: Welch's t-test (does not require equal variances)
            test_details = _run_welch_t_test(groups, alpha)

        else:
            # Three or more groups
            variances_equal = levene.get("variances_equal_at_0_05")

            if variances_equal is False:
                test_details = _run_welch_anova(groups, alpha)
            else:
                # Includes True (equal) and None (could not be tested -> default to classic)
                test_details = _run_one_way_anova(groups, alpha)

            # Run post-hoc only when the omnibus is significant
            if test_details.get("significant_at_alpha"):
                if variances_equal is False:
                    post_hoc_rows = _run_games_howell(groups, alpha)
                    post_hoc_method = "Games-Howell (FWER-controlled, unequal variances)"
                else:
                    post_hoc_rows = _run_tukey_hsd(work, alpha)
                    post_hoc_method = "Tukey HSD (FWER-controlled, equal variances)"

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

        # ----------------------------------------------------
        # Build assumptions/limitations
        # ----------------------------------------------------
        assumptions_and_limitations = [
            "This is an observational comparison unless the data came from a randomized experiment.",
            "The test assumes independent observations within and across groups.",
            "The numeric outcome is compared across observed groups after dropping missing target/group values.",
        ]

        if len(groups) == 2:
            assumptions_and_limitations.append(
                "Welch's t-test does not assume equal variances, but extremely small or highly skewed groups still require caution."
            )
        else:
            if levene.get("variances_equal_at_0_05") is False:
                assumptions_and_limitations.append(
                    "Levene's test indicated unequal variances; Welch's ANOVA was used instead of the classic F-test."
                )
            elif levene.get("variances_equal_at_0_05") is True:
                assumptions_and_limitations.append(
                    "Levene's test did not flag unequal variances; the classic one-way ANOVA assumption appears acceptable."
                )

            if test_details.get("significant_at_alpha"):
                assumptions_and_limitations.append(
                    f"Significant omnibus test; pairwise post-hoc comparisons reported using {post_hoc_method}."
                )
            else:
                assumptions_and_limitations.append(
                    "Omnibus test not significant; no pairwise post-hoc comparisons were performed."
                )

        if np_decision["switch_to_nonparametric"]:
            reason_str = " and ".join(np_decision["reasons"])
            assumptions_and_limitations.append(
                "Shapiro-Wilk indicated non-normality and "
                f"{reason_str}; the primary test was switched to a rank-based "
                "method (Mann-Whitney U for two groups, Kruskal-Wallis for three "
                "or more). The parametric result is retained as a secondary "
                "test for comparison."
            )
        elif any_group_non_normal:
            assumptions_and_limitations.append(
                "Shapiro-Wilk flagged possible non-normality, but the sample is "
                "large enough and not highly skewed, so the parametric test "
                "remains valid (central limit theorem) and was kept as primary."
            )

        # ----------------------------------------------------
        # Compose result
        # ----------------------------------------------------
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
            "levene_test": levene,
            "shapiro_per_group": shapiro_rows,
            "nonparametric_switch": np_decision,
            "secondary_test": secondary_test,
            "post_hoc_method": post_hoc_method,
            "post_hoc_pairwise": post_hoc_rows,
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


# ==========================================================
# Extractor / Display
# ==========================================================

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

    if payload.get("post_hoc_pairwise"):
        n_sig_pairs = sum(
            1 for row in payload.get("post_hoc_pairwise", [])
            if row.get("significant_at_alpha")
        )
        summary += f" Post-hoc ({payload.get('post_hoc_method')}): {n_sig_pairs} significant pair(s)."

    levene = payload.get("levene_test", {}) or {}

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
        "effect_size_ci_low": payload.get("effect_size_ci_low"),
        "effect_size_ci_high": payload.get("effect_size_ci_high"),
        "effect_size_magnitude": payload.get("effect_size_magnitude"),
        "cohens_d": payload.get("cohens_d"),
        "cohens_d_ci_low": payload.get("cohens_d_ci_low"),
        "cohens_d_ci_high": payload.get("cohens_d_ci_high"),
        "eta_squared": payload.get("eta_squared"),
        "eta_squared_ci_low": payload.get("eta_squared_ci_low"),
        "eta_squared_ci_high": payload.get("eta_squared_ci_high"),
        "omega_squared": payload.get("omega_squared"),
        "omega_squared_ci_low": payload.get("omega_squared_ci_low"),
        "omega_squared_ci_high": payload.get("omega_squared_ci_high"),
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
        "levene_p_value": levene.get("p_value"),
        "variances_equal_at_0_05": levene.get("variances_equal_at_0_05"),
        "max_to_min_variance_ratio": levene.get("max_to_min_variance_ratio"),
    })

    tables: Dict[str, Any] = {}

    if payload.get("group_summaries"):
        tables["group_summaries"] = payload.get("group_summaries")

    if payload.get("post_hoc_pairwise"):
        tables["post_hoc_pairwise"] = payload.get("post_hoc_pairwise")

    if payload.get("shapiro_per_group"):
        # Only surface the table if at least one group could actually be tested
        has_tested = any(
            row.get("p_value") is not None for row in payload.get("shapiro_per_group", [])
        )
        if has_tested:
            tables["shapiro_per_group"] = payload.get("shapiro_per_group")

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
        "levene_test": payload.get("levene_test"),
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
            "significant_at_0_05": "Significant at 0.05",
            "effect_size_name": "Effect size",
            "effect_size": "Effect size value",
            "effect_size_ci_low": "Effect size 95% CI lower",
            "effect_size_ci_high": "Effect size 95% CI upper",
            "effect_size_magnitude": "Effect size magnitude",
            "cohens_d": "Cohen's d",
            "cohens_d_ci_low": "Cohen's d 95% CI lower",
            "cohens_d_ci_high": "Cohen's d 95% CI upper",
            "eta_squared": "Eta squared",
            "eta_squared_ci_low": "Eta squared 95% CI lower",
            "eta_squared_ci_high": "Eta squared 95% CI upper",
            "omega_squared": "Omega squared",
            "omega_squared_ci_low": "Omega squared 95% CI lower",
            "omega_squared_ci_high": "Omega squared 95% CI upper",
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
            "levene_p_value": "Levene p-value",
            "variances_equal_at_0_05": "Variances equal (Levene 0.05)",
            "max_to_min_variance_ratio": "Max/min variance ratio",
        },
        formatters={
            "p_value": format_p_value,
            "significant_at_alpha": format_bool_yes_no,
            "significant_at_0_05": format_bool_yes_no,
            "effect_size": format_number,
            "effect_size_ci_low": format_number,
            "effect_size_ci_high": format_number,
            "cohens_d": format_number,
            "cohens_d_ci_low": format_number,
            "cohens_d_ci_high": format_number,
            "eta_squared": format_number,
            "eta_squared_ci_low": format_number,
            "eta_squared_ci_high": format_number,
            "omega_squared": format_number,
            "omega_squared_ci_low": format_number,
            "omega_squared_ci_high": format_number,
            "top_group_mean": format_number,
            "lowest_group_mean": format_number,
            "top_minus_lowest_mean_difference": format_number,
            "top_vs_lowest_relative_lift": format_number,
            "F_statistic": format_number,
            "t_statistic": format_number,
            "degrees_of_freedom": format_number,
            "levene_p_value": format_p_value,
            "variances_equal_at_0_05": format_bool_yes_no,
            "max_to_min_variance_ratio": format_number,
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
            "effect_size_ci_low",
            "effect_size_ci_high",
            "effect_size_magnitude",
            "cohens_d",
            "cohens_d_ci_low",
            "cohens_d_ci_high",
            "eta_squared",
            "eta_squared_ci_low",
            "eta_squared_ci_high",
            "omega_squared",
            "omega_squared_ci_low",
            "omega_squared_ci_high",
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
            "levene_p_value",
            "variances_equal_at_0_05",
            "max_to_min_variance_ratio",
        ],
    ),
    tables={
        "group_summaries": TableDisplayConfig(
            column_labels={
                "group": "Group",
                "n": "N",
                "mean": "Mean",
                "std": "SD",
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
        "post_hoc_pairwise": TableDisplayConfig(
            column_labels={
                "group1": "Group 1",
                "group2": "Group 2",
                "mean_difference_g1_minus_g2": "Mean diff (g1-g2)",
                "t_statistic": "t statistic",
                "degrees_of_freedom": "DF",
                "p_value_adjusted": "Adjusted p-value",
                "ci_lower": "95% CI lower",
                "ci_upper": "95% CI upper",
                "significant_at_alpha": "Significant",
                "adjustment_method": "Adjustment",
            },
            column_order=[
                "group1",
                "group2",
                "mean_difference_g1_minus_g2",
                "t_statistic",
                "degrees_of_freedom",
                "p_value_adjusted",
                "ci_lower",
                "ci_upper",
                "significant_at_alpha",
                "adjustment_method",
            ],
            column_formatters={
                "mean_difference_g1_minus_g2": format_number,
                "t_statistic": format_number,
                "degrees_of_freedom": format_number,
                "p_value_adjusted": format_p_value,
                "ci_lower": format_number,
                "ci_upper": format_number,
                "significant_at_alpha": format_bool_yes_no,
            },
        ),
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
    tool_name="statistical_group_comparison",
    display_name="Statistical Group Comparison",
    evidence_categories=["group_comparison", "statistical_inference", "confidence_interval"],
    description=(
        "Run an inferential statistical comparison of a numeric outcome across "
        "levels of a categorical group variable. This is the correct tool for "
        "questions about whether an outcome differs across groups, segments, "
        "regions, cohorts, treatments, or categories."
    ),
    usage_guidance=(
        "Use this tool when the user asks whether a numeric outcome differs across "
        "groups, segments, regions, cohorts, treatments, or categories. "
        "This tool produces group_comparison evidence. "
        "Examples: "
        "to test whether total_revenue differs by region, use one row per customer or order with "
        "both region and total_revenue, not SELECT region, SUM(revenue) GROUP BY region."
        "compare total_revenue by segment -> target_col='total_revenue', group_col='segment'; "
        "compare total_revenue by region -> target_col='total_revenue', group_col='region'. "
        "Do not use groupby_summary to satisfy group_comparison evidence."
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
    is_inferential=True,
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
    guardrail_evaluators=[
        evaluate_group_comparison_guardrails,
    ],
    apa_methods_writer=write_apa_statistical_group_comparison,
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