from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple
import math

import numpy as np
import pandas as pd
from scipy import stats
from scipy import optimize

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
from core.analysis_tool_plugins.shared.apa_writers import write_apa_paired_comparison
from core.guardrails import _new_finding


MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    "NA", "N/A", "NaN", "NULL", "None", "Missing", "Unknown",
}


# ==========================================================
# Helpers (mirrored)
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
# Effect size and CI helpers
# ==========================================================

def _interpret_cohens_d(d: float | None) -> str | None:
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


def _cohens_d_z_ci(
    d_z: float,
    n: int,
    alpha: float = 0.05,
) -> tuple[float | None, float | None]:
    """
    Confidence interval for Cohen's d_z using the noncentral t distribution.

    Approach:
      - The observed t statistic is t_obs = d_z * sqrt(n) with df = n - 1.
      - The CI for the noncentrality parameter (ncp) is obtained by inversion:
          lower L: P(T(df, L) > t_obs) = alpha / 2   (i.e. sf = alpha/2)
          upper U: P(T(df, U) <= t_obs) = alpha / 2  (i.e. cdf = alpha/2)
      - The CI on d_z is then [L, U] / sqrt(n).
    """
    if not math.isfinite(d_z) or n < 2:
        return None, None

    df = n - 1
    t_obs = d_z * math.sqrt(n)

    # Generous bracket; scales with sqrt(df) plus a fixed buffer
    search_range = max(15.0, 6.0 * math.sqrt(max(df, 1)))

    def lower_target(ncp: float) -> float:
        return float(stats.nct.sf(t_obs, df, ncp)) - alpha / 2.0

    def upper_target(ncp: float) -> float:
        return float(stats.nct.cdf(t_obs, df, ncp)) - alpha / 2.0

    try:
        ncp_lower = optimize.brentq(
            lower_target,
            t_obs - search_range,
            t_obs,
            maxiter=200,
        )
    except Exception:
        ncp_lower = None

    try:
        ncp_upper = optimize.brentq(
            upper_target,
            t_obs,
            t_obs + search_range,
            maxiter=200,
        )
    except Exception:
        ncp_upper = None

    d_lower = ncp_lower / math.sqrt(n) if ncp_lower is not None else None
    d_upper = ncp_upper / math.sqrt(n) if ncp_upper is not None else None

    return d_lower, d_upper


def _walsh_pseudomedian_and_ci(
    diffs_nonzero: np.ndarray,
    alpha: float,
    max_n_for_full_walsh: int = 4000,
) -> tuple[float | None, float | None, float | None]:
    """
    Hodges-Lehmann pseudomedian and distribution-free CI for the one-sample
    Wilcoxon signed-rank test, computed from the Walsh averages of the
    nonzero differences.

      Walsh averages: W_{ij} = (D_i + D_j) / 2 for all 1 <= i <= j <= n.
      Pseudomedian = median(W).
      CI: (1 - alpha) bounds taken from order statistics of sorted W using the
          asymptotic-normal critical value from the signed-rank null variance
          V = n(n+1)(2n+1)/24.

    For very large n (n > max_n_for_full_walsh), the full pairwise enumeration
    becomes memory-heavy; the point estimate falls back to the sample median of
    differences and the CI is suppressed.
    """
    diffs_nonzero = np.asarray(diffs_nonzero, dtype=float)
    diffs_nonzero = diffs_nonzero[np.isfinite(diffs_nonzero)]

    n = int(diffs_nonzero.size)

    if n < 2:
        return None, None, None

    if n > max_n_for_full_walsh:
        return float(np.median(diffs_nonzero)), None, None

    # Vectorised Walsh averages using broadcasting; take upper triangle inc. diag
    pair_sum = diffs_nonzero[:, None] + diffs_nonzero[None, :]
    iu = np.triu_indices(n, k=0)
    walsh = (pair_sum[iu] / 2.0)
    walsh.sort()

    K = walsh.size  # K = n(n+1)/2

    pseudomedian = float(np.median(walsh))

    if K < 4:
        return pseudomedian, None, None

    z_crit = float(stats.norm.ppf(1.0 - alpha / 2.0))
    variance_signed_rank = n * (n + 1) * (2 * n + 1) / 24.0
    spread = z_crit * math.sqrt(variance_signed_rank)

    # Lower order statistic index for the (1 - alpha) CI on the pseudomedian
    k_low = int(math.floor(K / 2.0 - spread))

    if k_low < 1:
        k_low = 1
    if k_low > K:
        k_low = K

    k_high = K - k_low + 1

    ci_low = float(walsh[k_low - 1])
    ci_high = float(walsh[k_high - 1])

    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low

    return pseudomedian, ci_low, ci_high


# ==========================================================
# Main test routines
# ==========================================================

def _run_paired_t_test(diffs: np.ndarray, alpha: float) -> dict[str, Any]:
    """
    Paired (one-sample) t-test on differences, with Cohen's d_z and a
    noncentral-t-based CI on d_z. The mean-difference CI uses the standard
    Student t critical value.
    """
    n = int(diffs.size)

    if n < 2:
        return {
            "method": "Paired samples t-test",
            "test_family": "paired_numeric_comparison",
            "n_pairs": n,
        }

    mean_diff = float(np.mean(diffs))
    sd_diff = float(np.std(diffs, ddof=1))

    if sd_diff == 0:
        return {
            "method": "Paired samples t-test",
            "test_family": "paired_numeric_comparison",
            "n_pairs": n,
            "mean_difference": _round_or_none(mean_diff),
            "sd_difference": _round_or_none(sd_diff),
            "t_statistic": None,
            "degrees_of_freedom": int(n - 1),
            "p_value": None,
            "mean_difference_ci_low": None,
            "mean_difference_ci_high": None,
            "cohens_d_z": None,
            "cohens_d_z_ci_low": None,
            "cohens_d_z_ci_high": None,
            "effect_size_magnitude": None,
            "significant_at_alpha": None,
            "significant_at_0_05": None,
            "degenerate_zero_variance": True,
        }

    se = sd_diff / math.sqrt(n)
    t_stat = mean_diff / se
    df = n - 1
    p_value = float(2.0 * (1.0 - stats.t.cdf(abs(t_stat), df)))

    t_crit = float(stats.t.ppf(1.0 - alpha / 2.0, df))
    ci_low = mean_diff - t_crit * se
    ci_high = mean_diff + t_crit * se

    d_z = mean_diff / sd_diff
    d_z_low, d_z_high = _cohens_d_z_ci(d_z, n, alpha=alpha)

    return {
        "method": "Paired samples t-test",
        "test_family": "paired_numeric_comparison",
        "n_pairs": n,
        "mean_difference": _round_or_none(mean_diff),
        "sd_difference": _round_or_none(sd_diff),
        "t_statistic": _round_or_none(t_stat),
        "degrees_of_freedom": int(df),
        "p_value": _round_or_none(p_value),
        "mean_difference_ci_low": _round_or_none(ci_low),
        "mean_difference_ci_high": _round_or_none(ci_high),
        "cohens_d_z": _round_or_none(d_z),
        "cohens_d_z_ci_low": _round_or_none(d_z_low),
        "cohens_d_z_ci_high": _round_or_none(d_z_high),
        "effect_size_magnitude": _interpret_cohens_d(d_z),
        "significant_at_alpha": bool(p_value < alpha) if math.isfinite(p_value) else None,
        "significant_at_0_05": bool(p_value < 0.05) if math.isfinite(p_value) else None,
        "degenerate_zero_variance": False,
    }


def _run_wilcoxon_signed_rank(diffs: np.ndarray, alpha: float) -> dict[str, Any]:
    """
    Wilcoxon signed-rank test on paired differences.

    Zero-difference pairs are excluded (the standard 'wilcox' method). The
    effect size is the matched-pairs rank-biserial correlation

        r = (T+ - T-) / (T+ + T-)

    bounded in [-1, 1], where T+ and T- are the sums of positive and negative
    signed ranks. Hodges-Lehmann pseudomedian and distribution-free CI are
    computed from the Walsh averages of the nonzero differences.
    """
    n_total = int(diffs.size)

    nonzero = diffs[diffs != 0]
    n_zeros = int(n_total - nonzero.size)
    n = int(nonzero.size)

    if n < 1:
        return {
            "method": "Wilcoxon signed-rank test",
            "test_family": "paired_numeric_comparison_nonparametric",
            "n_pairs_used": n,
            "n_zero_difference_pairs_dropped": n_zeros,
            "W_statistic": None,
            "p_value": None,
            "rank_biserial_correlation": None,
            "effect_size_magnitude": None,
            "hodges_lehmann_pseudomedian": None,
            "hodges_lehmann_ci_low": None,
            "hodges_lehmann_ci_high": None,
            "significant_at_alpha": None,
            "significant_at_0_05": None,
        }

    try:
        # scipy returns the smaller of T+ and T- by default; we ask for the
        # sum-of-positive-ranks form for unambiguous rank-biserial computation.
        ranks = stats.rankdata(np.abs(nonzero))
        signs = np.sign(nonzero)
        t_pos = float(np.sum(ranks[signs > 0]))
        t_neg = float(np.sum(ranks[signs < 0]))

        # Use scipy for the p-value with its default tie/continuity handling.
        # zero_method='wilcox' is the modern default (drops zeros, which we
        # have already done by passing nonzero).
        result = stats.wilcoxon(
            nonzero,
            zero_method="wilcox",
            alternative="two-sided",
            mode="auto",
        )
        w_statistic = float(result.statistic)
        p_value = float(result.pvalue)
    except Exception:
        t_pos = float("nan")
        t_neg = float("nan")
        w_statistic = float("nan")
        p_value = float("nan")

    if math.isfinite(t_pos) and math.isfinite(t_neg) and (t_pos + t_neg) > 0:
        r_rb = (t_pos - t_neg) / (t_pos + t_neg)
    else:
        r_rb = None

    pm, pm_low, pm_high = _walsh_pseudomedian_and_ci(nonzero, alpha)

    return {
        "method": "Wilcoxon signed-rank test",
        "test_family": "paired_numeric_comparison_nonparametric",
        "n_pairs_used": n,
        "n_zero_difference_pairs_dropped": n_zeros,
        "W_statistic": _round_or_none(w_statistic),
        "sum_of_positive_ranks": _round_or_none(t_pos),
        "sum_of_negative_ranks": _round_or_none(t_neg),
        "p_value": _round_or_none(p_value),
        "rank_biserial_correlation": _round_or_none(r_rb),
        "effect_size_magnitude": _interpret_rank_biserial(r_rb),
        "hodges_lehmann_pseudomedian": _round_or_none(pm),
        "hodges_lehmann_ci_low": _round_or_none(pm_low),
        "hodges_lehmann_ci_high": _round_or_none(pm_high),
        "significant_at_alpha": (
            bool(p_value < alpha) if math.isfinite(p_value) else None
        ),
        "significant_at_0_05": (
            bool(p_value < 0.05) if math.isfinite(p_value) else None
        ),
    }


# ==========================================================
# Differences normality (Shapiro on the differences)
# ==========================================================

def _run_differences_shapiro(diffs: np.ndarray) -> dict[str, Any]:
    n = int(diffs.size)

    if n < 3:
        return {
            "n": n,
            "statistic": None,
            "p_value": None,
            "normal_at_0_05": None,
            "note": "n < 3 (Shapiro-Wilk not applicable).",
        }

    if n > 5000:
        return {
            "n": n,
            "statistic": None,
            "p_value": None,
            "normal_at_0_05": None,
            "note": "n > 5000 (Shapiro-Wilk not valid; consider Anderson-Darling).",
        }

    try:
        stat, p = stats.shapiro(diffs)
        return {
            "n": n,
            "statistic": _round_or_none(stat),
            "p_value": _round_or_none(p),
            "normal_at_0_05": bool(p >= 0.05) if math.isfinite(float(p)) else None,
            "note": None,
        }
    except Exception:
        return {
            "n": n,
            "statistic": None,
            "p_value": None,
            "normal_at_0_05": None,
            "note": "Shapiro-Wilk failed.",
        }


# ==========================================================
# Main execute
# ==========================================================

def execute_paired_comparison(context) -> Dict[str, Any]:
    arguments = _get_arguments(context)

    target_col_1 = arguments.get("target_col_1") or arguments.get("col1")
    target_col_2 = arguments.get("target_col_2") or arguments.get("col2")

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

        if not target_col_1 or not target_col_2:
            return _blocked(
                "MISSING_ARGUMENTS",
                "target_col_1 and target_col_2 are required for paired_comparison.",
                details={
                    "target_col_1": target_col_1,
                    "target_col_2": target_col_2,
                },
                suggested_next_actions=[
                    "Specify the two numeric measurement columns (e.g., pre and post) for paired_comparison.",
                ],
            )

        if target_col_1 == target_col_2:
            return _blocked(
                "IDENTICAL_COLUMNS",
                "target_col_1 and target_col_2 must be different columns.",
                details={
                    "target_col_1": target_col_1,
                    "target_col_2": target_col_2,
                },
            )

        missing_cols = [c for c in [target_col_1, target_col_2] if c not in df.columns]

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

        col1 = pd.to_numeric(df[target_col_1], errors="coerce")
        col2 = pd.to_numeric(df[target_col_2], errors="coerce")

        if col1.notna().sum() == 0 or col2.notna().sum() == 0:
            return _blocked(
                "COLUMNS_NOT_NUMERIC",
                "Both measurement columns must contain numeric values.",
                details={
                    "target_col_1": target_col_1,
                    "target_col_2": target_col_2,
                    "dtype_1": str(df[target_col_1].dtype),
                    "dtype_2": str(df[target_col_2].dtype),
                },
            )

        n_rows_total = int(df.shape[0])

        work = pd.DataFrame({
            "col1": col1,
            "col2": col2,
        }).dropna()

        n_pairs = int(work.shape[0])
        n_dropped = n_rows_total - n_pairs

        if n_pairs < 3:
            return _blocked(
                "INSUFFICIENT_PAIRS",
                "Paired comparison requires at least 3 complete pairs after dropping missing values.",
                details={
                    "n_rows_total": n_rows_total,
                    "n_complete_pairs": n_pairs,
                    "n_dropped_pairs": n_dropped,
                },
                suggested_next_actions=[
                    "Inspect missing values; consider missingness_report or clean_data.",
                ],
            )

        diffs = (work["col1"] - work["col2"]).to_numpy(dtype=float)

        # Descriptive summary of each measurement and the difference
        descriptive_rows = [
            {
                "series": str(target_col_1),
                "n": n_pairs,
                "mean": _round_or_none(work["col1"].mean()),
                "sd": _round_or_none(work["col1"].std(ddof=1)),
                "median": _round_or_none(work["col1"].median()),
                "min": _round_or_none(work["col1"].min()),
                "max": _round_or_none(work["col1"].max()),
            },
            {
                "series": str(target_col_2),
                "n": n_pairs,
                "mean": _round_or_none(work["col2"].mean()),
                "sd": _round_or_none(work["col2"].std(ddof=1)),
                "median": _round_or_none(work["col2"].median()),
                "min": _round_or_none(work["col2"].min()),
                "max": _round_or_none(work["col2"].max()),
            },
            {
                "series": f"difference ({target_col_1} - {target_col_2})",
                "n": n_pairs,
                "mean": _round_or_none(float(np.mean(diffs))),
                "sd": _round_or_none(float(np.std(diffs, ddof=1))) if n_pairs > 1 else None,
                "median": _round_or_none(float(np.median(diffs))),
                "min": _round_or_none(float(np.min(diffs))),
                "max": _round_or_none(float(np.max(diffs))),
            },
        ]

        # Run both tests
        paired_t = _run_paired_t_test(diffs, alpha)
        wilcoxon = _run_wilcoxon_signed_rank(diffs, alpha)

        # Normality of differences
        diffs_normality = _run_differences_shapiro(diffs)

        # Decide recommended test
        normality_verdict = diffs_normality.get("normal_at_0_05")
        if normality_verdict is False:
            recommended_test = "wilcoxon_signed_rank"
            primary = wilcoxon
        else:
            # Default to paired t when normality is supported or unknown
            recommended_test = "paired_t_test"
            primary = paired_t

        # Disagreement check (only meaningful when both have p-values)
        agreement = None
        try:
            sig_t = paired_t.get("significant_at_alpha")
            sig_w = wilcoxon.get("significant_at_alpha")
            if sig_t is not None and sig_w is not None:
                agreement = bool(sig_t == sig_w)
        except Exception:
            agreement = None

        # Top-level assumptions
        assumptions_and_limitations: list[str] = [
            "Paired comparison assumes independent observation pairs (between subjects/units).",
            "Each pair represents two measurements of the same subject/unit (e.g., pre-post, matched case-control).",
            "The paired t-test assumes that the differences are approximately normally distributed; the Wilcoxon signed-rank test assumes the differences are symmetrically distributed around the location parameter.",
        ]

        if n_dropped > 0:
            assumptions_and_limitations.append(
                f"{n_dropped} row(s) were dropped because of missing values in one or both columns; "
                "listwise deletion is used."
            )

        if wilcoxon.get("n_zero_difference_pairs_dropped"):
            assumptions_and_limitations.append(
                f"{wilcoxon['n_zero_difference_pairs_dropped']} zero-difference pair(s) were excluded from "
                "the Wilcoxon signed-rank test (Wilcoxon's recommendation)."
            )

        if normality_verdict is False:
            assumptions_and_limitations.append(
                "Shapiro-Wilk indicates the differences deviate from normality; the Wilcoxon signed-rank "
                "result is recommended as the primary inference."
            )
        elif normality_verdict is True:
            assumptions_and_limitations.append(
                "Shapiro-Wilk does not reject normality of differences; the paired t-test is recommended as the primary inference."
            )

        if agreement is False:
            assumptions_and_limitations.append(
                "The paired t-test and Wilcoxon signed-rank test disagree at the chosen alpha. "
                "Report both results and prefer the test whose assumptions are best supported."
            )

        details = {
            "target_col_1": target_col_1,
            "target_col_2": target_col_2,
            "alpha": alpha,
            "n_rows_total": n_rows_total,
            "n_complete_pairs": n_pairs,
            "n_dropped_pairs": n_dropped,
            "descriptive_summary": descriptive_rows,
            "differences_normality": diffs_normality,
            "paired_t_test": paired_t,
            "wilcoxon_signed_rank": wilcoxon,
            "recommended_test": recommended_test,
            "tests_agree_on_significance": agreement,
            "assumptions_and_limitations": assumptions_and_limitations,
            # Promote primary-test fields to the top level for downstream
            # consumers (guardrails, extractor) that expect the standard
            # group-comparison contract.
            "method": primary.get("method"),
            "test_family": primary.get("test_family"),
            "p_value": primary.get("p_value"),
            "significant_at_alpha": primary.get("significant_at_alpha"),
            "significant_at_0_05": primary.get("significant_at_0_05"),
            "effect_size_name": (
                "Cohen's d_z" if recommended_test == "paired_t_test"
                else "matched-pairs rank-biserial r"
            ),
            "effect_size": (
                primary.get("cohens_d_z") if recommended_test == "paired_t_test"
                else primary.get("rank_biserial_correlation")
            ),
            "effect_size_magnitude": primary.get("effect_size_magnitude"),
        }

        return {
            "status": "ok",
            "message": (
                f"Paired comparison completed for {target_col_1} vs {target_col_2}. "
                f"Recommended primary test: {recommended_test}."
            ),
            "recoverable": False,
            "details": details,
            "artifacts": [],
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_code": "PAIRED_COMPARISON_FAILED",
            "message": f"paired_comparison failed: {exc}",
            "recoverable": True,
            "details": {
                "exception_type": type(exc).__name__,
                "error_message": str(exc),
                "received_arguments": arguments,
            },
            "artifacts": [],
        }


# ==========================================================
# Plugin-specific guardrail (inline; co-located with the plugin)
# ==========================================================

def evaluate_paired_comparison_specific_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}
    metadata = run.get("metadata", {}) or {}

    # Reach into the raw payload for full diagnostics
    payload = metadata.get("paired_comparison_payload", {}) or run.get("payload", {}) or {}
    differences_normality = payload.get("differences_normality") or {}
    wilcoxon = payload.get("wilcoxon_signed_rank") or {}
    paired_t = payload.get("paired_t_test") or {}

    recommended_test = payload.get("recommended_test")
    tests_agree = payload.get("tests_agree_on_significance")
    n_complete_pairs = payload.get("n_complete_pairs")
    n_dropped_pairs = payload.get("n_dropped_pairs")

    # Differences non-normal -> trust Wilcoxon
    if differences_normality.get("normal_at_0_05") is False:
        findings.append(_new_finding(
            category="assumption_check",
            severity="warning",
            title="Differences are not normally distributed",
            message=(
                "Shapiro-Wilk on the paired differences rejected normality at 0.05. "
                "For paired-sample inference, the Wilcoxon signed-rank result is the more "
                "appropriate primary test in this dataset."
            ),
            evidence={
                "shapiro_w": differences_normality.get("statistic"),
                "shapiro_p": differences_normality.get("p_value"),
                "n_pairs": differences_normality.get("n"),
            },
            recommendation=(
                "Report the Wilcoxon signed-rank statistic, the matched-pairs rank-biserial "
                "correlation, and the Hodges-Lehmann pseudomedian with its CI as the primary "
                "result; report the paired t-test as a sensitivity check."
            ),
        ))

    # Tests disagree at the chosen alpha
    if tests_agree is False:
        findings.append(_new_finding(
            category="interpretation",
            severity="warning",
            title="Paired t-test and Wilcoxon signed-rank disagree",
            message=(
                "The two tests reach different conclusions at the chosen alpha. Inference is "
                "sensitive to the distributional assumptions."
            ),
            evidence={
                "paired_t_p_value": paired_t.get("p_value"),
                "wilcoxon_p_value": wilcoxon.get("p_value"),
                "recommended_test": recommended_test,
            },
            recommendation=(
                "Report both results. Anchor the conclusion on the test whose assumptions are "
                "better supported by the data; consider bootstrap-based inference if a definitive "
                "answer is required."
            ),
        ))

    # Zero-difference pairs dropped (informational)
    zeros_dropped = wilcoxon.get("n_zero_difference_pairs_dropped", 0) or 0

    try:
        if int(zeros_dropped) > 0:
            findings.append(_new_finding(
                category="data_handling",
                severity="info",
                title=f"{zeros_dropped} zero-difference pair(s) dropped from Wilcoxon",
                message=(
                    "Pairs with a zero difference are excluded from the Wilcoxon signed-rank "
                    "test under the standard 'wilcox' rule. This reduces the effective sample "
                    "size for the nonparametric test but does not affect the paired t-test."
                ),
                evidence={
                    "n_zero_difference_pairs_dropped": zeros_dropped,
                    "n_complete_pairs": n_complete_pairs,
                    "n_pairs_used_by_wilcoxon": wilcoxon.get("n_pairs_used"),
                },
            ))
    except Exception:
        pass

    # Listwise-deleted rows (informational)
    try:
        if int(n_dropped_pairs or 0) > 0:
            findings.append(_new_finding(
                category="data_handling",
                severity="info",
                title=f"{n_dropped_pairs} row(s) dropped due to missing measurements",
                message=(
                    "Listwise deletion was applied: any row missing either measurement was excluded."
                ),
                evidence={
                    "n_dropped_pairs": n_dropped_pairs,
                    "n_complete_pairs": n_complete_pairs,
                },
                recommendation=(
                    "Run missingness_report to verify that missingness is plausibly MCAR; "
                    "otherwise consider multiple imputation or mixed-effects models."
                ),
            ))
    except Exception:
        pass

    # Small-sample warning
    try:
        if n_complete_pairs is not None and int(n_complete_pairs) < 10:
            findings.append(_new_finding(
                category="sample_size",
                severity="warning",
                title=f"Small paired sample (n = {n_complete_pairs})",
                message=(
                    "With fewer than 10 complete pairs, both the paired t-test and the Wilcoxon "
                    "signed-rank test have low power and the confidence intervals are wide."
                ),
                evidence={"n_complete_pairs": n_complete_pairs},
                recommendation=(
                    "Treat the point estimates cautiously and rely on the confidence intervals; "
                    "consider bootstrap-based inference if available."
                ),
            ))
    except Exception:
        pass

    return findings


# ==========================================================
# Extractor / Display
# ==========================================================

def extract_paired_comparison(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    target_col_1 = payload.get("target_col_1") or arguments.get("target_col_1")
    target_col_2 = payload.get("target_col_2") or arguments.get("target_col_2")

    title = "Paired Comparison"
    if target_col_1 and target_col_2:
        title = f"Paired Comparison: {target_col_1} vs {target_col_2}"

    paired_t = payload.get("paired_t_test") or {}
    wilcoxon = payload.get("wilcoxon_signed_rank") or {}
    diffs_normality = payload.get("differences_normality") or {}
    recommended_test = payload.get("recommended_test")

    significance_phrase = ""

    sig = payload.get("significant_at_alpha")
    if sig is True:
        significance_phrase = " The paired difference is statistically significant at the selected alpha level."
    elif sig is False:
        significance_phrase = " The paired difference is not statistically significant at the selected alpha level."

    summary = (
        f"Compared paired measurements `{target_col_1}` and `{target_col_2}` with "
        f"n = {payload.get('n_complete_pairs')} complete pairs."
        f" Recommended primary test: {recommended_test}."
        f"{significance_phrase}"
    )

    if payload.get("tests_agree_on_significance") is False:
        summary += " (Note: the paired t-test and Wilcoxon disagree at the chosen alpha.)"

    metrics = compact_dict({
        "method": payload.get("method"),
        "recommended_test": recommended_test,
        "alpha": payload.get("alpha"),
        "n_complete_pairs": payload.get("n_complete_pairs"),
        "n_dropped_pairs": payload.get("n_dropped_pairs"),
        "p_value": payload.get("p_value"),
        "significant_at_alpha": payload.get("significant_at_alpha"),
        "significant_at_0_05": payload.get("significant_at_0_05"),
        "effect_size_name": payload.get("effect_size_name"),
        "effect_size": payload.get("effect_size"),
        "effect_size_magnitude": payload.get("effect_size_magnitude"),
        # Paired t-test details
        "t_statistic": paired_t.get("t_statistic"),
        "degrees_of_freedom": paired_t.get("degrees_of_freedom"),
        "mean_difference": paired_t.get("mean_difference"),
        "mean_difference_ci_low": paired_t.get("mean_difference_ci_low"),
        "mean_difference_ci_high": paired_t.get("mean_difference_ci_high"),
        "paired_t_p_value": paired_t.get("p_value"),
        "cohens_d_z": paired_t.get("cohens_d_z"),
        "cohens_d_z_ci_low": paired_t.get("cohens_d_z_ci_low"),
        "cohens_d_z_ci_high": paired_t.get("cohens_d_z_ci_high"),
        # Wilcoxon details
        "W_statistic": wilcoxon.get("W_statistic"),
        "wilcoxon_p_value": wilcoxon.get("p_value"),
        "rank_biserial_correlation": wilcoxon.get("rank_biserial_correlation"),
        "hodges_lehmann_pseudomedian": wilcoxon.get("hodges_lehmann_pseudomedian"),
        "hodges_lehmann_ci_low": wilcoxon.get("hodges_lehmann_ci_low"),
        "hodges_lehmann_ci_high": wilcoxon.get("hodges_lehmann_ci_high"),
        "n_zero_difference_pairs_dropped": wilcoxon.get("n_zero_difference_pairs_dropped"),
        # Normality
        "differences_shapiro_w": diffs_normality.get("statistic"),
        "differences_shapiro_p": diffs_normality.get("p_value"),
        "differences_normal_at_0_05": diffs_normality.get("normal_at_0_05"),
        "tests_agree_on_significance": payload.get("tests_agree_on_significance"),
    })

    tables: Dict[str, Any] = {}

    if payload.get("descriptive_summary"):
        tables["descriptive_summary"] = payload.get("descriptive_summary")

    if payload.get("assumptions_and_limitations"):
        tables["assumptions_and_limitations"] = [
            {"item": item}
            for item in payload.get("assumptions_and_limitations", [])
        ]

    metadata = compact_dict({
        "target_col_1": target_col_1,
        "target_col_2": target_col_2,
        "recommended_test": recommended_test,
        # Stash full payload so the plugin-specific guardrail can introspect
        "paired_comparison_payload": payload,
    })

    return title, summary, metrics, tables, metadata


PAIRED_COMPARISON_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method (primary)",
            "recommended_test": "Recommended primary test",
            "alpha": "Alpha",
            "n_complete_pairs": "Complete pairs",
            "n_dropped_pairs": "Pairs dropped (listwise)",
            "p_value": "p-value (primary)",
            "significant_at_alpha": "Significant",
            "significant_at_0_05": "Significant at 0.05",
            "effect_size_name": "Effect size",
            "effect_size": "Effect size value",
            "effect_size_magnitude": "Effect size magnitude",
            "t_statistic": "t statistic",
            "degrees_of_freedom": "Degrees of freedom",
            "mean_difference": "Mean difference",
            "mean_difference_ci_low": "Mean diff CI lower",
            "mean_difference_ci_high": "Mean diff CI upper",
            "paired_t_p_value": "Paired t p-value",
            "cohens_d_z": "Cohen's d_z",
            "cohens_d_z_ci_low": "d_z CI lower",
            "cohens_d_z_ci_high": "d_z CI upper",
            "W_statistic": "Wilcoxon W",
            "wilcoxon_p_value": "Wilcoxon p-value",
            "rank_biserial_correlation": "Rank-biserial r (matched)",
            "hodges_lehmann_pseudomedian": "Hodges-Lehmann pseudomedian",
            "hodges_lehmann_ci_low": "HL pseudomedian CI lower",
            "hodges_lehmann_ci_high": "HL pseudomedian CI upper",
            "n_zero_difference_pairs_dropped": "Zero-diff pairs dropped (Wilcoxon)",
            "differences_shapiro_w": "Differences Shapiro W",
            "differences_shapiro_p": "Differences Shapiro p-value",
            "differences_normal_at_0_05": "Differences normal at 0.05",
            "tests_agree_on_significance": "Paired t and Wilcoxon agree",
        },
        formatters={
            "p_value": format_p_value,
            "significant_at_alpha": format_bool_yes_no,
            "significant_at_0_05": format_bool_yes_no,
            "effect_size": format_number,
            "t_statistic": format_number,
            "mean_difference": format_number,
            "mean_difference_ci_low": format_number,
            "mean_difference_ci_high": format_number,
            "paired_t_p_value": format_p_value,
            "cohens_d_z": format_number,
            "cohens_d_z_ci_low": format_number,
            "cohens_d_z_ci_high": format_number,
            "W_statistic": format_number,
            "wilcoxon_p_value": format_p_value,
            "rank_biserial_correlation": format_number,
            "hodges_lehmann_pseudomedian": format_number,
            "hodges_lehmann_ci_low": format_number,
            "hodges_lehmann_ci_high": format_number,
            "differences_shapiro_w": format_number,
            "differences_shapiro_p": format_p_value,
            "differences_normal_at_0_05": format_bool_yes_no,
            "tests_agree_on_significance": format_bool_yes_no,
        },
        order=[
            "method",
            "recommended_test",
            "alpha",
            "n_complete_pairs",
            "n_dropped_pairs",
            "p_value",
            "significant_at_alpha",
            "significant_at_0_05",
            "effect_size_name",
            "effect_size",
            "effect_size_magnitude",
            "t_statistic",
            "degrees_of_freedom",
            "mean_difference",
            "mean_difference_ci_low",
            "mean_difference_ci_high",
            "paired_t_p_value",
            "cohens_d_z",
            "cohens_d_z_ci_low",
            "cohens_d_z_ci_high",
            "W_statistic",
            "wilcoxon_p_value",
            "rank_biserial_correlation",
            "hodges_lehmann_pseudomedian",
            "hodges_lehmann_ci_low",
            "hodges_lehmann_ci_high",
            "n_zero_difference_pairs_dropped",
            "differences_shapiro_w",
            "differences_shapiro_p",
            "differences_normal_at_0_05",
            "tests_agree_on_significance",
        ],
    ),
    tables={
        "descriptive_summary": TableDisplayConfig(
            column_labels={
                "series": "Series",
                "n": "N",
                "mean": "Mean",
                "sd": "SD",
                "median": "Median",
                "min": "Min",
                "max": "Max",
            },
            column_order=[
                "series",
                "n",
                "mean",
                "sd",
                "median",
                "min",
                "max",
            ],
            column_formatters={
                "mean": format_number,
                "sd": format_number,
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
    tool_name="paired_comparison",
    display_name="Paired Comparison (t-test + Wilcoxon)",
    is_inferential=True,
    evidence_categories=["group_comparison", "statistical_inference"],
    description=(
        "Paired-sample comparison of two numeric measurements on the same units. "
        "Runs both the paired-samples t-test and the Wilcoxon signed-rank test, "
        "and recommends a primary test based on the Shapiro-Wilk normality check "
        "of the differences. Reports mean difference with CI, Cohen's d_z with a "
        "noncentral-t-based CI, the matched-pairs rank-biserial correlation, and "
        "the Hodges-Lehmann pseudomedian with its distribution-free CI."
    ),
    usage_guidance=(
        "Use this for within-subject / matched designs (pre-post studies, "
        "matched case-control pairs, repeated measurements on the same unit). "
        "Requires the data to be in wide format: one row per subject/unit, with "
        "two columns holding the two measurements. If the data are in long format "
        "(one row per measurement with a subject/time identifier), pivot first "
        "with clean_data."
    ),
    use_when=[
        "The user asks to compare two measurements on the same units (e.g., pre vs post).",
        "The dataset has two numeric columns representing matched observations.",
        "The user asks for a paired t-test or Wilcoxon signed-rank test.",
    ],
    do_not_use_when=[
        "No active DataFrame dataset exists.",
        "The two measurements belong to independent groups; use statistical_group_comparison or run_independent_t_test instead.",
        "The dataset is in long format with subject/time identifiers; pivot to wide format first.",
    ],
    requires_data_source="dataframe",
    produces_active_dataset=False,
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={
            "target_col_1": str,
            "target_col_2": str,
        },
        optional={
            "alpha": float,
        },
        column_args=[
            "target_col_1",
            "target_col_2",
        ],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_paired_comparison,
    extractor=extract_paired_comparison,
    guardrail_evaluators=[
        evaluate_group_comparison_guardrails,
        evaluate_paired_comparison_specific_guardrails,
    ],
    apa_methods_writer=write_apa_paired_comparison,
    display_config=PAIRED_COMPARISON_DISPLAY,
    examples=[
        {
            "user_request": "Compare pre and post test scores for the same students.",
            "arguments": {
                "target_col_1": "pre_score",
                "target_col_2": "post_score",
            },
        },
        {
            "user_request": "Did weight change after the intervention? Compare weight_before and weight_after.",
            "arguments": {
                "target_col_1": "weight_before",
                "target_col_2": "weight_after",
            },
        },
    ],
))