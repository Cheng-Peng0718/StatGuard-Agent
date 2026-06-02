"""
bootstrap_inference: bootstrap confidence intervals for paired-difference
statistics, with an optional Sequential Bootstrap mode and a built-in
cross-seed CI stability diagnostic.

When the LLM should route to this plugin
----------------------------------------
This plugin fills a specific inferential regime that the existing
`paired_comparison` plugin does not cover:

  - The user wants a confidence interval on the *mean* difference even
    though the differences are non-normal. `paired_comparison` in that
    regime switches to the Wilcoxon signed-rank test, whose distribution-free
    CI is on the Hodges-Lehmann pseudomedian, not the mean. When the user
    needs a CI on the original mean-difference scale, bootstrap is the
    appropriate method.
  - The user wants a CI on a non-standard paired statistic (median diff,
    trimmed mean diff, Cohen's d_z), for which no closed-form CI is
    standard.
  - The user wants the CI itself to be reproducible across bootstrap RNG
    seeds, e.g., for regulatory submissions or clinical reporting. In
    that case `use_sequential=True` stabilises the resampler-side variance
    (Peng 2025, arXiv:2511.18065).

The plugin always reports a cross-seed CI stability diagnostic alongside
the primary CI, regardless of which resampler is used. The diagnostic is
the operational version of the variance decomposition in Peng (2025),
Section 3.4.
"""

from __future__ import annotations

from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.bootstrap_utils import (
    PAIRED_STATISTICS,
    bootstrap_with_stability,
    expected_kn,
    get_paired_statistic,
)


# ==========================================================
# Result helpers (match other plugins' status contract)
# ==========================================================

def _ok(message: str, details: Dict[str, Any], artifacts=None) -> Dict[str, Any]:
    return {
        "status": "ok",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": artifacts or [],
    }


def _blocked(error_code: str, message: str, details: Dict[str, Any] = None) -> Dict[str, Any]:
    return {
        "status": "blocked",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": details or {},
        "artifacts": [],
    }


def _failed(error_code: str, message: str, exc: Exception) -> Dict[str, Any]:
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


# ==========================================================
# Argument parsing
# ==========================================================

def _get_argument(context, name: str, default=None):
    args = getattr(context, "arguments", None) or getattr(context, "args", None) or {}
    if hasattr(context, "get_arg"):
        return context.get_arg(name, default)
    return args.get(name, default)


def _coerce_paired_differences(
    df: pd.DataFrame,
    col_1: str,
    col_2: str,
) -> Tuple[np.ndarray, int, int]:
    """Compute paired differences d = col_2 - col_1 with NaN-pair dropping."""
    if col_1 not in df.columns:
        raise KeyError(f"Column '{col_1}' not found in active dataset.")
    if col_2 not in df.columns:
        raise KeyError(f"Column '{col_2}' not found in active dataset.")

    a = pd.to_numeric(df[col_1], errors="coerce").to_numpy(dtype=float)
    b = pd.to_numeric(df[col_2], errors="coerce").to_numpy(dtype=float)

    n_rows_total = len(a)

    mask = np.isfinite(a) & np.isfinite(b)

    diffs = b[mask] - a[mask]
    n_pairs = int(mask.sum())

    return diffs, n_rows_total, n_pairs


# ==========================================================
# Execute
# ==========================================================

def execute_bootstrap_inference(context) -> Dict[str, Any]:
    """Compute bootstrap CI and stability diagnostic for paired differences."""
    try:
        # ----- load data -----
        df = context.load_df() if hasattr(context, "load_df") else None

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        # ----- read arguments -----
        col_1 = _get_argument(context, "target_col_1")
        col_2 = _get_argument(context, "target_col_2")

        if not col_1 or not col_2:
            return _blocked(
                "MISSING_ARGUMENTS",
                "target_col_1 and target_col_2 are required for "
                "bootstrap_inference. Specify the two paired measurement "
                "columns (e.g., pre and post).",
            )

        statistic_name = str(_get_argument(context, "statistic", "mean_diff"))

        if statistic_name not in PAIRED_STATISTICS:
            return _blocked(
                "UNKNOWN_STATISTIC",
                f"Unknown statistic '{statistic_name}'. Choose one of: "
                f"{sorted(PAIRED_STATISTICS)}.",
            )

        ci_method = str(_get_argument(context, "ci_method", "BCa"))

        if ci_method.lower() not in {"bca", "percentile", "basic"}:
            return _blocked(
                "UNKNOWN_CI_METHOD",
                f"Unknown ci_method '{ci_method}'. Choose one of: BCa, percentile, basic.",
            )

        B = int(_get_argument(context, "B", 5000))
        n_seeds = int(_get_argument(context, "n_seeds", 5))
        alpha = float(_get_argument(context, "alpha", 0.05))
        rho = float(_get_argument(context, "rho", 0.632))

        use_sequential_raw = _get_argument(context, "use_sequential", False)
        use_sequential = bool(use_sequential_raw) if use_sequential_raw is not None else False

        seed = int(_get_argument(context, "seed", 0))

        if not (0.0 < alpha < 1.0):
            return _blocked("INVALID_ALPHA", f"alpha must be in (0, 1); got {alpha}.")
        if B < 100:
            return _blocked(
                "INVALID_B",
                f"B (bootstrap replicates) should be >= 100 for usable CIs; got {B}.",
            )
        if n_seeds < 2:
            return _blocked(
                "INVALID_N_SEEDS",
                "n_seeds must be >= 2 to compute the cross-seed stability "
                f"diagnostic; got {n_seeds}.",
            )

        # ----- compute paired differences -----
        diffs, n_rows_total, n_pairs = _coerce_paired_differences(df, col_1, col_2)

        if n_pairs < 8:
            return _blocked(
                "INSUFFICIENT_PAIRS",
                "bootstrap_inference requires at least 8 complete pairs "
                "after dropping missing values; the BCa jackknife is unstable "
                f"below that threshold. Got n_pairs={n_pairs}.",
                details={
                    "n_rows_total": n_rows_total,
                    "n_complete_pairs": n_pairs,
                },
            )

        # ----- run bootstrap -----
        statistic_fn = get_paired_statistic(statistic_name)

        result = bootstrap_with_stability(
            diffs,
            statistic_fn,
            B=B,
            n_seeds=n_seeds,
            alpha=alpha,
            method=ci_method,
            use_sequential=use_sequential,
            rho=rho,
            seed=seed,
        )

        # ----- assemble details -----
        diag = result["stability_diagnostic"]

        details: Dict[str, Any] = {
            "tool": "bootstrap_inference",
            "target_col_1": col_1,
            "target_col_2": col_2,
            "statistic": statistic_name,
            "ci_method": ci_method,
            "alpha": alpha,
            "confidence_level": 1.0 - alpha,
            "B_total": result["B_total"],
            "n_seeds_for_diagnostic": result["n_seeds_for_diagnostic"],
            "B_per_seed": result["B_per_seed"],
            "resampler": result["resampler"],
            "use_sequential": use_sequential,
            "rho": rho if use_sequential else None,
            "k_n": expected_kn(n_pairs, rho=rho) if use_sequential else None,
            "n_rows_total": n_rows_total,
            "n_complete_pairs": n_pairs,
            "observed_statistic": result["observed_statistic"],
            "ci_lower": result["ci_lower"],
            "ci_upper": result["ci_upper"],
            "ci_width": float(result["ci_upper"] - result["ci_lower"]),
            "stability_diagnostic": {
                "endpoint_drift": diag["endpoint_drift"],
                "ci_lower_sd": diag["ci_lower_sd"],
                "ci_upper_sd": diag["ci_upper_sd"],
                "ci_lower_cv": diag["ci_lower_cv"],
                "ci_upper_cv": diag["ci_upper_cv"],
                "interpretation": diag["interpretation"],
                "recommendation": diag["recommendation"],
            },
        }

        # ----- summary message -----
        ci_width_str = format_number(details["ci_width"])
        message = (
            f"Bootstrap {int(100 * (1 - alpha))}% CI for {statistic_name} "
            f"({result['resampler']} bootstrap, B={result['B_total']}): "
            f"[{format_number(result['ci_lower'])}, "
            f"{format_number(result['ci_upper'])}] "
            f"(width {ci_width_str}). "
            f"Cross-seed CI stability: {diag['interpretation']}."
        )

        return _ok(message, details)

    except KeyError as e:
        return _blocked("COLUMN_NOT_FOUND", str(e))

    except ValueError as e:
        return _blocked("INVALID_INPUT", str(e))

    except Exception as e:  # pragma: no cover -- safety net
        return _failed(
            "BOOTSTRAP_INFERENCE_EXCEPTION",
            "bootstrap_inference failed.",
            e,
        )


# ==========================================================
# Extractor (report-builder API)
# ==========================================================

def extract_bootstrap_inference(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    statistic = payload.get("statistic", "")
    resampler = payload.get("resampler", "classical")

    title = f"Bootstrap CI ({resampler}) — {statistic}"

    metrics = compact_dict({
        "statistic": payload.get("statistic"),
        "observed_statistic": payload.get("observed_statistic"),
        "ci_method": payload.get("ci_method"),
        "ci_lower": payload.get("ci_lower"),
        "ci_upper": payload.get("ci_upper"),
        "ci_width": payload.get("ci_width"),
        "confidence_level": payload.get("confidence_level"),
        "B_total": payload.get("B_total"),
        "n_complete_pairs": payload.get("n_complete_pairs"),
        "resampler": payload.get("resampler"),
        "endpoint_drift": (
            payload.get("stability_diagnostic", {}) or {}
        ).get("endpoint_drift"),
        "stability_interpretation": (
            payload.get("stability_diagnostic", {}) or {}
        ).get("interpretation"),
    })

    tables: Dict[str, Any] = {}

    diag = payload.get("stability_diagnostic", {}) or {}

    if diag:
        tables["stability_diagnostic"] = {
            "endpoint_drift": diag.get("endpoint_drift"),
            "ci_lower_sd": diag.get("ci_lower_sd"),
            "ci_upper_sd": diag.get("ci_upper_sd"),
            "ci_lower_cv": diag.get("ci_lower_cv"),
            "ci_upper_cv": diag.get("ci_upper_cv"),
            "interpretation": diag.get("interpretation"),
            "recommendation": diag.get("recommendation"),
        }

    metadata: Dict[str, Any] = {
        "use_sequential": payload.get("use_sequential", False),
        "rho": payload.get("rho"),
        "k_n": payload.get("k_n"),
    }

    summary = (
        f"Bootstrap {int(100 * (1 - (payload.get('alpha') or 0.05)))}% CI for "
        f"{statistic} using {resampler} bootstrap. "
        f"Cross-seed CI stability: {diag.get('interpretation', 'undefined')}."
    )

    return title, summary, metrics, tables, metadata


# ==========================================================
# Display config
# ==========================================================

BOOTSTRAP_INFERENCE_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "statistic": "Statistic",
            "observed_statistic": "Point estimate",
            "ci_method": "CI method",
            "ci_lower": "CI lower",
            "ci_upper": "CI upper",
            "ci_width": "CI width",
            "confidence_level": "Confidence level",
            "B_total": "Bootstrap replicates",
            "n_complete_pairs": "Complete pairs",
            "resampler": "Resampler",
            "endpoint_drift": "Cross-seed endpoint drift (fraction of CI width)",
            "stability_interpretation": "CI stability",
        },
        order=[
            "statistic",
            "observed_statistic",
            "ci_method",
            "ci_lower",
            "ci_upper",
            "ci_width",
            "confidence_level",
            "B_total",
            "n_complete_pairs",
            "resampler",
            "endpoint_drift",
            "stability_interpretation",
        ],
    ),
)


# ==========================================================
# Plugin registration
# ==========================================================

PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="bootstrap_inference",
    display_name="Bootstrap CI (Paired) with Cross-Seed Stability Diagnostic",
    description=(
        "Bootstrap confidence intervals for paired-difference statistics, "
        "with an optional Sequential Bootstrap mode (Peng 2025, "
        "arXiv:2511.18065) and a built-in cross-seed CI stability diagnostic. "
        "Default resampler is the classical multinomial bootstrap. "
        "Set use_sequential=True to hold the number of distinct training "
        "observations fixed in each replicate at k_n = floor(0.632 * n_pairs); "
        "this stabilises the CI endpoints across bootstrap RNG seeds, which "
        "matters for regulatory or clinical reporting where the CI itself "
        "must be reproducible. The plugin always reports a cross-seed CI "
        "endpoint CV, regardless of which resampler is used."
    ),
    usage_guidance=(
        "Supported paired statistics: mean_diff (default), median_diff, "
        "trimmed_mean_diff (10% symmetric trim), cohens_dz. Supported CI "
        "methods: BCa (default, bias-corrected and accelerated, requires "
        "n_pairs >= 8 for stable jackknife), percentile, basic (Hall). "
        "Use the default classical bootstrap (use_sequential=False) for "
        "routine analysis. "
        "SET use_sequential=True whenever the user's request mentions ANY "
        "of the following contexts, because each one implies the CI must "
        "be numerically reproducible if the analysis is re-run: "
        "(a) regulatory submission of any kind -- FDA, EMA, MHRA, NMPA, "
        "PMDA, or generic 'regulatory'; "
        "(b) clinical report, clinical study, clinical trial, or any "
        "reporting destined for a clinician or clinical audience; "
        "(c) the user describes the analysis as audit-grade, auditable, "
        "or for an audit trail; "
        "(d) IRB / ethics submission, peer-reviewed publication, or "
        "the user expects the analysis to be re-run by a third party; "
        "(e) the user explicitly asks for a stable / reproducible CI, "
        "or says the same numbers must appear on re-runs. "
        "If any of these contexts are present, use_sequential=True is "
        "the right choice even if the user does not literally say "
        "'use Sequential Bootstrap'. "
        "The plugin always reports a cross-seed CI endpoint drift under "
        "the key 'stability_diagnostic.endpoint_drift'; values >= 0.05 "
        "are flagged as 'high'."
    ),
    use_when=[
        "User wants a confidence interval on the mean difference and the differences are non-normal "
        "(paired_comparison would switch to Wilcoxon, whose CI is on the pseudomedian, not the mean).",
        "User wants a confidence interval on a non-standard paired statistic such as median difference, "
        "trimmed mean difference, or Cohen's d_z under non-normal differences.",
        "User explicitly requests a bootstrap confidence interval.",
        "regression_diagnostics flagged non-normal residuals together with heteroskedasticity, and the "
        "user wants robust paired-style inference rather than HC robust standard errors.",
        # ---- Sequential Bootstrap triggers (set use_sequential=True) ----
        "The analysis is for an FDA, EMA, MHRA, or any regulatory submission -- set use_sequential=True.",
        "The analysis is for a clinical report, clinical study, or clinical trial -- set use_sequential=True.",
        "The user describes the analysis as audit-grade, auditable, or for an audit trail -- set use_sequential=True.",
        "The user expects the analysis to be re-run by a third party (regulators, reviewers, IRB, auditors) -- set use_sequential=True.",
        "The user explicitly asks for a stable / reproducible CI, or for numbers that will be the same on re-runs -- set use_sequential=True.",
    ],
    do_not_use_when=[
        "The differences are approximately normal and the user only needs the standard paired-t mean-difference CI "
        "(use paired_comparison instead; it is faster and gives a closed-form CI).",
        "The user only asked for a hypothesis test (a p-value) and not a confidence interval "
        "(use paired_comparison; it runs paired t and Wilcoxon directly).",
        "The data are not paired (no within-subject / matched structure); use statistical_group_comparison "
        "for independent groups.",
        "Sample size is very small (n_pairs < 8); BCa jackknife is unstable below this threshold.",
    ],
    requires_data_source="dataframe",
    requires_confirmation=False,
    is_inferential=True,
    argument_schema=ArgumentSchema(
        required={
            "target_col_1": str,
            "target_col_2": str,
        },
        optional={
            "statistic": str,
            "ci_method": str,
            "B": int,
            "n_seeds": int,
            "alpha": float,
            "use_sequential": bool,
            "rho": float,
            "seed": int,
        },
        column_args=["target_col_1", "target_col_2"],
        column_list_args=[],
        allow_all_columns=False,
    ),
    examples=[
        {
            "scenario": "Bootstrap 95% CI for the mean weight change after an intervention.",
            "arguments": {
                "target_col_1": "weight_before",
                "target_col_2": "weight_after",
                "statistic": "mean_diff",
            },
        },
        {
            "scenario": (
                "Audit-grade CI for the median pre-post difference in a clinical study; "
                "the CI must be reproducible across bootstrap seeds."
            ),
            "arguments": {
                "target_col_1": "score_pre",
                "target_col_2": "score_post",
                "statistic": "median_diff",
                "use_sequential": True,
            },
        },
        {
            "scenario": (
                "FDA regulatory submission: bootstrap CI for the mean change in "
                "primary endpoint. Regulators will re-run the analysis and the "
                "reported CI must match numerically."
            ),
            "arguments": {
                "target_col_1": "endpoint_baseline",
                "target_col_2": "endpoint_week12",
                "statistic": "mean_diff",
                "use_sequential": True,
            },
        },
    ],
    evidence_categories=["paired_inference", "confidence_interval"],
    evidence_category_roles={
        "paired_inference": "substantive",
        "confidence_interval": "substantive",
    },
    execute=execute_bootstrap_inference,
    extractor=extract_bootstrap_inference,
    guardrail_evaluators=[],
    display_config=BOOTSTRAP_INFERENCE_DISPLAY,
))