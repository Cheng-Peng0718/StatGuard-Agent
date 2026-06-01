"""
Logistic regression (binary outcome) analysis tool plugin.

Design notes
------------
This plugin deliberately reuses existing shared infrastructure rather than
re-implementing it:

  * Data preparation (missing-token normalization, id-like column dropping,
    categorical one-hot encoding, zero-variance dropping, effective-sample
    bookkeeping) is delegated to
    ``core.analysis_tool_plugins.shared.regression_utils.prepare_regression_data``
    -- exactly the same function the linear model uses.
  * Cross-plugin statistical guardrails live in
    ``core.analysis_tool_plugins.shared.logistic_guardrails`` so they can be
    shared by any future binary-outcome model.

What this plugin adds that linear regression does NOT have:

  * A *binary outcome gate*. ``prepare_regression_data`` happily accepts any
    numeric column as a continuous outcome, which is wrong for logistic
    regression. We coerce/validate the outcome to exactly two classes (mapping
    them to 0/1) BEFORE fitting, and block clearly otherwise.
  * Detection of (quasi-)complete separation and a convergence check.
  * Events-per-variable (EPV) and class-balance bookkeeping, which the shared
    guardrails turn into reviewer-facing findings.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import statsmodels.api as sm

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    format_number,
    format_p_value,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.regression_utils import prepare_regression_data
from core.analysis_tool_plugins.shared.logistic_guardrails import (
    evaluate_logistic_guardrails,
    SEPARATION_ODDS_RATIO_FLAG,
)


# ==========================================================
# Small local result helpers (mirror the linear_model conventions)
# ==========================================================

def _ok(message: str, details: Dict[str, Any], artifacts=None) -> Dict[str, Any]:
    out = {"status": "ok", "message": message, "recoverable": False, "details": details}
    if artifacts is not None:
        out["artifacts"] = artifacts
    return out


def _blocked(error_code: str, message: str, details: Optional[Dict[str, Any]] = None,
             suggested_next_actions: Optional[List[str]] = None) -> Dict[str, Any]:
    out = {
        "status": "blocked",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": details or {},
    }
    if suggested_next_actions:
        out["suggested_next_actions"] = suggested_next_actions
    return out


def _failed(error_code: str, message: str, exc: Exception) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error_code": error_code,
        "message": f"{message}: {exc}",
        "recoverable": False,
        "details": {"exception_type": type(exc).__name__},
    }


def _get_arg(context, name: str, default: Any = None) -> Any:
    getter = getattr(context, "get_arg", None)
    if callable(getter):
        return getter(name, default)
    args = getattr(context, "arguments", {}) or {}
    return args.get(name, default)


def _safe_float(value: Any) -> Optional[float]:
    try:
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    except Exception:
        return None


# ==========================================================
# Binary outcome gate
# ==========================================================

def _coerce_binary_outcome(
    y_raw: pd.Series,
    positive_label: Any = None,
) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    """
    Validate that the outcome has exactly two distinct non-missing values and
    map it to 0/1.

    Returns (mapped_series_or_None, info_dict). When mapping fails, the series
    is None and info_dict carries the reason for a clear block.
    """
    s = y_raw.dropna()
    classes = sorted([c for c in s.unique()], key=lambda x: str(x))
    n_classes = len(classes)

    if n_classes < 2:
        return None, {"reason": "outcome_not_variable", "n_classes": n_classes, "classes": classes}
    if n_classes > 2:
        return None, {"reason": "outcome_not_binary", "n_classes": n_classes, "classes": classes[:10]}

    # Decide which class is the positive (1) class.
    if positive_label is not None and positive_label in classes:
        positive = positive_label
    else:
        # Default: if classes look like {0,1} keep them; else the larger by
        # string order is positive, but prefer common positive tokens.
        positive_tokens = {"1", "yes", "true", "y", "positive", "pos", "case", "event"}
        lowered = {str(c).strip().lower(): c for c in classes}
        positive = None
        for tok in positive_tokens:
            if tok in lowered:
                positive = lowered[tok]
                break
        if positive is None:
            # Fall back to numeric max if numeric, else last by string order.
            try:
                positive = max(classes, key=lambda x: float(x))
            except Exception:
                positive = classes[-1]

    negative = [c for c in classes if c != positive][0]
    mapped = y_raw.map(lambda v: 1 if v == positive else (0 if v == negative else np.nan))

    info = {
        "positive_class": positive,
        "negative_class": negative,
        "classes": classes,
    }
    return mapped, info


# ==========================================================
# VIF (reuse statsmodels; mirrors regression_diagnostics behavior)
# ==========================================================

def _compute_vif(X: pd.DataFrame) -> List[Dict[str, Any]]:
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
    except Exception:
        return []

    if X.shape[1] < 2:
        return []

    X_const = sm.add_constant(X, has_constant="add")
    cols = list(X_const.columns)
    rows: List[Dict[str, Any]] = []
    arr = X_const.values
    for i, col in enumerate(cols):
        if col == "const":
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                v = variance_inflation_factor(arr, i)
            rows.append({"term": col, "vif": _safe_float(v)})
        except Exception:
            rows.append({"term": col, "vif": None})
    return rows


# ==========================================================
# Execute
# ==========================================================

def execute_logistic_regression(context) -> Dict[str, Any]:
    """
    Fit a binary logistic regression (logit link) via maximum likelihood.

    Args:
        target_col: binary outcome column (exactly two classes)
        feature_cols: list of predictor columns
        positive_label: optional; which outcome value is the modeled "1"
        alpha: optional, default 0.05
        plus the usual prepare_regression_data tuning args.
    """
    try:
        df = context.load_df()

        target_col = _get_arg(context, "target_col")
        feature_cols = _get_arg(context, "feature_cols", []) or []
        positive_label = _get_arg(context, "positive_label", None)

        try:
            alpha = float(_get_arg(context, "alpha", 0.05))
        except Exception:
            alpha = 0.05
        if not (0 < alpha < 1):
            alpha = 0.05

        if not target_col:
            return _blocked("MISSING_TARGET", "target_col is required for logistic regression.",
                            suggested_next_actions=["Ask the user to specify a binary outcome variable."])
        if not isinstance(feature_cols, list) or not feature_cols:
            return _blocked("MISSING_FEATURES", "feature_cols must be a non-empty list.",
                            suggested_next_actions=["Ask the user to specify predictors."])

        # --- Binary outcome gate, BEFORE the shared numeric-outcome prep. ---
        if target_col not in df.columns:
            return _blocked("COLUMNS_NOT_FOUND", f"Outcome column not found: {target_col}",
                            details={"missing_cols": [target_col]})

        mapped_y, binfo = _coerce_binary_outcome(df[target_col], positive_label=positive_label)
        if mapped_y is None:
            reason = binfo.get("reason")
            if reason == "outcome_not_binary":
                return _blocked(
                    "OUTCOME_NOT_BINARY",
                    f"Logistic regression requires a binary outcome, but '{target_col}' has "
                    f"{binfo.get('n_classes')} distinct values.",
                    details=binfo,
                    suggested_next_actions=[
                        "Choose a binary outcome",
                        "Dichotomize the outcome first (e.g. via clean_data)",
                        "Use linear regression (run_multiple_regression) for a continuous outcome",
                    ],
                )
            return _blocked(
                "OUTCOME_NOT_VARIABLE",
                f"Outcome column '{target_col}' does not have two distinct classes.",
                details=binfo,
                suggested_next_actions=["Choose an outcome with both classes present."],
            )

        # Reuse the shared preprocessing for predictors. We feed the *mapped*
        # 0/1 outcome through a temporary frame so prep handles X exactly like
        # the linear model (encoding, id drop, zero-variance drop, sample size).
        work = df.copy()
        work[target_col] = mapped_y
        prep_kwargs = dict(
            max_missing_rate=float(_get_arg(context, "max_missing_rate", 0.40)),
            max_categorical_levels=int(_get_arg(context, "max_categorical_levels", 10)),
            numeric_parse_threshold=float(_get_arg(context, "numeric_parse_threshold", 0.85)),
            # For logistic, min sample is governed by EPV; keep prep's guard mild
            # and let the EPV guardrail do the statistical talking.
            min_n_per_parameter=int(_get_arg(context, "min_n_per_parameter", 1)),
        )
        prep = prepare_regression_data(work, target_col, feature_cols, **prep_kwargs)

        # The shared prep heuristically drops "id-like" (high-uniqueness) columns.
        # That is sensible for accidental identifier predictors, but it also
        # discards legitimate high-cardinality continuous predictors. If id-like
        # dropping is the *only* reason no predictors remain, retry once without
        # it so genuine continuous predictors are not silently lost.
        if prep.get("status") == "blocked" and prep.get("error_code") == "NO_USABLE_PREDICTORS":
            excluded = prep.get("details", {}).get("excluded_features", []) or []
            reasons = {e.get("reason") for e in excluded}
            if reasons and reasons.issubset({"id_like"}):
                prep = prepare_regression_data(
                    work, target_col, feature_cols, drop_id_like=False, **prep_kwargs
                )

        if prep.get("status") != "ok":
            return prep

        y = prep["y"].astype(int)
        X = prep["X"]

        # Outcome must still be binary after complete-case filtering.
        y_classes = sorted(y.unique().tolist())
        if len(y_classes) < 2:
            return _blocked(
                "OUTCOME_NOT_VARIABLE_AFTER_FILTERING",
                "After removing rows with missing predictors, the outcome has only one class.",
                details={"remaining_classes": y_classes},
                suggested_next_actions=["Reduce predictors or inspect missingness patterns."],
            )

        n_obs = int(len(y))
        n_pos = int(y.sum())
        n_neg = n_obs - n_pos
        n_events = min(n_pos, n_neg)
        n_predictors = int(X.shape[1])
        epv = float(n_events) / float(n_predictors) if n_predictors else None
        positive_rate = n_pos / n_obs if n_obs else None
        minority_rate = n_events / n_obs if n_obs else None

        # --- Fit ---
        X_const = sm.add_constant(X, has_constant="add")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                model = sm.Logit(y, X_const).fit(disp=0, maxiter=200)
            except np.linalg.LinAlgError as e:
                # Perfect/quasi-complete separation very commonly surfaces as a
                # singular Hessian / LinAlgError rather than a named
                # PerfectSeparationError. Report it as separation, which is the
                # actionable diagnosis, while still naming the singular matrix.
                return _blocked(
                    "PERFECT_SEPARATION",
                    "Maximum-likelihood estimation failed with a singular matrix, which for "
                    "logistic regression almost always indicates (quasi-)complete separation "
                    "or perfectly collinear predictors. Coefficient estimates do not exist.",
                    details={"exception": str(e)[:200], "exception_type": "LinAlgError"},
                    suggested_next_actions=[
                        "Use penalized logistic regression (Firth/L2)",
                        "Remove or collapse the separating/collinear predictor",
                        "Check the coefficients table from a simpler model",
                    ],
                )
            except Exception as e:
                # PerfectSeparationError lives in different places across versions.
                if "Separation" in type(e).__name__ or "separation" in str(e).lower():
                    return _blocked(
                        "PERFECT_SEPARATION",
                        "Perfect separation detected: a predictor perfectly predicts the outcome, "
                        "so maximum-likelihood estimates do not exist.",
                        details={"exception": str(e)[:200]},
                        suggested_next_actions=[
                            "Use penalized logistic regression (Firth/L2)",
                            "Remove or collapse the separating predictor",
                        ],
                    )
                raise

        converged = bool(model.mle_retvals.get("converged", True))

        # --- Coefficient / odds-ratio table ---
        params = model.params
        conf = model.conf_int(alpha=alpha)
        bse = model.bse
        pvalues = model.pvalues

        coef_rows: List[Dict[str, Any]] = []
        max_abs_or = 0.0
        for term in params.index:
            coef = _safe_float(params[term])
            odds_ratio = _safe_float(np.exp(params[term])) if coef is not None else None
            lo = _safe_float(np.exp(conf.loc[term, 0]))
            hi = _safe_float(np.exp(conf.loc[term, 1]))
            p = _safe_float(pvalues[term])
            if term != "const" and odds_ratio is not None:
                max_abs_or = max(max_abs_or, abs(odds_ratio), abs(1.0 / odds_ratio) if odds_ratio else 0.0)
            coef_rows.append({
                "term": term,
                "coefficient_log_odds": coef,
                "std_error": _safe_float(bse[term]),
                "odds_ratio": odds_ratio,
                "or_ci_low": lo,
                "or_ci_high": hi,
                "p_value": p,
                "significant_at_alpha": (p is not None and p < alpha),
            })

        # --- Separation heuristics (when statsmodels does not raise) ---
        pseudo_r2 = _safe_float(model.prsquared)
        separation_detected = bool(
            (not converged)
            or (max_abs_or >= SEPARATION_ODDS_RATIO_FLAG)
            or (pseudo_r2 is not None and pseudo_r2 > 0.99)
        )
        separating_terms = [
            r["term"] for r in coef_rows
            if r["term"] != "const" and r["odds_ratio"] is not None
            and (abs(r["odds_ratio"]) >= SEPARATION_ODDS_RATIO_FLAG
                 or (r["odds_ratio"] and abs(1.0 / r["odds_ratio"]) >= SEPARATION_ODDS_RATIO_FLAG))
        ]

        vif_rows = _compute_vif(X)

        metrics = {
            "method": "Binary logistic regression (logit, MLE)",
            "n_obs": n_obs,
            "n_events": n_events,
            "n_positive": n_pos,
            "n_negative": n_neg,
            "n_predictors": n_predictors,
            "events_per_variable": round(epv, 3) if epv is not None else None,
            "positive_rate": round(positive_rate, 4) if positive_rate is not None else None,
            "minority_class_rate": round(minority_rate, 4) if minority_rate is not None else None,
            "converged": converged,
            "separation_detected": separation_detected,
            "max_abs_odds_ratio": max_abs_or if max_abs_or else None,
            "pseudo_r2_mcfadden": pseudo_r2,
            "log_likelihood": _safe_float(model.llf),
            "lr_test_statistic": _safe_float(model.llr),
            "lr_test_p_value": _safe_float(model.llr_pvalue),
            "alpha": alpha,
        }

        tables = {
            "coefficients": coef_rows,
            "vif": vif_rows,
        }

        metadata = {
            "is_inferential": True,
            "positive_class": binfo.get("positive_class"),
            "negative_class": binfo.get("negative_class"),
            "separating_terms": separating_terms,
            "used_features": prep["details"].get("used_features"),
            "excluded_features": prep["details"].get("excluded_features"),
            "encoded_columns": prep["details"].get("encoded_columns"),
        }

        pos = binfo.get("positive_class")
        summary = (
            f"Fitted binary logistic regression for '{target_col}' "
            f"(modeling P(outcome = {pos!r})) on {n_predictors} predictor term(s), "
            f"n = {n_obs} ({n_events} events of the rarer class). "
            f"McFadden pseudo-R\u00b2 = {pseudo_r2 if pseudo_r2 is not None else 'n/a'}."
        )

        return _ok(summary, {
            "metrics": metrics,
            "tables": tables,
            "metadata": metadata,
            "summary": summary,
        })

    except Exception as e:
        return _failed("LOGISTIC_REGRESSION_FAILED", "Logistic regression failed", e)


# ==========================================================
# Extractor (payload -> title/summary/metrics/tables/metadata)
# ==========================================================

def extract_logistic_regression(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    details = payload.get("details", payload) if isinstance(payload, dict) else {}
    metrics = details.get("metrics", {}) or {}
    tables = details.get("tables", {}) or {}
    metadata = details.get("metadata", {}) or {}
    summary = details.get("summary") or default_summary
    title = default_title
    return title, summary, metrics, tables, metadata


# ==========================================================
# Display config
# ==========================================================

LOGISTIC_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "n_obs": "N (complete cases)",
            "n_events": "Events (rarer class)",
            "n_predictors": "Predictor terms",
            "events_per_variable": "Events per variable",
            "positive_rate": "Positive rate",
            "converged": "Converged",
            "separation_detected": "Separation detected",
            "pseudo_r2_mcfadden": "McFadden pseudo-R\u00b2",
            "lr_test_p_value": "LR test p-value",
            "alpha": "Alpha",
        },
        formatters={
            "events_per_variable": format_number,
            "positive_rate": format_number,
            "pseudo_r2_mcfadden": format_number,
            "lr_test_p_value": format_p_value,
        },
        order=[
            "method", "n_obs", "n_events", "n_predictors", "events_per_variable",
            "positive_rate", "converged", "separation_detected",
            "pseudo_r2_mcfadden", "lr_test_statistic", "lr_test_p_value", "alpha",
        ],
    ),
    tables={
        "coefficients": TableDisplayConfig(
            column_labels={
                "term": "Term",
                "coefficient_log_odds": "Coef (log-odds)",
                "std_error": "Std. Error",
                "odds_ratio": "Odds Ratio",
                "or_ci_low": "OR CI Low",
                "or_ci_high": "OR CI High",
                "p_value": "p-value",
                "significant_at_alpha": "Significant",
            },
            column_formatters={
                "coefficient_log_odds": format_number,
                "std_error": format_number,
                "odds_ratio": format_number,
                "or_ci_low": format_number,
                "or_ci_high": format_number,
                "p_value": format_p_value,
            },
            column_order=[
                "term", "coefficient_log_odds", "std_error", "odds_ratio",
                "or_ci_low", "or_ci_high", "p_value", "significant_at_alpha",
            ],
        ),
        "vif": TableDisplayConfig(
            column_labels={"term": "Term", "vif": "VIF"},
            column_formatters={"vif": format_number},
            column_order=["term", "vif"],
        ),
    },
)


# ==========================================================
# Registration
# ==========================================================

PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_logistic_regression",
    display_name="Logistic Regression",
    description=(
        "Fit a binary logistic regression (logit link) and report odds ratios with "
        "confidence intervals, model fit, and statistical guardrails (separation, "
        "events-per-variable, class balance, multicollinearity)."
    ),
    use_when=[
        "The outcome is binary (two classes) and the question is about how predictors "
        "relate to the probability/odds of one class.",
    ],
    do_not_use_when=[
        "The outcome is continuous (use run_multiple_regression).",
        "The outcome has more than two classes.",
    ],
    evidence_categories=["regression_model", "statistical_inference"],
    requires_confirmation=False,
    is_inferential=True,
    argument_schema=ArgumentSchema(
        required={
            "target_col": str,
            "feature_cols": list,
        },
        optional={
            "positive_label": object,
            "max_missing_rate": float,
            "max_categorical_levels": int,
            "numeric_parse_threshold": float,
            "min_n_per_parameter": int,
            "alpha": float,
        },
        column_args=["target_col"],
        column_list_args=["feature_cols"],
        allow_all_columns=False,
    ),
    execute=execute_logistic_regression,
    extractor=extract_logistic_regression,
    guardrail_evaluators=[
        evaluate_logistic_guardrails,
    ],
    display_config=LOGISTIC_DISPLAY,
))