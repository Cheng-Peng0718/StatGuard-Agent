"""
Shared guardrails for logistic / binary-outcome regression tools.

Used by:
- logistic_regression.py

These guardrails read only from the standard analysis-run payload structure
(metrics / tables / metadata), exactly like group_comparison_guardrails.py, so
any current or future binary-outcome model plugin (e.g. a regularized or
penalized logistic variant) can reuse them without recomputation.

Why a dedicated shared module instead of reusing the linear-regression
guardrails: logistic regression has failure modes that simply do not exist for
OLS and that silently produce nonsense if unguarded --

  * (quasi-)complete separation -> coefficients/odds ratios diverge to +/-inf,
    Wald standard errors explode, p-values become meaningless;
  * events-per-variable (EPV) too low -> the n/p heuristic used for OLS does
    NOT apply; what matters is the count of the *rarer* outcome class;
  * severe class imbalance -> the model can be "accurate" while never
    predicting the minority class.

The plugin computes the underlying numbers (separation flags, EPV, class
balance, max |odds ratio|, convergence) and stores them in the run payload;
these evaluators turn those numbers into reviewer-facing findings.
"""

from typing import Any, Dict, List

from core.guardrails import _new_finding


# Conventional thresholds. Kept here so any binary-outcome plugin shares one
# definition rather than hard-coding magic numbers in several places.
EPV_WARN_THRESHOLD = 10          # events-per-variable below this is risky
SEPARATION_ODDS_RATIO_FLAG = 1e6  # |OR| at/above this strongly suggests separation
SEVERE_IMBALANCE_MINORITY_RATE = 0.05  # minority class < 5% of rows


def evaluate_logistic_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}
    tables = run.get("tables", {}) or {}
    metadata = run.get("metadata", {}) or {}

    # ------------------------------------------------------------------
    # 1. (Quasi-)complete separation -- the signature logistic failure mode.
    # ------------------------------------------------------------------
    separation_detected = metrics.get("separation_detected")
    converged = metrics.get("converged")
    max_abs_or = metrics.get("max_abs_odds_ratio")

    if separation_detected is True:
        findings.append(_new_finding(
            category="assumption_check",
            severity="warning",
            title="Possible (quasi-)complete separation detected",
            message=(
                "One or more predictors appear to perfectly or near-perfectly "
                "separate the outcome classes. When this happens, maximum-likelihood "
                "estimates and odds ratios diverge and their standard errors and "
                "p-values are not trustworthy."
            ),
            evidence={
                "converged": converged,
                "max_abs_odds_ratio": max_abs_or,
                "separating_terms": metadata.get("separating_terms"),
            },
            recommendation=(
                "Do not interpret the affected coefficients or p-values at face value. "
                "Consider penalized logistic regression (e.g. Firth's method or L2 "
                "regularization), collapsing sparse categories, or removing the "
                "separating predictor."
            ),
        ))
    elif converged is False:
        findings.append(_new_finding(
            category="assumption_check",
            severity="warning",
            title="Logistic regression did not converge",
            message=(
                "The fitting routine did not converge. This often indicates separation, "
                "collinearity, or insufficient data for the requested model."
            ),
            evidence={
                "converged": converged,
                "max_abs_odds_ratio": max_abs_or,
            },
            recommendation=(
                "Treat all estimates as unreliable. Simplify the model, check for "
                "separation/collinearity, or use a penalized estimator."
            ),
        ))

    # ------------------------------------------------------------------
    # 2. Events-per-variable (EPV). The OLS n/p rule does NOT transfer:
    #    what constrains a logistic fit is the count of the rarer outcome.
    # ------------------------------------------------------------------
    epv = metrics.get("events_per_variable")
    n_events = metrics.get("n_events")
    n_params = metrics.get("n_predictors")

    if isinstance(epv, (int, float)) and epv < EPV_WARN_THRESHOLD:
        findings.append(_new_finding(
            category="assumption_check",
            severity="warning",
            title=f"Low events-per-variable (EPV \u2248 {round(float(epv), 1)})",
            message=(
                f"There are about {round(float(epv), 1)} events of the rarer outcome class "
                f"per estimated predictor, below the conventional minimum of "
                f"{EPV_WARN_THRESHOLD}. Coefficient estimates may be biased and unstable."
            ),
            evidence={
                "events_per_variable": epv,
                "n_events": n_events,
                "n_predictors": n_params,
                "epv_threshold": EPV_WARN_THRESHOLD,
            },
            recommendation=(
                "Reduce the number of predictors, collect more data for the rarer class, "
                "or use a penalized/regularized model. Interpret coefficients cautiously."
            ),
        ))

    # ------------------------------------------------------------------
    # 3. Severe class imbalance.
    # ------------------------------------------------------------------
    minority_rate = metrics.get("minority_class_rate")
    positive_rate = metrics.get("positive_rate")

    if isinstance(minority_rate, (int, float)) and 0 < minority_rate < SEVERE_IMBALANCE_MINORITY_RATE:
        findings.append(_new_finding(
            category="assumption_check",
            severity="info",
            title="Severe outcome class imbalance",
            message=(
                f"The minority outcome class is only about {round(float(minority_rate) * 100, 1)}% "
                "of usable rows. Overall classification accuracy is a misleading metric here, "
                "and the model may rarely predict the minority class."
            ),
            evidence={
                "minority_class_rate": minority_rate,
                "positive_rate": positive_rate,
            },
            recommendation=(
                "Report discrimination/calibration metrics that are robust to imbalance "
                "(e.g. ROC AUC, precision-recall) rather than raw accuracy, and consider "
                "whether resampling or class weighting is appropriate."
            ),
        ))

    # ------------------------------------------------------------------
    # 4. Multicollinearity among predictors (shared concern with OLS, but
    #    surfaced from this run's own computed VIF table).
    # ------------------------------------------------------------------
    vif_rows = tables.get("vif") or []
    high_vif = [
        row for row in vif_rows
        if isinstance(row, dict) and isinstance(row.get("vif"), (int, float)) and row.get("vif") >= 10
    ]
    if high_vif:
        findings.append(_new_finding(
            category="assumption_check",
            severity="warning",
            title=f"High multicollinearity in {len(high_vif)} predictor(s)",
            message=(
                "One or more predictors have a variance inflation factor at or above 10, "
                "indicating strong multicollinearity. This inflates standard errors and "
                "destabilizes the affected odds-ratio estimates."
            ),
            evidence={
                "high_vif_terms": [
                    {"term": r.get("term"), "vif": r.get("vif")} for r in high_vif
                ],
            },
            recommendation=(
                "Drop or combine collinear predictors, or use a regularized model, before "
                "interpreting the affected coefficients."
            ),
        ))

    # ------------------------------------------------------------------
    # 5. Overall model significance (likelihood-ratio test) for interpretation.
    # ------------------------------------------------------------------
    lr_p = metrics.get("lr_test_p_value")
    if isinstance(lr_p, (int, float)):
        if lr_p < 0.05:
            findings.append(_new_finding(
                category="interpretation",
                severity="info",
                title="Model improves on the null (likelihood-ratio test)",
                message=(
                    "The likelihood-ratio test indicates the predictors jointly improve fit "
                    "over an intercept-only model. Interpret individual odds ratios with their "
                    "confidence intervals, and avoid causal language unless the design supports it."
                ),
                evidence={"lr_test_p_value": lr_p, "pseudo_r2": metrics.get("pseudo_r2_mcfadden")},
            ))
        else:
            findings.append(_new_finding(
                category="interpretation",
                severity="info",
                title="Model does not clearly improve on the null",
                message=(
                    "The likelihood-ratio test does not show that the predictors jointly improve "
                    "fit over an intercept-only model. Individual significant terms, if any, "
                    "should be interpreted with caution."
                ),
                evidence={"lr_test_p_value": lr_p, "pseudo_r2": metrics.get("pseudo_r2_mcfadden")},
            ))

    return findings