import uuid
from typing import Any, Dict, List, Optional


def _new_finding(
    *,
    category: str,
    severity: str,
    title: str,
    message: str,
    evidence: Optional[Dict[str, Any]] = None,
    recommendation: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "finding_id": f"gr_{uuid.uuid4().hex[:8]}",
        "category": category,
        "severity": severity,
        "title": title,
        "message": message,
        "evidence": evidence or {},
        "recommendation": recommendation,
    }


def evaluate_regression_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Guardrails for a fitted linear/regression-type model.

    Uses:
    - run["metrics"] for user-facing model metrics
    - run["metadata"] for internal fields such as p_eff / n_eff
    """
    findings: List[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}
    metadata = run.get("metadata", {}) or {}
    tables = run.get("tables", {}) or {}
    arguments = run.get("arguments", {}) or {}

    nobs = metrics.get("nobs", metadata.get("nobs"))
    r2 = metrics.get("r_squared")
    adj_r2 = metrics.get("adj_r_squared")
    f_p = metrics.get("f_p_value")

    # Important after Plugin Quality Layer:
    # p_eff should live in metadata, not user-facing metrics.
    p_eff = metadata.get("p_eff", metrics.get("p_eff"))
    n_eff = metadata.get("n_eff", metrics.get("n_eff"))

    target = arguments.get("target_col")
    features = arguments.get("feature_cols", [])

    # ------------------------------------------------------------
    # Sample size / model complexity
    # ------------------------------------------------------------
    if nobs is not None and p_eff is not None:
        try:
            nobs_f = float(nobs)
            p_eff_f = float(p_eff)

            if p_eff_f > 0:
                ratio = nobs_f / p_eff_f

                if ratio < 10:
                    findings.append(_new_finding(
                        category="sample_size",
                        severity="warning",
                        title="Low observations-per-parameter ratio",
                        message=(
                            "The model may be underpowered or unstable because the number "
                            "of observations per effective predictor is low."
                        ),
                        evidence={
                            "nobs": nobs,
                            "n_eff": n_eff,
                            "p_eff": p_eff,
                            "nobs_per_parameter": ratio,
                        },
                        recommendation=(
                            "Consider reducing model complexity, collecting more data, "
                            "or using regularized methods."
                        ),
                    ))
                else:
                    findings.append(_new_finding(
                        category="sample_size",
                        severity="info",
                        title="Sample size appears adequate for model size",
                        message=(
                            "The observations-per-parameter ratio does not raise an immediate "
                            "sample-size warning."
                        ),
                        evidence={
                            "nobs": nobs,
                            "n_eff": n_eff,
                            "p_eff": p_eff,
                            "nobs_per_parameter": ratio,
                        },
                    ))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Explanatory power
    # ------------------------------------------------------------
    if r2 is not None:
        try:
            r2_f = float(r2)

            if r2_f < 0.10:
                findings.append(_new_finding(
                    category="model_fit",
                    severity="warning",
                    title="Low explanatory power",
                    message=(
                        "The model explains only a small fraction of outcome variation. "
                        "A statistically significant predictor may still have limited "
                        "practical predictive value."
                    ),
                    evidence={
                        "r_squared": r2,
                        "adj_r_squared": adj_r2,
                    },
                    recommendation=(
                        "Consider adding theoretically relevant predictors, checking nonlinear "
                        "relationships, or evaluating prediction error."
                    ),
                ))
            elif r2_f < 0.30:
                findings.append(_new_finding(
                    category="model_fit",
                    severity="info",
                    title="Moderate-to-low explanatory power",
                    message=(
                        "The model explains some variation, but substantial unexplained "
                        "variation remains."
                    ),
                    evidence={
                        "r_squared": r2,
                        "adj_r_squared": adj_r2,
                    },
                ))
            else:
                findings.append(_new_finding(
                    category="model_fit",
                    severity="info",
                    title="Model explains a nontrivial share of variation",
                    message=(
                        "The R-squared value suggests the model captures a meaningful share "
                        "of variation in the outcome."
                    ),
                    evidence={
                        "r_squared": r2,
                        "adj_r_squared": adj_r2,
                    },
                ))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Significance vs causality
    # ------------------------------------------------------------
    if f_p is not None:
        try:
            f_p_f = float(f_p)

            if f_p_f < 0.05:
                findings.append(_new_finding(
                    category="interpretation",
                    severity="info",
                    title="Statistically significant association",
                    message=(
                        "The overall model is statistically significant. This supports an "
                        "association between the predictor set and the outcome, but does not "
                        "establish causality."
                    ),
                    evidence={
                        "f_p_value": f_p,
                        "target_col": target,
                        "feature_cols": features,
                    },
                    recommendation=(
                        "Avoid causal language unless the study design supports causal inference."
                    ),
                ))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Coefficient table interpretation
    # ------------------------------------------------------------
    coef_table = tables.get("coef_table", []) or []

    if coef_table:
        for row in coef_table:
            if not isinstance(row, dict):
                continue

            term = row.get("term")

            # Do not interpret intercept as a predictor.
            if term in {"const", "intercept", "Intercept"}:
                continue

            p_value = row.get("p_value")
            coef = row.get("coef")

            try:
                p_f = float(p_value)
                coef_f = float(coef)

                if p_f < 0.05:
                    direction = "positive" if coef_f > 0 else "negative"

                    findings.append(_new_finding(
                        category="coefficient_interpretation",
                        severity="info",
                        title=f"Significant {direction} coefficient: {term}",
                        message=(
                            f"The coefficient for `{term}` is statistically significant "
                            f"and {direction}. Interpret this as an association conditional "
                            "on the model specification."
                        ),
                        evidence={
                            "term": term,
                            "coef": coef,
                            "p_value": p_value,
                            "ci_lower": row.get("ci_lower"),
                            "ci_upper": row.get("ci_upper"),
                        },
                    ))
            except Exception:
                pass

    return findings


def evaluate_diagnostics_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Guardrails for model diagnostics.
    """
    findings: List[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}

    max_vif = metrics.get("max_vif")
    bp_p = metrics.get("breusch_pagan_lm_p_value")
    hetero_flag = metrics.get("heteroscedasticity_flag_0_05")

    # ------------------------------------------------------------
    # Multicollinearity / VIF
    # ------------------------------------------------------------
    if max_vif is not None:
        try:
            v = float(max_vif)

            if v >= 10:
                findings.append(_new_finding(
                    category="multicollinearity",
                    severity="critical",
                    title="Severe multicollinearity possible",
                    message=(
                        "The maximum VIF is very high, suggesting severe multicollinearity."
                    ),
                    evidence={"max_vif": max_vif},
                    recommendation=(
                        "Inspect correlated predictors, remove redundant variables, "
                        "or use regularization/dimension reduction."
                    ),
                ))
            elif v >= 5:
                findings.append(_new_finding(
                    category="multicollinearity",
                    severity="warning",
                    title="Moderate multicollinearity possible",
                    message=(
                        "The maximum VIF is elevated, suggesting possible multicollinearity."
                    ),
                    evidence={"max_vif": max_vif},
                    recommendation=(
                        "Inspect pairwise correlations and consider removing redundant predictors."
                    ),
                ))
            else:
                findings.append(_new_finding(
                    category="multicollinearity",
                    severity="info",
                    title="No apparent multicollinearity issue",
                    message=(
                        "The maximum VIF does not indicate a multicollinearity problem."
                    ),
                    evidence={"max_vif": max_vif},
                ))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Heteroscedasticity
    # ------------------------------------------------------------
    if bp_p is not None:
        try:
            p = float(bp_p)

            if p < 0.05 or hetero_flag is True:
                findings.append(_new_finding(
                    category="heteroscedasticity",
                    severity="warning",
                    title="Possible heteroscedasticity",
                    message=(
                        "The Breusch-Pagan test suggests non-constant error variance. "
                        "Standard model standard errors may be unreliable."
                    ),
                    evidence={
                        "breusch_pagan_lm_p_value": bp_p,
                        "heteroscedasticity_flag_0_05": hetero_flag,
                    },
                    recommendation=(
                        "Consider robust standard errors, transformation, or model respecification."
                    ),
                ))
            else:
                findings.append(_new_finding(
                    category="heteroscedasticity",
                    severity="info",
                    title="No strong evidence of heteroscedasticity",
                    message=(
                        "The Breusch-Pagan test does not suggest strong evidence of "
                        "heteroscedasticity."
                    ),
                    evidence={
                        "breusch_pagan_lm_p_value": bp_p,
                        "heteroscedasticity_flag_0_05": hetero_flag,
                    },
                ))
        except Exception:
            pass

    return findings


def evaluate_residual_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Guardrails for residual summaries / residual histogram outputs.
    """
    findings: List[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}

    skew = metrics.get("residual_skewness")
    kurt = metrics.get("residual_kurtosis_fisher")
    outliers_3sd = metrics.get("outliers_abs_3sd")
    flags = metrics.get("diagnostic_flags") or []

    # ------------------------------------------------------------
    # Residual skewness
    # ------------------------------------------------------------
    if skew is not None:
        try:
            s = float(skew)

            if abs(s) >= 1:
                findings.append(_new_finding(
                    category="residual_distribution",
                    severity="warning",
                    title="Residual skewness detected",
                    message=(
                        "Residuals appear meaningfully skewed. Normal-error-based inference "
                        "should be interpreted cautiously."
                    ),
                    evidence={"residual_skewness": skew},
                    recommendation=(
                        "Inspect residual plots and consider transformations, nonlinear terms, "
                        "or robust methods if appropriate."
                    ),
                ))
            else:
                findings.append(_new_finding(
                    category="residual_distribution",
                    severity="info",
                    title="Residual skewness not severe",
                    message=(
                        "Residual skewness does not appear severe by the current threshold."
                    ),
                    evidence={"residual_skewness": skew},
                ))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Residual kurtosis / heavy tails
    # ------------------------------------------------------------
    if kurt is not None:
        try:
            k = float(kurt)

            if k >= 3:
                findings.append(_new_finding(
                    category="residual_distribution",
                    severity="warning",
                    title="Heavy-tailed residuals possible",
                    message=(
                        "Residual kurtosis is elevated, suggesting possible heavy tails or outliers."
                    ),
                    evidence={"residual_kurtosis_fisher": kurt},
                    recommendation=(
                        "Inspect influential observations and consider robust inference."
                    ),
                ))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Extreme residual outliers
    # ------------------------------------------------------------
    if outliers_3sd is not None:
        try:
            n = int(outliers_3sd)

            if n > 0:
                findings.append(_new_finding(
                    category="outliers",
                    severity="warning",
                    title="Extreme residual outliers detected",
                    message=(
                        f"{n} residual(s) exceed 3 standard deviations in absolute value."
                    ),
                    evidence={"outliers_abs_3sd": n},
                    recommendation=(
                        "Review these observations for data quality issues or high influence."
                    ),
                ))
        except Exception:
            pass

    # ------------------------------------------------------------
    # Recorded screening flags
    # ------------------------------------------------------------
    if flags:
        findings.append(_new_finding(
            category="diagnostic_flags",
            severity="info",
            title="Residual diagnostic flags recorded",
            message=(
                "The residual diagnostic tool reported one or more screening flags."
            ),
            evidence={"diagnostic_flags": flags},
        ))

    return findings