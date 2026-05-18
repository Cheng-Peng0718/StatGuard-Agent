import math
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

# ============================================================
# Residual histogram guardrails (existing, retained)
# ============================================================

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


# ============================================================
# Multiple comparison correction (session-level)
# ============================================================

# These categories indicate the run produced an inferential p-value test.
# Used to count tests against family-wise error rate.
_INFERENTIAL_EVIDENCE_CATEGORIES = {
    "statistical_inference",
    "group_comparison",
    "regression_model",
}


def _is_inferential_run(run: Dict[str, Any]) -> bool:
    if not isinstance(run, dict):
        return False

    if run.get("status") not in {"ok", "warning"}:
        return False

    categories = set(run.get("evidence_categories", []) or [])

    if categories & _INFERENTIAL_EVIDENCE_CATEGORIES:
        return True

    # Fallback: if a p_value is present in metrics, treat as inferential.
    metrics = run.get("metrics", {}) or {}

    if "p_value" in metrics and metrics.get("p_value") is not None:
        return True

    return False


def _count_inferential_runs(analysis_runs: List[Dict[str, Any]]) -> int:
    if not analysis_runs:
        return 0

    return sum(1 for r in analysis_runs if _is_inferential_run(r))


def evaluate_multiple_comparison_guardrails(
    context_or_run: Any,
    analysis_runs: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """
    Session-level guardrail. Recommends multiple-comparison correction when more
    than one inferential test has been run in the session.

    Accepts either:
      - A context-like object exposing `.analysis_runs` (list of run dicts), or
      - A run dict whose `metadata.analysis_runs` carries the list, or
      - A run dict plus an explicit analysis_runs list (for plugin use).

    Returns findings that recommend a Bonferroni-corrected alpha = 0.05/K when
    K >= 2 inferential tests have been performed, and notes that BH-FDR control
    is preferable when many tests are run and discoveries (not strict FWER) are
    the goal.
    """
    runs: List[Dict[str, Any]] = []

    if analysis_runs is not None:
        runs = analysis_runs or []
    elif isinstance(context_or_run, dict):
        metadata = context_or_run.get("metadata", {}) or {}
        runs = (
            metadata.get("analysis_runs")
            or context_or_run.get("analysis_runs")
            or []
        )
    else:
        runs = list(getattr(context_or_run, "analysis_runs", []) or [])

    k = _count_inferential_runs(runs)

    findings: List[Dict[str, Any]] = []

    if k <= 1:
        return findings

    try:
        bonferroni_alpha = 0.05 / k
    except Exception:
        bonferroni_alpha = None

    findings.append(_new_finding(
        category="multiple_comparisons",
        severity="warning",
        title=f"{k} inferential tests in this session; consider multiple-comparison correction",
        message=(
            f"This session has produced {k} inferential test results at a nominal alpha of "
            f"0.05. The family-wise error rate is no longer 5% when multiple tests are "
            f"interpreted together."
        ),
        evidence={
            "k_inferential_tests": k,
            "bonferroni_alpha_per_test": (
                round(bonferroni_alpha, 6) if bonferroni_alpha is not None else None
            ),
        },
        recommendation=(
            f"For strict family-wise error control, compare each p-value against "
            f"alpha/k = 0.05/{k}"
            + (
                f" ≈ {round(bonferroni_alpha, 4)}"
                if bonferroni_alpha is not None else ""
            )
            + ". For discovery-oriented analysis with many tests, prefer the Benjamini-Hochberg "
              "FDR procedure. State the chosen correction explicitly in the report."
        ),
    ))

    return findings

