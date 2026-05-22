"""
Per-plugin statistical claim builders.

Each builder takes a finished analysis_run dict and emits a list of structured
Claim objects, computed mechanically from the run's metrics. This mirrors the
shared/apa_writers.py pattern: one function per tool family, plus a dispatch
table keyed by tool_name.

Claim IDs are stable and derived from the run id, so the supervisor can
reference them by ID in its final answer and the renderer can resolve them.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from core.claims import (
    Claim,
    CLAIM_SIGNIFICANCE,
    CLAIM_EFFECT_SIZE,
    CLAIM_TEST_STATISTIC,
)


def _rid(run: Dict[str, Any]) -> str:
    """A short, stable run identifier used to namespace claim ids."""
    return str(run.get("run_id") or run.get("action_id") or "run")


def _metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    return run.get("metrics", {}) or {}


def _subject_group(run: Dict[str, Any]) -> str:
    m = _metrics(run)
    target = m.get("target_col") or (run.get("arguments", {}) or {}).get("target_col") or "the outcome"
    group = m.get("group_col") or (run.get("arguments", {}) or {}).get("group_col") or "group"
    return f"{target} across {group}"


def _bounded_unit_for(effect_name: Optional[str]) -> bool:
    if not effect_name:
        return False
    n = effect_name.lower()
    return any(tok in n for tok in ["eta", "omega", "epsilon", "r\u00b2", "squared"])


# ============================================================
# Group comparison (Welch t / ANOVA) and independent t-test share a shape
# ============================================================

def _build_group_like_claims(run: Dict[str, Any]) -> List[Claim]:
    m = _metrics(run)
    rid = _rid(run)
    tool = run.get("tool_name")
    subject = _subject_group(run)
    claims: List[Claim] = []

    # Significance
    if m.get("p_value") is not None:
        claims.append(Claim(
            claim_id=f"{rid}_sig",
            kind=CLAIM_SIGNIFICANCE,
            subject=subject,
            data={
                "p_value": m.get("p_value"),
                "alpha": m.get("alpha", 0.05),
                "significant": m.get("significant_at_alpha"),
            },
            source_run_id=rid,
            source_tool_name=tool,
        ))

    # Effect size
    if m.get("effect_size") is not None:
        ename = m.get("effect_size_name", "effect size")
        claims.append(Claim(
            claim_id=f"{rid}_es",
            kind=CLAIM_EFFECT_SIZE,
            subject=subject,
            data={
                "name": ename,
                "value": m.get("effect_size"),
                "magnitude": m.get("effect_size_magnitude"),
                "ci_low": m.get("effect_size_ci_low"),
                "ci_high": m.get("effect_size_ci_high"),
                "bounded_unit": _bounded_unit_for(ename),
            },
            source_run_id=rid,
            source_tool_name=tool,
        ))

    # Test statistic (t or F)
    if m.get("t_statistic") is not None:
        claims.append(Claim(
            claim_id=f"{rid}_stat",
            kind=CLAIM_TEST_STATISTIC,
            subject=subject,
            data={"label": "t", "value": m.get("t_statistic"),
                  "df": m.get("degrees_of_freedom")},
            source_run_id=rid, source_tool_name=tool,
        ))
    elif m.get("F_statistic") is not None:
        claims.append(Claim(
            claim_id=f"{rid}_stat",
            kind=CLAIM_TEST_STATISTIC,
            subject=subject,
            data={"label": "F", "value": m.get("F_statistic"),
                  "df": m.get("degrees_of_freedom_between")},
            source_run_id=rid, source_tool_name=tool,
        ))

    # Direction is descriptive ("which group is higher") and is left to the
    # model's narrative; only tamper-proof numeric claims (significance,
    # effect size, test statistic) are bound here.

    return claims


def build_claims_statistical_group_comparison(run: Dict[str, Any]) -> List[Claim]:
    return _build_group_like_claims(run)


def build_claims_independent_t_test(run: Dict[str, Any]) -> List[Claim]:
    return _build_group_like_claims(run)


# ============================================================
# Paired comparison
# ============================================================

def build_claims_paired_comparison(run: Dict[str, Any]) -> List[Claim]:
    m = _metrics(run)
    rid = _rid(run)
    tool = run.get("tool_name")
    args = run.get("arguments", {}) or {}
    c1 = args.get("target_col_1", "measurement 1")
    c2 = args.get("target_col_2", "measurement 2")
    subject = f"{c1} vs {c2} (paired)"
    claims: List[Claim] = []

    if m.get("p_value") is not None:
        claims.append(Claim(
            claim_id=f"{rid}_sig",
            kind=CLAIM_SIGNIFICANCE,
            subject=subject,
            data={"p_value": m.get("p_value"), "alpha": m.get("alpha", 0.05),
                  "significant": m.get("significant_at_alpha")},
            source_run_id=rid, source_tool_name=tool,
        ))

    if m.get("effect_size") is not None:
        ename = m.get("effect_size_name", "effect size")
        claims.append(Claim(
            claim_id=f"{rid}_es",
            kind=CLAIM_EFFECT_SIZE,
            subject=subject,
            data={"name": ename, "value": m.get("effect_size"),
                  "magnitude": m.get("effect_size_magnitude"),
                  "bounded_unit": _bounded_unit_for(ename)},
            source_run_id=rid, source_tool_name=tool,
        ))

    return claims


# ============================================================
# Nonparametric group comparison (Mann-Whitney / Kruskal-Wallis)
# ============================================================

def build_claims_nonparametric_group_comparison(run: Dict[str, Any]) -> List[Claim]:
    m = _metrics(run)
    rid = _rid(run)
    tool = run.get("tool_name")
    subject = _subject_group(run)
    claims: List[Claim] = []

    if m.get("p_value") is not None:
        claims.append(Claim(
            claim_id=f"{rid}_sig",
            kind=CLAIM_SIGNIFICANCE,
            subject=subject,
            data={"p_value": m.get("p_value"), "alpha": m.get("alpha", 0.05),
                  "significant": m.get("significant_at_alpha")},
            source_run_id=rid, source_tool_name=tool,
        ))

    if m.get("effect_size") is not None:
        ename = m.get("effect_size_name", "effect size")
        claims.append(Claim(
            claim_id=f"{rid}_es",
            kind=CLAIM_EFFECT_SIZE,
            subject=subject,
            data={"name": ename, "value": m.get("effect_size"),
                  "magnitude": m.get("effect_size_magnitude"),
                  "bounded_unit": _bounded_unit_for(ename)},
            source_run_id=rid, source_tool_name=tool,
        ))

    # Mann-Whitney U or Kruskal-Wallis H as the test statistic
    if m.get("U_statistic") is not None:
        claims.append(Claim(
            claim_id=f"{rid}_stat", kind=CLAIM_TEST_STATISTIC, subject=subject,
            data={"label": "U", "value": m.get("U_statistic")},
            source_run_id=rid, source_tool_name=tool,
        ))
    elif m.get("H_statistic") is not None:
        claims.append(Claim(
            claim_id=f"{rid}_stat", kind=CLAIM_TEST_STATISTIC, subject=subject,
            data={"label": "H", "value": m.get("H_statistic"),
                  "df": m.get("degrees_of_freedom_between")},
            source_run_id=rid, source_tool_name=tool,
        ))

    return claims


# ============================================================
# Dispatch
# ============================================================

CLAIMS_BUILDERS_BY_TOOL_NAME: Dict[str, Any] = {
    "statistical_group_comparison": build_claims_statistical_group_comparison,
    "run_independent_t_test": build_claims_independent_t_test,
    "independent_t_test": build_claims_independent_t_test,
    "paired_comparison": build_claims_paired_comparison,
    "nonparametric_group_comparison": build_claims_nonparametric_group_comparison,
}


def build_claims_for_run(run: Dict[str, Any]) -> List[Claim]:
    """Look up the builder for a run's tool and return its claims (or [])."""
    if not isinstance(run, dict):
        return []
    if run.get("status") not in {"ok", "warning"}:
        return []
    builder = CLAIMS_BUILDERS_BY_TOOL_NAME.get(run.get("tool_name"))
    if builder is None:
        return []
    try:
        return builder(run) or []
    except Exception:
        return []