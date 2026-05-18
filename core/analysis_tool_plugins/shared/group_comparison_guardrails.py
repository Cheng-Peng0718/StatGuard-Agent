"""
Shared guardrails for group-comparison-style tools.

Used by:
- statistical_group_comparison.py (multi-group + 2-group umbrella)
- independent_t_test.py
- anova.py (which delegates to statistical_group_comparison)

These guardrails read only from the standard run payload structure
(metrics / tables / metadata), so any plugin that conforms to that
contract can reuse them.
"""

from typing import Any, Dict, List

from core.guardrails import _new_finding


def evaluate_group_comparison_guardrails(run: Dict[str, Any]) -> List[Dict[str, Any]]:
    findings: List[Dict[str, Any]] = []

    metrics = run.get("metrics", {}) or {}
    tables = run.get("tables", {}) or {}
    metadata = run.get("metadata", {}) or {}

    sig = metrics.get("significant_at_alpha")
    p_value = metrics.get("p_value")
    effect_size = metrics.get("effect_size")
    effect_name = metrics.get("effect_size_name")
    effect_magnitude = metrics.get("effect_size_magnitude")
    method = metrics.get("method")

    # Significance with effect-size context
    if sig is True:
        findings.append(_new_finding(
            category="interpretation",
            severity="info",
            title="Statistically significant group difference",
            message=(
                "The group comparison is statistically significant at the chosen alpha. "
                "Statistical significance reflects evidence against the null of no difference; "
                "use the effect size to judge practical importance."
            ),
            evidence={
                "method": method,
                "p_value": p_value,
                "effect_size_name": effect_name,
                "effect_size": effect_size,
                "effect_size_magnitude": effect_magnitude,
            },
            recommendation=(
                "Report the effect size alongside the p-value. Avoid causal language unless "
                "the data come from a randomized design."
            ),
        ))
    elif sig is False:
        findings.append(_new_finding(
            category="interpretation",
            severity="info",
            title="No statistically significant group difference",
            message=(
                "The group comparison is not statistically significant at the chosen alpha. "
                "Absence of evidence is not evidence of absence; small samples can hide real effects."
            ),
            evidence={
                "method": method,
                "p_value": p_value,
                "effect_size_name": effect_name,
                "effect_size": effect_size,
            },
            recommendation=(
                "Consider whether sample size is adequate to detect a practically meaningful "
                "difference; report a confidence interval for the mean difference."
            ),
        ))

    # Variance equality (Levene)
    variances_equal = metrics.get("variances_equal_at_0_05")
    levene_p = metrics.get("levene_p_value")
    var_ratio = metrics.get("max_to_min_variance_ratio")

    if variances_equal is False:
        findings.append(_new_finding(
            category="assumption_check",
            severity="info",
            title="Unequal variances detected; Welch's ANOVA used",
            message=(
                "Levene's test rejected equality of variances at 0.05; Welch's/Alexander-Govern "
                "ANOVA was used instead of the classic F-test."
            ),
            evidence={
                "levene_p_value": levene_p,
                "max_to_min_variance_ratio": var_ratio,
            },
        ))
    elif variances_equal is True:
        findings.append(_new_finding(
            category="assumption_check",
            severity="info",
            title="Variances appear approximately equal",
            message=(
                "Levene's test did not reject equality of variances; classic ANOVA was used."
            ),
            evidence={
                "levene_p_value": levene_p,
                "max_to_min_variance_ratio": var_ratio,
            },
        ))

    # Post-hoc presence
    post_hoc = tables.get("post_hoc_pairwise") or []
    post_hoc_method = metadata.get("post_hoc_method")

    if post_hoc:
        n_sig = sum(1 for row in post_hoc if row.get("significant_at_alpha"))

        findings.append(_new_finding(
            category="post_hoc",
            severity="info",
            title=f"Post-hoc: {n_sig} significant pair(s) of {len(post_hoc)}",
            message=(
                f"Pairwise comparisons were computed using {post_hoc_method or 'a FWER-controlled procedure'}. "
                f"{n_sig} of {len(post_hoc)} pairwise comparisons are significant at the chosen alpha."
            ),
            evidence={
                "post_hoc_method": post_hoc_method,
                "n_significant_pairs": n_sig,
                "n_pairs": len(post_hoc),
            },
        ))
    elif sig is True and metrics.get("valid_group_count") and int(metrics.get("valid_group_count") or 0) >= 3:
        findings.append(_new_finding(
            category="post_hoc",
            severity="warning",
            title="Significant ANOVA without reported post-hoc",
            message=(
                "The omnibus test is significant, but no pairwise post-hoc results are present "
                "in this run. Conclusions about specific group differences cannot be drawn from "
                "the omnibus alone."
            ),
            evidence={
                "method": method,
                "p_value": p_value,
                "valid_group_count": metrics.get("valid_group_count"),
            },
            recommendation=(
                "Re-run with a tool that computes Tukey HSD or Games-Howell post-hoc comparisons."
            ),
        ))

    # Group-level non-normality
    shapiro_rows = tables.get("shapiro_per_group") or []
    non_normal_groups = [
        row.get("group") for row in shapiro_rows
        if row.get("normal_at_0_05") is False
    ]

    if non_normal_groups:
        findings.append(_new_finding(
            category="assumption_check",
            severity="info",
            title=f"Non-normality flagged in {len(non_normal_groups)} group(s)",
            message=(
                "Shapiro-Wilk rejected normality at 0.05 in one or more groups. With small "
                "groups this can materially affect parametric inference."
            ),
            evidence={
                "non_normal_groups": non_normal_groups,
            },
            recommendation=(
                "For small samples, consider Mann-Whitney U (two groups) or Kruskal-Wallis "
                "(three or more groups) as non-parametric alternatives. For large samples, "
                "the CLT typically protects inference; document the deviation."
            ),
        ))

    return findings