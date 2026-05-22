"""
Benchmark scoring rubric.

For each trap case, this defines the checklist of statistical-rigor elements
that a CORRECT analysis must surface. Each checklist item is a (key, label)
pair plus a detector function that inspects a tool's output.

The rubric is product-agnostic: the same checklist is applied to our agent's
output AND (manually) to competitor outputs. The checklist is the objective
yardstick, which is what makes the comparison defensible rather than
cherry-picked.

Scoring: each case yields a score = (rigor items hit) / (rigor items total).
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, NamedTuple


class RubricItem(NamedTuple):
    key: str
    label: str
    # Why this matters -- shown in the benchmark report so readers understand
    # the stakes, not just the checkmark.
    rationale: str


# ============================================================
# Per-case rubrics
# ============================================================

RUBRICS: Dict[str, Dict[str, Any]] = {
    "case1_unequal_variance_anova": {
        "title": "Three-group comparison with unequal variances",
        "dataset": "case1_unequal_variance_anova",
        "task_prompt": "Compare `outcome` across the three `treatment` groups. Is there a significant difference?",
        "trap": "Group variances differ ~6x. Classic ANOVA + Tukey assume equal variance and are invalid here.",
        "rigor_items": [
            RubricItem("levene_or_variance_check", "Tests homogeneity of variance (Levene/Bartlett)",
                       "Without checking variance equality, the choice between classic and variance-robust ANOVA is unjustified."),
            RubricItem("variance_robust_anova_used", "Uses a variance-robust ANOVA (Welch or Alexander-Govern), not classic ANOVA",
                       "Classic ANOVA inflates Type I error when variances are unequal; Welch and Alexander-Govern correct for this."),
            RubricItem("games_howell_posthoc", "Uses Games-Howell post-hoc (not Tukey) for unequal variance",
                       "Tukey HSD assumes equal variances; Games-Howell does not."),
            RubricItem("effect_size_reported", "Reports an effect size (eta squared / epsilon squared)",
                       "A p-value alone does not convey the magnitude of the difference."),
        ],
    },
    "case2_high_leverage_regression": {
        "title": "Simple regression with a high-leverage outlier",
        "dataset": "case2_high_leverage_regression",
        "task_prompt": "Fit a linear regression of `y` on `x` and report the relationship.",
        "trap": "One point sits ~6 SDs out in x and off the line; it can dominate the slope.",
        "rigor_items": [
            RubricItem("influence_diagnostics", "Computes influence diagnostics (Cook's distance / leverage / DFFITS)",
                       "Without influence diagnostics, a single point can silently drive the entire fit."),
            RubricItem("flags_the_outlier", "Explicitly flags the high-leverage / influential point",
                       "Detecting it is useless unless the user is told which observation to scrutinize."),
            RubricItem("residual_normality", "Checks residual normality (Shapiro / Jarque-Bera)",
                       "OLS inference assumes normally distributed residuals."),
            RubricItem("heteroscedasticity_check", "Checks for heteroscedasticity (Breusch-Pagan) or reports robust SE",
                       "Non-constant error variance biases standard errors."),
            RubricItem("r_squared_reported", "Reports R-squared / model fit",
                       "Fit quality is needed to judge whether the model is useful at all."),
        ],
    },
    "case3_multiple_comparisons": {
        "title": "Multiple tests in one session (family-wise error)",
        "dataset": "case3_multiple_comparisons",
        "task_prompt": "Run a separate t-test comparing control vs variant for each of metric_1 through metric_5.",
        "trap": "All five metrics are pure noise. Running five tests at alpha=.05 inflates the chance of a false positive to ~23%.",
        "rigor_items": [
            RubricItem("multiple_comparison_warning", "Warns that running multiple tests inflates family-wise error",
                       "The single most common silent error in applied statistics."),
            RubricItem("correction_suggested", "Suggests a correction (Bonferroni / Benjamini-Hochberg)",
                       "Warning without a remedy leaves the user stuck."),
            RubricItem("per_test_effect_size", "Reports effect size for each test",
                       "With pure-noise data, effect sizes near zero reveal the absence of real signal."),
            RubricItem("no_false_discovery_overclaim", "Does NOT over-claim a 'significant' noise result as a real finding",
                       "The trap is whether the tool narrates a false positive as a discovery."),
        ],
    },
    "case4_nonnormal_two_group": {
        "title": "Two-group comparison with non-normal data",
        "dataset": "case4_nonnormal_two_group",
        "task_prompt": "Compare `response_time` between the placebo and drug arms.",
        "trap": "Data are strongly right-skewed (log-normal). A Student/Welch t-test is not the best primary test.",
        "rigor_items": [
            RubricItem("normality_check", "Checks normality per group (Shapiro-Wilk)",
                       "The test choice depends on the normality verdict."),
            RubricItem("nonparametric_recommended", "Recommends / uses Mann-Whitney U given non-normality",
                       "Mann-Whitney is robust to the skew that breaks the t-test's assumptions."),
            RubricItem("rank_effect_size", "Reports a rank-based effect size (rank-biserial) or location shift",
                       "A p-value without an effect size is incomplete."),
            RubricItem("does_not_blindly_ttest", "Does NOT silently default to a plain t-test as the only analysis",
                       "Blindly applying a t-test to skewed data is the exact failure mode being tested."),
        ],
    },
    "case5_paired_nonnormal": {
        "title": "Paired pre/post design with non-normal differences",
        "dataset": "case5_paired_nonnormal",
        "task_prompt": "Compare `pre_score` and `post_score` (same subjects). Did scores change?",
        "trap": "The paired differences are non-normal (a few large responders). The data are PAIRED, not independent.",
        "rigor_items": [
            RubricItem("treats_as_paired", "Treats the data as paired (not two independent groups)",
                       "Using an independent-samples test on paired data discards the pairing and loses power."),
            RubricItem("difference_normality", "Checks normality of the differences (Shapiro on diffs)",
                       "Paired-t validity depends on normality of the differences, not the raw scores."),
            RubricItem("wilcoxon_recommended", "Recommends / uses Wilcoxon signed-rank given non-normal diffs",
                       "Wilcoxon is the robust paired alternative."),
            RubricItem("paired_effect_size", "Reports a paired effect size (Cohen's d_z or rank-biserial)",
                       "Effect size communicates the magnitude of change."),
        ],
    },
    "case6_effect_size_reporting": {
        "title": "Effect-size reporting on a clean difference",
        "dataset": "case6_effect_size_reporting",
        "task_prompt": "Compare `test_score` between the 2024 and 2025 cohorts.",
        "trap": "No assumption is violated. The trap is in REPORTING: a bare p-value or bare d is insufficient.",
        "rigor_items": [
            RubricItem("effect_size_reported", "Reports Cohen's d or Hedges' g",
                       "Statistical significance does not imply practical importance."),
            RubricItem("effect_size_ci", "Reports a 95% CI on the effect size",
                       "APA 7 and most journals now require effect-size CIs."),
            RubricItem("magnitude_label", "Provides a magnitude interpretation (small/medium/large)",
                       "Helps the reader judge practical significance."),
            RubricItem("hedges_correction", "Uses or mentions Hedges' g (small-sample bias correction)",
                       "Cohen's d is biased upward in small samples; Hedges' g corrects it."),
            RubricItem("mean_diff_ci", "Reports a CI on the raw mean difference",
                       "The raw difference in original units is what stakeholders ultimately care about."),
        ],
    },
}


def total_rigor_items(case_key: str) -> int:
    return len(RUBRICS[case_key]["rigor_items"])


def all_case_keys() -> List[str]:
    return list(RUBRICS.keys())
