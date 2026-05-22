"""
Benchmark dataset generator.

Each dataset is engineered to contain a specific statistical "trap" that a
naive LLM-as-statistician (Julius, ChatGPT ADA, Auto-Analyst) is known to
mishandle, but that a rigorous framework should catch automatically.

Datasets are written to benchmark/datasets/ as both CSV (for uploading to
competitor products) and parquet (for the local plugin runner).

Seeds are fixed so the entire benchmark is reproducible -- which is itself
part of the thesis.
"""

from __future__ import annotations

import os
import numpy as np
import pandas as pd

OUT_DIR = os.path.join(os.path.dirname(__file__), "datasets")
os.makedirs(OUT_DIR, exist_ok=True)

RNG_SEED = 20260520


def _save(df: pd.DataFrame, name: str) -> None:
    csv_path = os.path.join(OUT_DIR, f"{name}.csv")
    pq_path = os.path.join(OUT_DIR, f"{name}.parquet")
    df.to_csv(csv_path, index=False)
    df.to_parquet(pq_path, index=False)
    print(f"  {name}: {df.shape[0]} rows x {df.shape[1]} cols -> {name}.csv / .parquet")


def case1_unequal_variance_anova() -> pd.DataFrame:
    """
    Trap: three groups with WILDLY unequal variances.
    Correct handling: Levene's test rejects equal variance -> use Welch's
    ANOVA + Games-Howell, NOT classic ANOVA + Tukey.
    Group C has 6x the SD of group A.
    """
    rng = np.random.default_rng(RNG_SEED + 1)
    a = rng.normal(100, 5, 40)    # tight
    b = rng.normal(105, 8, 40)    # medium
    c = rng.normal(110, 30, 40)   # very wide
    return pd.DataFrame({
        "outcome": np.concatenate([a, b, c]),
        "treatment": ["A"] * 40 + ["B"] * 40 + ["C"] * 40,
    })


def case2_high_leverage_regression() -> pd.DataFrame:
    """
    Trap: a clean linear relationship plus ONE extreme high-leverage outlier
    that drags the slope. Correct handling: flag the point via Cook's distance
    / leverage / influence diagnostics.

    Columns use realistic names (hours_studied / exam_score). A bare 'x' name
    can be misclassified as an ID column by some preprocessors, which would
    confound the trap.
    """
    rng = np.random.default_rng(RNG_SEED + 2)
    n = 60
    # Round to 1 decimal so the predictor has realistic repeated values
    # (continuous predictors with 100% unique values can trip naive
    # "id-like column" heuristics; real measured data has ties).
    x = np.round(rng.uniform(0, 10, n), 1)
    y = 2.0 * x + rng.normal(0, 1.5, n)
    # Inject one extreme high-leverage point
    x = np.append(x, 35.0)          # far outside x range
    y = np.append(y, 5.0)           # off the regression line
    return pd.DataFrame({"hours_studied": x, "exam_score": y})


def case3_multiple_comparisons() -> pd.DataFrame:
    """
    Trap: a dataset that invites running MANY independent tests in one session.
    5 outcome metrics x one grouping -> if a user runs 5 t-tests, the
    family-wise error rate balloons. Correct handling: warn about multiple
    comparisons at the session level.
    All 5 outcomes are PURE NOISE (no real group difference) so any
    "significant" result is a false positive -- the whole point.
    """
    rng = np.random.default_rng(RNG_SEED + 3)
    n_per = 50
    group = ["control"] * n_per + ["variant"] * n_per
    data = {"group": group}
    for i in range(1, 6):
        # No real effect; both groups same distribution
        vals = rng.normal(0, 1, 2 * n_per)
        data[f"metric_{i}"] = vals
    return pd.DataFrame(data)


def case4_nonnormal_two_group() -> pd.DataFrame:
    """
    Trap: two groups, severely right-skewed (log-normal) data.
    Correct handling: Shapiro-Wilk rejects normality -> recommend Mann-Whitney
    U rather than defaulting to a t-test.

    Note: sample size and skew are tuned so Shapiro-Wilk reliably rejects
    (small n + mild skew can fail to reject; that would defeat the trap).
    """
    rng = np.random.default_rng(RNG_SEED + 4)
    a = rng.lognormal(0, 1.3, 30)        # strongly skewed
    b = rng.lognormal(0.6, 1.3, 30)      # strongly skewed, shifted
    return pd.DataFrame({
        "response_time": np.concatenate([a, b]),
        "arm": ["placebo"] * 30 + ["drug"] * 30,
    })


def case5_paired_nonnormal() -> pd.DataFrame:
    """
    Trap: pre/post paired design where the DIFFERENCES are non-normal
    (a few large responders). Correct handling: Shapiro on the differences
    rejects normality -> recommend Wilcoxon signed-rank, while still
    reporting the paired t-test.
    """
    rng = np.random.default_rng(RNG_SEED + 5)
    n = 25
    pre = rng.normal(50, 8, n)
    # Most people change a little; a few change enormously -> skewed diffs
    small = rng.normal(2, 2, n)
    big_responders = rng.choice([0, 1], size=n, p=[0.8, 0.2]) * rng.normal(25, 5, n)
    post = pre + small + big_responders
    return pd.DataFrame({"pre_score": pre, "post_score": post})


def case6_effect_size_reporting() -> pd.DataFrame:
    """
    Trap: a clean, well-behaved two-group difference. There is no assumption
    violation here -- the trap is in the REPORTING. A rigorous tool reports
    Cohen's d / Hedges' g WITH a 95% confidence interval and a magnitude
    label; naive tools report a bare d (or omit effect size entirely).
    """
    rng = np.random.default_rng(RNG_SEED + 6)
    a = rng.normal(100, 15, 45)
    b = rng.normal(108, 15, 45)   # ~0.53 SD difference => medium effect
    return pd.DataFrame({
        "test_score": np.concatenate([a, b]),
        "cohort": ["2024"] * 45 + ["2025"] * 45,
    })


def main():
    print("Generating benchmark trap datasets...")
    _save(case1_unequal_variance_anova(), "case1_unequal_variance_anova")
    _save(case2_high_leverage_regression(), "case2_high_leverage_regression")
    _save(case3_multiple_comparisons(), "case3_multiple_comparisons")
    _save(case4_nonnormal_two_group(), "case4_nonnormal_two_group")
    _save(case5_paired_nonnormal(), "case5_paired_nonnormal")
    _save(case6_effect_size_reporting(), "case6_effect_size_reporting")
    print(f"\nDatasets written to: {OUT_DIR}")


if __name__ == "__main__":
    main()
