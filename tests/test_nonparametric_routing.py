"""
Tests for P2: deterministic nonparametric routing in statistical_group_comparison.

When data are non-normal AND (small sample OR severe skew), the plugin switches
its primary test to a rank-based method (Mann-Whitney / Kruskal-Wallis) while
keeping the parametric test as a secondary result. The decision is a pure
function of computed statistics (Shapiro, n, skew) -- no LLM involvement.

Test groups:
  1. _decide_nonparametric: the two-factor rule and its boundaries
  2. _max_abs_skew helper
  3. End-to-end routing on synthetic data (switch vs no-switch)
  4. Secondary parametric test is retained and conclusion direction agrees
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from core.analysis_tool_plugins.plugins.statistical_group_comparison import (
    _decide_nonparametric,
    _max_abs_skew,
    _SMALL_GROUP_N,
    _HIGH_SKEW,
    execute_statistical_group_comparison,
)
from core.analysis_tool_plugins.registry import get_plugin


DATASETS = os.path.join(os.path.dirname(__file__), "..", "benchmark", "datasets")


class DummyContext:
    def __init__(self, df, args=None):
        self.df = df
        self.arguments = args or {}
        self.active_data_version_id = "data_v_test"
        self.data_versions = []

    def load_df(self):
        return self.df

    def get_arg(self, key, default=None):
        return self.arguments.get(key, default)


def _groups(*arrays):
    return {f"g{i}": np.asarray(a, dtype=float) for i, a in enumerate(arrays)}


# ============================================================
# Group 1 — the decision rule
# ============================================================

class TestDecision:
    def test_thresholds_are_expected(self):
        # Pin the agreed thresholds so a silent change is caught.
        assert _SMALL_GROUP_N == 30
        assert _HIGH_SKEW == 1.5

    def test_normal_data_never_switches(self):
        rng = np.random.default_rng(0)
        g = _groups(rng.normal(0, 1, 50), rng.normal(0, 1, 50))
        d = _decide_nonparametric(g, any_group_non_normal=False)
        assert d["switch_to_nonparametric"] is False

    def test_small_sample_with_nonnormal_switches(self):
        g = _groups([1, 2, 3, 4, 5], [2, 3, 4, 5, 6])  # n=5 each
        d = _decide_nonparametric(g, any_group_non_normal=True)
        assert d["switch_to_nonparametric"] is True
        assert d["small_sample"] is True

    def test_large_sample_mild_skew_does_not_switch(self):
        # Non-normal flagged (e.g. large-n Shapiro), but n>=30 and skew<1.5:
        # CLT applies, keep parametric.
        rng = np.random.default_rng(1)
        g = _groups(rng.normal(0, 1, 200), rng.normal(0.1, 1, 200))
        d = _decide_nonparametric(g, any_group_non_normal=True)
        assert d["switch_to_nonparametric"] is False
        assert d["small_sample"] is False
        assert d["high_skew"] is False

    def test_large_sample_severe_skew_switches(self):
        # n>=30 but heavy skew -> CLT inadequate -> switch.
        rng = np.random.default_rng(2)
        skewed = rng.lognormal(0, 1.0, 60)  # strong positive skew
        g = _groups(skewed, rng.lognormal(0, 1.0, 60))
        d = _decide_nonparametric(g, any_group_non_normal=True)
        assert d["high_skew"] is True
        assert d["switch_to_nonparametric"] is True

    def test_nonnormal_required_even_if_skewed(self):
        # If Shapiro did NOT flag non-normality, do not switch regardless.
        rng = np.random.default_rng(3)
        g = _groups(rng.normal(0, 1, 10), rng.normal(0, 1, 10))
        d = _decide_nonparametric(g, any_group_non_normal=False)
        assert d["switch_to_nonparametric"] is False

    def test_reasons_recorded_when_switching(self):
        g = _groups([1, 2, 3], [4, 5, 6])
        d = _decide_nonparametric(g, any_group_non_normal=True)
        assert d["switch_to_nonparametric"] is True
        assert len(d["reasons"]) >= 1


# ============================================================
# Group 2 — skew helper
# ============================================================

class TestSkew:
    def test_symmetric_near_zero(self):
        rng = np.random.default_rng(4)
        g = _groups(rng.normal(0, 1, 500))
        s = _max_abs_skew(g)
        assert s is not None and s < 0.5

    def test_skewed_positive(self):
        rng = np.random.default_rng(5)
        g = _groups(rng.lognormal(0, 1, 500))
        assert _max_abs_skew(g) > 1.0

    def test_tiny_groups_return_none(self):
        g = _groups([1, 2], [3])  # all < 3 obs
        assert _max_abs_skew(g) is None


# ============================================================
# Group 3 & 4 — end-to-end routing on the benchmark datasets
# ============================================================

@pytest.mark.skipif(
    not os.path.exists(os.path.join(DATASETS, "case4_nonnormal_two_group.parquet")),
    reason="benchmark datasets not present",
)
class TestEndToEndRouting:
    def _run(self, fname, args):
        df = pd.read_parquet(os.path.join(DATASETS, fname))
        return execute_statistical_group_comparison(DummyContext(df, args))

    def test_case4_switches_to_mann_whitney(self):
        out = self._run("case4_nonnormal_two_group.parquet",
                        {"target_col": "response_time", "group_col": "arm"})
        d = out["details"]
        assert "Mann-Whitney" in d["method"]
        assert d["nonparametric_switch"]["switch_to_nonparametric"] is True
        # Parametric test retained as secondary for transparency.
        assert d["secondary_test"] is not None
        assert "t-test" in d["secondary_test"]["method"]

    def test_case4_conclusion_agrees_both_tests(self):
        # Both primary (Mann-Whitney) and secondary (Welch t) should agree the
        # difference is not significant -- the switch changes the test, not the
        # honest conclusion.
        out = self._run("case4_nonnormal_two_group.parquet",
                        {"target_col": "response_time", "group_col": "arm"})
        d = out["details"]
        assert d["significant_at_alpha"] is False
        assert d["secondary_test"]["significant_at_alpha"] is False

    def test_case6_keeps_parametric(self):
        out = self._run("case6_effect_size_reporting.parquet",
                        {"target_col": "test_score", "group_col": "cohort"})
        d = out["details"]
        assert "t-test" in d["method"]
        assert d["nonparametric_switch"]["switch_to_nonparametric"] is False
        assert d["secondary_test"] is None

    def test_case1_anova_not_over_switched(self):
        # case1 has skew ~1.0 (< 1.5) and n=40 (>=30): must stay on the
        # variance-robust ANOVA path, not switch to Kruskal-Wallis.
        out = self._run("case1_unequal_variance_anova.parquet",
                        {"target_col": "outcome", "group_col": "treatment"})
        d = out["details"]
        assert "ANOVA" in d["method"]
        assert d["nonparametric_switch"]["switch_to_nonparametric"] is False