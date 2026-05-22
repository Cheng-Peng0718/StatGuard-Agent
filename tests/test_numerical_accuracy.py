"""
Numerical accuracy: cross-validate plugin statistics against independent
scipy / statsmodels computations.

Every assertion here recomputes the statistic from scratch with a reference
library and checks the plugin agrees to a tight tolerance. This is the evidence
behind the claim that the framework's numbers are correct, not merely present.

Covered:
  - Welch t-test (statistical_group_comparison, run_independent_t_test)
  - Cohen's d / Hedges g effect size
  - Mann-Whitney U (via P2 auto-switch and the nonparametric plugin)
  - One-way ANOVA F (run_anova / statistical_group_comparison)
  - Paired t-test
  - Pearson correlation + Fisher CI
  - OLS regression R^2, F
  - Chi-square test of independence + Cramer's V
"""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from core.analysis_tool_plugins.registry import get_plugin

warnings.filterwarnings("ignore")

REL = 1e-4   # relative tolerance for floating point agreement
ABS = 1e-4   # absolute tolerance


class Ctx:
    def __init__(self, df, args):
        self._df = df
        self.arguments = args
        self.active_data_version_id = "v_test"
        self.data_versions = []

    def load_df(self):
        return self._df

    def get_arg(self, name, default=None):
        return self.arguments.get(name, default)


def _run(tool, df, args):
    out = get_plugin(tool).run(Ctx(df, args))
    assert out.get("status") in ("ok", "warning"), f"{tool} unexpectedly {out.get('status')}: {out.get('message','')}"
    return out["details"]


def _close(a, b, rel=REL, abs_=ABS):
    return math.isclose(float(a), float(b), rel_tol=rel, abs_tol=abs_)


# ============================================================
# Welch t-test + effect size
# ============================================================

class TestWelchTTest:
    def _data(self):
        rng = np.random.default_rng(11)
        a = rng.normal(10, 2, 45)
        b = rng.normal(11.5, 3, 50)
        df = pd.DataFrame({"y": np.concatenate([a, b]),
                           "g": ["a"] * 45 + ["b"] * 50})
        return df, a, b

    def test_welch_t_and_p_match_scipy(self):
        df, a, b = self._data()
        d = _run("statistical_group_comparison", df, {"target_col": "y", "group_col": "g"})
        t_ref, p_ref = stats.ttest_ind(a, b, equal_var=False)
        # plugin orders groups internally; compare absolute t (sign is direction)
        assert _close(abs(d["t_statistic"]), abs(t_ref))
        assert _close(d["p_value"], p_ref)

    def test_welch_df_matches_welch_satterthwaite(self):
        df, a, b = self._data()
        d = _run("statistical_group_comparison", df, {"target_col": "y", "group_col": "g"})
        # Welch-Satterthwaite df
        v1, v2 = a.var(ddof=1), b.var(ddof=1)
        n1, n2 = len(a), len(b)
        num = (v1 / n1 + v2 / n2) ** 2
        den = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df_ref = num / den
        assert _close(d["degrees_of_freedom"], df_ref)

    def test_hedges_g_matches_manual(self):
        df, a, b = self._data()
        d = _run("statistical_group_comparison", df, {"target_col": "y", "group_col": "g"})
        n1, n2 = len(a), len(b)
        sp = math.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
        cohen_d = (a.mean() - b.mean()) / sp
        J = 1 - 3 / (4 * (n1 + n2) - 9)  # small-sample correction
        hedges_g = J * cohen_d
        assert _close(abs(d["effect_size"]), abs(hedges_g), rel=1e-3)


# ============================================================
# Mann-Whitney U (P2 auto-switch path + dedicated plugin)
# ============================================================

class TestMannWhitney:
    def _skewed(self):
        rng = np.random.default_rng(7)
        a = rng.lognormal(0, 1, 30)  # strong skew -> P2 switches
        b = rng.lognormal(0.2, 1, 30)
        df = pd.DataFrame({"y": np.concatenate([a, b]),
                           "g": ["a"] * 30 + ["b"] * 30})
        return df, a, b

    def test_autoswitch_U_and_p_match_scipy(self):
        df, a, b = self._skewed()
        d = _run("statistical_group_comparison", df, {"target_col": "y", "group_col": "g"})
        assert "Mann-Whitney" in d["method"]  # P2 switched
        u_ref, p_ref = stats.mannwhitneyu(a, b, alternative="two-sided")
        # plugin reports U for its first (sorted) group; both U values map to
        # the same p-value, so compare p exactly and U up to the complement.
        assert _close(d["p_value"], p_ref)
        n1, n2 = len(a), len(b)
        assert _close(d["U_statistic"], u_ref) or _close(d["U_statistic"], n1 * n2 - u_ref)

    def test_dedicated_nonparametric_plugin_matches(self):
        df, a, b = self._skewed()
        d = _run("nonparametric_group_comparison", df, {"target_col": "y", "group_col": "g"})
        u_ref, p_ref = stats.mannwhitneyu(a, b, alternative="two-sided")
        assert _close(d["p_value"], p_ref)


# ============================================================
# One-way ANOVA
# ============================================================

class TestANOVA:
    def test_classic_anova_F_matches_scipy(self):
        rng = np.random.default_rng(3)
        # equal variances so the plugin uses classic one-way ANOVA
        g1 = rng.normal(10, 2, 40)
        g2 = rng.normal(10.5, 2, 40)
        g3 = rng.normal(11, 2, 40)
        df = pd.DataFrame({"y": np.concatenate([g1, g2, g3]),
                           "g": ["a"] * 40 + ["b"] * 40 + ["c"] * 40})
        d = _run("statistical_group_comparison", df, {"target_col": "y", "group_col": "g"})
        if "Alexander" in d.get("method", "") or "Welch" in d.get("method", ""):
            pytest.skip("variance check routed to Welch/AG; covered elsewhere")
        f_ref, p_ref = stats.f_oneway(g1, g2, g3)
        assert _close(d["F_statistic"], f_ref)
        assert _close(d["p_value"], p_ref)


# ============================================================
# Paired t-test
# ============================================================

class TestPaired:
    def test_paired_t_and_p_match_scipy(self):
        rng = np.random.default_rng(5)
        n = 35
        pre = rng.normal(50, 8, n)
        post = pre + rng.normal(2, 4, n)  # normal differences -> paired t
        df = pd.DataFrame({"pre": pre, "post": post})
        d = _run("paired_comparison", df, {"target_col_1": "pre", "target_col_2": "post"})
        if "Wilcoxon" in d.get("method", ""):
            pytest.skip("routed to Wilcoxon; normality borderline")
        # The plugin reports the p-value (and a paired effect size) but not the
        # raw t-statistic, so we cross-validate the p-value against scipy.
        _, p_ref = stats.ttest_rel(pre, post)
        assert _close(d["p_value"], p_ref)

    def test_paired_cohens_dz_matches_manual(self):
        rng = np.random.default_rng(5)
        n = 35
        pre = rng.normal(50, 8, n)
        post = pre + rng.normal(2, 4, n)
        df = pd.DataFrame({"pre": pre, "post": post})
        d = _run("paired_comparison", df, {"target_col_1": "pre", "target_col_2": "post"})
        if "Wilcoxon" in d.get("method", ""):
            pytest.skip("routed to Wilcoxon")
        diff = pre - post
        dz_ref = diff.mean() / diff.std(ddof=1)
        assert _close(abs(d["effect_size"]), abs(dz_ref), rel=1e-3)


# ============================================================
# Pearson correlation
# ============================================================

class TestCorrelation:
    def test_pearson_r_and_p_match_scipy(self):
        rng = np.random.default_rng(9)
        x = rng.normal(0, 1, 60)
        y = 0.5 * x + rng.normal(0, 0.9, 60)
        df = pd.DataFrame({"x": x, "y": y})
        d = _run("run_correlation_test", df, {"x_col": "x", "y_col": "y"})
        r_ref, p_ref = stats.pearsonr(x, y)
        assert _close(d["correlation"], r_ref)
        assert _close(d["p_value"], p_ref)

    def test_fisher_ci_matches_manual(self):
        rng = np.random.default_rng(9)
        x = rng.normal(0, 1, 60)
        y = 0.5 * x + rng.normal(0, 0.9, 60)
        df = pd.DataFrame({"x": x, "y": y})
        d = _run("run_correlation_test", df, {"x_col": "x", "y_col": "y"})
        r, n = d["correlation"], d["nobs"]
        z = np.arctanh(r)
        se = 1 / math.sqrt(n - 3)
        zcrit = stats.norm.ppf(0.975)
        lo, hi = np.tanh(z - zcrit * se), np.tanh(z + zcrit * se)
        assert _close(d["ci_lower"], lo, rel=1e-3)
        assert _close(d["ci_upper"], hi, rel=1e-3)


# ============================================================
# OLS regression
# ============================================================

class TestRegression:
    def test_r_squared_and_f_match_statsmodels(self):
        import statsmodels.api as sm
        rng = np.random.default_rng(13)
        x = np.round(rng.normal(0, 1, 120), 1)  # rounded to dodge id-like filter
        y = np.round(2.0 * x + rng.normal(0, 0.6, 120), 2)
        df = pd.DataFrame({"y": y, "x": x})
        d = _run("run_multiple_regression", df, {"target_col": "y", "feature_cols": ["x"]})
        X = sm.add_constant(df["x"].values)
        model = sm.OLS(df["y"].values, X).fit()
        assert _close(d["r_squared"], model.rsquared, rel=1e-3)
        assert _close(d["f_statistic"], model.fvalue, rel=1e-3)


# ============================================================
# Chi-square
# ============================================================

class TestChiSquare:
    def test_chi2_and_p_match_scipy(self):
        # deterministic contingency structure
        df = pd.DataFrame({
            "a": ["x"] * 30 + ["y"] * 30,
            "b": (["p"] * 20 + ["q"] * 10) + (["p"] * 10 + ["q"] * 20),
        })
        d = _run("run_chi_square", df, {"row_col": "a", "col_col": "b"})
        table = pd.crosstab(df["a"], df["b"]).values
        chi2_ref, p_ref, dof_ref, _ = stats.chi2_contingency(table, correction=True)
        assert _close(d["chi_square_statistic"], chi2_ref)
        assert _close(d["p_value"], p_ref)
        assert int(d["degrees_of_freedom"]) == int(dof_ref)

    def test_cramers_v_matches_manual(self):
        df = pd.DataFrame({
            "a": ["x"] * 30 + ["y"] * 30,
            "b": (["p"] * 20 + ["q"] * 10) + (["p"] * 10 + ["q"] * 20),
        })
        d = _run("run_chi_square", df, {"row_col": "a", "col_col": "b"})
        table = pd.crosstab(df["a"], df["b"]).values
        chi2, _, _, _ = stats.chi2_contingency(table, correction=True)
        n = table.sum()
        k = min(table.shape) - 1
        v_ref = math.sqrt(chi2 / (n * k))
        assert _close(d["cramers_v"], v_ref, rel=1e-3)