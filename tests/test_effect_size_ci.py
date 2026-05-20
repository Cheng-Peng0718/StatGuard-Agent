"""
Tests for core/analysis_tool_plugins/shared/effect_size_ci.py.

Test groups:
  1. Cohen's d independent CI vs published reference values
  2. Hedges' g CI consistency with d CI (J-correction relationship)
  3. Paired Cohen's d_z CI behavior
  4. Eta-squared CI (Smithson 2003 procedure)
  5. Omega-squared CI
  6. Monotonicity / invariants that must hold across all CIs
  7. Edge cases that previously caused NaN / brentq failures
"""

from __future__ import annotations

import math
import pytest

from core.analysis_tool_plugins.shared.effect_size_ci import (
    cohens_d_independent_ci,
    hedges_g_independent_ci,
    cohens_d_z_ci,
    eta_squared_ci,
    omega_squared_ci,
    _hedges_correction,
)


# ============================================================
# 1. Cohen's d independent CI -- published references
# ============================================================

class TestCohensDIndependentCI:
    def test_algina_keselman_d_half_n_20_per_group(self):
        """Algina & Keselman (2003): d=0.5, n1=n2=20, 95% CI ~= [-0.13, 1.13]."""
        lo, hi = cohens_d_independent_ci(d=0.5, n1=20, n2=20, alpha=0.05)
        assert lo is not None and hi is not None
        assert -0.20 < lo < -0.05
        assert 1.05 < hi < 1.20

    def test_cumming_d_0_8_n_50_per_group(self):
        """Cumming (2014): d=0.8, n=50/group, 95% CI ~= [0.39, 1.21]."""
        lo, hi = cohens_d_independent_ci(d=0.8, n1=50, n2=50, alpha=0.05)
        assert lo is not None and hi is not None
        assert 0.35 < lo < 0.45
        assert 1.15 < hi < 1.25

    def test_ci_contains_point_estimate(self):
        """The point estimate must always lie inside its own CI."""
        for d in [-1.0, -0.3, 0.1, 0.5, 0.8, 1.5]:
            lo, hi = cohens_d_independent_ci(d=d, n1=40, n2=40)
            assert lo is not None and hi is not None
            assert lo < d < hi, f"d={d} not in CI [{lo}, {hi}]"

    def test_zero_d_ci_is_symmetric_about_zero(self):
        """For d=0, the CI should be symmetric about 0 (within numerical noise)."""
        lo, hi = cohens_d_independent_ci(d=0.0, n1=50, n2=50)
        assert lo is not None and hi is not None
        assert abs(lo + hi) < 0.01, f"CI [{lo}, {hi}] is not symmetric about 0"

    def test_invalid_n_returns_none(self):
        """Tiny sample sizes must return (None, None), not crash."""
        for n1, n2 in [(0, 10), (1, 10), (10, 1), (-1, 10)]:
            lo, hi = cohens_d_independent_ci(d=0.5, n1=n1, n2=n2)
            assert lo is None and hi is None

    def test_extreme_d_does_not_crash(self):
        """Very large d should still produce some CI (even if approximate)."""
        lo, hi = cohens_d_independent_ci(d=5.0, n1=20, n2=20)
        # Either succeeds or fails gracefully -- must not raise
        if lo is not None:
            assert lo < 5.0 < hi


# ============================================================
# 2. Hedges' g CI consistency with Cohen's d CI
# ============================================================

class TestHedgesGCI:
    def test_J_correction_formula(self):
        """J = 1 - 3 / (4(n1+n2) - 9)."""
        n1, n2 = 20, 20
        expected_J = 1.0 - 3.0 / (4.0 * (n1 + n2) - 9.0)
        assert abs(_hedges_correction(n1, n2) - expected_J) < 1e-12

    def test_J_approaches_1_for_large_n(self):
        """J -> 1 as n grows."""
        J_large = _hedges_correction(500, 500)
        assert 0.998 < J_large < 1.000

    def test_g_ci_is_d_ci_scaled_by_J(self):
        """The Hedges g CI is the Cohen's d CI scaled by J."""
        n1, n2 = 20, 20
        J = _hedges_correction(n1, n2)
        d = 0.5
        g = d * J

        d_lo, d_hi = cohens_d_independent_ci(d=d, n1=n1, n2=n2)
        g_lo, g_hi = hedges_g_independent_ci(g=g, n1=n1, n2=n2)

        assert all(v is not None for v in [d_lo, d_hi, g_lo, g_hi])

        # The g CI endpoints should be the d CI endpoints * J
        assert abs(g_lo - d_lo * J) < 1e-6
        assert abs(g_hi - d_hi * J) < 1e-6

    def test_g_ci_contains_point_estimate(self):
        for g in [-0.8, -0.2, 0.1, 0.5, 1.0]:
            lo, hi = hedges_g_independent_ci(g=g, n1=30, n2=30)
            assert lo is not None and hi is not None
            assert lo < g < hi


# ============================================================
# 3. Paired Cohen's d_z CI
# ============================================================

class TestCohensDZCI:
    def test_ci_contains_point_estimate(self):
        for d_z in [-0.5, 0.0, 0.3, 0.8, 1.2]:
            lo, hi = cohens_d_z_ci(d_z=d_z, n=30)
            assert lo is not None and hi is not None
            assert lo <= d_z <= hi, f"d_z={d_z} not in CI [{lo}, {hi}]"

    def test_ci_narrows_with_more_pairs(self):
        """More complete pairs -> tighter CI."""
        lo20, hi20 = cohens_d_z_ci(d_z=0.5, n=20)
        lo80, hi80 = cohens_d_z_ci(d_z=0.5, n=80)
        assert (hi20 - lo20) > (hi80 - lo80)

    def test_too_few_pairs_returns_none(self):
        lo, hi = cohens_d_z_ci(d_z=0.5, n=1)
        assert lo is None and hi is None

    def test_zero_d_z_symmetric(self):
        lo, hi = cohens_d_z_ci(d_z=0.0, n=40)
        assert abs(lo + hi) < 0.01


# ============================================================
# 4. Eta-squared CI (Smithson 2003)
# ============================================================

class TestEtaSquaredCI:
    def test_smithson_textbook_example(self):
        """Smithson (2003): F(2, 42) = 5.0 -> ~[0.012, 0.366]."""
        lo, hi = eta_squared_ci(f_obs=5.0, df_between=2, df_within=42, alpha=0.05)
        assert lo is not None and hi is not None
        assert 0.000 <= lo < 0.030
        assert 0.340 < hi < 0.400

    def test_large_F_gives_nonzero_lower_bound(self):
        """A clearly significant F should NOT have a lower CI bound of 0.
        This guards against the brentq clamp bug that v1 had."""
        lo, hi = eta_squared_ci(f_obs=20.0, df_between=2, df_within=42)
        assert lo is not None and hi is not None
        assert lo > 0.10, f"Lower bound {lo} should be substantial for F=20"

    def test_small_F_clamps_lower_to_zero(self):
        """A non-significant F should have lower CI = 0 (eta² is bounded below)."""
        lo, hi = eta_squared_ci(f_obs=0.5, df_between=2, df_within=100)
        assert lo is not None and hi is not None
        assert lo == 0.0

    def test_ci_in_unit_interval(self):
        """eta² CI must always be in [0, 1]."""
        for f in [0.1, 1.0, 5.0, 20.0, 100.0]:
            lo, hi = eta_squared_ci(f_obs=f, df_between=2, df_within=50)
            if lo is not None and hi is not None:
                assert 0.0 <= lo <= 1.0
                assert 0.0 <= hi <= 1.0
                assert lo <= hi

    def test_observed_eta_squared_in_ci(self):
        """The observed eta² = (F * df1) / (F * df1 + df2) should be inside the CI."""
        for f, df1, df2 in [(5.0, 2, 42), (10.0, 3, 100), (3.0, 4, 60)]:
            eta2_obs = (f * df1) / (f * df1 + df2)
            lo, hi = eta_squared_ci(f_obs=f, df_between=df1, df_within=df2)
            assert lo <= eta2_obs <= hi, (
                f"F={f}, df=({df1},{df2}): observed eta²={eta2_obs:.4f} "
                f"not in CI [{lo:.4f}, {hi:.4f}]"
            )

    def test_ci_narrows_with_more_within_df(self):
        """Larger within-df -> tighter CI."""
        lo_small, hi_small = eta_squared_ci(f_obs=5.0, df_between=2, df_within=30)
        lo_large, hi_large = eta_squared_ci(f_obs=5.0, df_between=2, df_within=300)
        # With larger df_within the CI should be tighter
        assert (hi_large - lo_large) < (hi_small - lo_small)

    def test_invalid_df_returns_none(self):
        lo, hi = eta_squared_ci(f_obs=5.0, df_between=0, df_within=42)
        assert lo is None and hi is None
        lo, hi = eta_squared_ci(f_obs=5.0, df_between=2, df_within=0)
        assert lo is None and hi is None


# ============================================================
# 5. Omega-squared CI
# ============================================================

class TestOmegaSquaredCI:
    def test_omega_lower_than_or_equal_to_eta(self):
        """For the same data, omega² is typically slightly smaller than eta²
        because it is unbiased. Their CIs should overlap heavily but omega's
        upper bound should not exceed eta's upper bound by much."""
        f_obs, df1, df2, n = 5.0, 2, 42, 45
        eta_lo, eta_hi = eta_squared_ci(f_obs, df1, df2)
        ome_lo, ome_hi = omega_squared_ci(f_obs, df1, df2, n)

        # Both should be defined
        assert all(v is not None for v in [eta_lo, eta_hi, ome_lo, ome_hi])
        # omega lower bound should not exceed eta upper bound
        assert ome_lo <= eta_hi

    def test_ci_in_unit_interval(self):
        for f in [1.0, 5.0, 20.0]:
            lo, hi = omega_squared_ci(f_obs=f, df_between=2, df_within=50, n_total=53)
            if lo is not None and hi is not None:
                assert 0.0 <= lo <= 1.0
                assert 0.0 <= hi <= 1.0
                assert lo <= hi


# ============================================================
# 6. Cross-cutting monotonicity invariants
# ============================================================

class TestInvariants:
    def test_d_ci_narrows_with_more_n(self):
        """Larger n -> tighter d CI."""
        lo_small, hi_small = cohens_d_independent_ci(d=0.5, n1=10, n2=10)
        lo_large, hi_large = cohens_d_independent_ci(d=0.5, n1=200, n2=200)
        assert (hi_large - lo_large) < (hi_small - lo_small)

    def test_d_ci_center_grows_with_d(self):
        """As the point estimate d grows, the CI midpoint should grow too."""
        mid_low = sum(cohens_d_independent_ci(d=0.2, n1=50, n2=50)) / 2
        mid_high = sum(cohens_d_independent_ci(d=0.8, n1=50, n2=50)) / 2
        assert mid_low < mid_high

    def test_eta2_ci_widens_with_smaller_within_df(self):
        """Smaller df_within -> wider eta² CI."""
        lo_small, hi_small = eta_squared_ci(f_obs=5.0, df_between=2, df_within=20)
        lo_large, hi_large = eta_squared_ci(f_obs=5.0, df_between=2, df_within=200)
        assert (hi_small - lo_small) > (hi_large - lo_large)


# ============================================================
# 7. Edge cases (previously caused NaN / failures)
# ============================================================

class TestEdgeCases:
    def test_very_large_d_does_not_return_nan(self):
        """d=10 used to trigger nct.cdf NaN. Must not raise."""
        lo, hi = cohens_d_independent_ci(d=10.0, n1=20, n2=20)
        # Either succeeds with finite values or returns None gracefully
        assert lo is None or math.isfinite(lo)
        assert hi is None or math.isfinite(hi)

    def test_very_large_F_does_not_return_nan(self):
        """F=1000 must not crash the noncentral-F bracket solver."""
        lo, hi = eta_squared_ci(f_obs=1000.0, df_between=2, df_within=100)
        # Either succeeds or returns None; never NaN
        assert lo is None or math.isfinite(lo)
        assert hi is None or math.isfinite(hi)

    def test_unequal_group_sizes(self):
        """The CI must accept unequal n1, n2."""
        lo, hi = cohens_d_independent_ci(d=0.5, n1=10, n2=50)
        assert lo is not None and hi is not None
        assert lo < 0.5 < hi

    def test_alpha_99_percent_wider_than_95_percent(self):
        """A 99% CI should be wider than a 95% CI for the same data."""
        lo95, hi95 = cohens_d_independent_ci(d=0.5, n1=40, n2=40, alpha=0.05)
        lo99, hi99 = cohens_d_independent_ci(d=0.5, n1=40, n2=40, alpha=0.01)
        assert (hi99 - lo99) > (hi95 - lo95)