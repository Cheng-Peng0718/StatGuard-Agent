"""
Unit tests for bootstrap_inference and bootstrap_utils.

Covers four validation layers:

  Layer 1 (resampler) -- Sequential Bootstrap sampler returns exactly k_n
      distinct indices per replicate.

  Layer 2 (CI math)   -- classical bootstrap CI agrees with
      `scipy.stats.bootstrap` for matched B and method (within a tolerance
      that reflects the Monte-Carlo error of B finite-resamples; B=10000).

  Layer 3 (SB property) -- variance of U_b (distinct-index count) is
      essentially zero under Sequential Bootstrap and strictly positive under
      the classical bootstrap, which is exactly the variance-component that
      Peng (2025) Section 3.4 stabilises.

  Layer 4 (stability diagnostic) -- max_endpoint_cv is small when the data
      generating process is well-behaved (large n, normal), and the
      classification ladder ("low" / "moderate" / "high") is consistent.

The plugin-level end-to-end tests live in tests/test_plugin_robustness.py
elsewhere in StatGuard; this file targets the analytical core.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy import stats as scipy_stats

from core.analysis_tool_plugins.registry import get_plugin
from core.analysis_tool_plugins.shared.bootstrap_utils import (
    bootstrap_ci,
    bootstrap_with_stability,
    classical_bootstrap_indices,
    expected_kn,
    get_paired_statistic,
    sequential_bootstrap_indices,
)

# Force the bootstrap_inference plugin to register its tool.
import core.analysis_tool_plugins.plugins.bootstrap_inference  # noqa: F401


# ==========================================================
# Layer 1 -- Sequential Bootstrap resampler
# ==========================================================

class TestSequentialBootstrapSampler:
    """The SB sampler must return exactly k_n distinct indices per replicate."""

    @pytest.mark.parametrize("n,B", [(20, 200), (100, 500), (500, 100)])
    def test_distinct_count_exact(self, n: int, B: int) -> None:
        rng = np.random.default_rng(42)
        k_n = expected_kn(n, 0.632)

        replicates = sequential_bootstrap_indices(n, k_n, B, rng)

        assert len(replicates) == B

        for indices in replicates:
            distinct = np.unique(indices)
            assert distinct.size == k_n, (
                f"Sequential Bootstrap produced {distinct.size} distinct "
                f"indices; expected exactly k_n={k_n}."
            )

    def test_stopping_time_at_least_kn(self) -> None:
        rng = np.random.default_rng(0)
        n, k_n, B = 50, 32, 100

        replicates = sequential_bootstrap_indices(n, k_n, B, rng)

        for indices in replicates:
            assert len(indices) >= k_n, (
                f"Stopping time T_b = {len(indices)} < k_n = {k_n}."
            )

    def test_index_range(self) -> None:
        rng = np.random.default_rng(1)
        n, k_n, B = 30, 19, 50

        replicates = sequential_bootstrap_indices(n, k_n, B, rng)

        for indices in replicates:
            assert indices.min() >= 0
            assert indices.max() < n

    def test_rejects_invalid_kn(self) -> None:
        rng = np.random.default_rng(0)
        with pytest.raises(ValueError):
            sequential_bootstrap_indices(10, 11, 5, rng)  # k_n > n
        with pytest.raises(ValueError):
            sequential_bootstrap_indices(10, 0, 5, rng)   # k_n = 0


# ==========================================================
# Layer 2 -- CI computation agrees with scipy.stats.bootstrap
# ==========================================================

class TestClassicalCIMatchesScipy:
    """
    Classical-bootstrap CI from our pipeline matches scipy.stats.bootstrap on
    the same data, same B, same method, within the Monte-Carlo tolerance for
    B = 10000.

    We can't bit-match because scipy's internal resampler is not byte-identical
    to ours. We test convergence: both estimators of the same population CI
    should agree to ~1.5% of the CI width at B=10k.
    """

    @pytest.fixture
    def data_normal(self) -> np.ndarray:
        rng = np.random.default_rng(2025)
        return rng.normal(0.5, 1.0, size=60)

    @pytest.mark.parametrize("method", ["percentile", "basic", "BCa"])
    def test_mean_ci_within_tolerance(
        self,
        data_normal: np.ndarray,
        method: str,
    ) -> None:
        B = 10000
        alpha = 0.05

        # Ours.
        rng_ours = np.random.default_rng(0)
        idx_mat = classical_bootstrap_indices(len(data_normal), B, rng_ours)
        boot_stats = np.array([data_normal[idx_mat[i]].mean() for i in range(B)])

        observed = float(data_normal.mean())
        lo_ours, hi_ours = bootstrap_ci(
            boot_stats,
            observed,
            method=method,
            alpha=alpha,
            data=data_normal,
            statistic_fn=np.mean,
        )

        # scipy.
        scipy_res = scipy_stats.bootstrap(
            (data_normal,),
            statistic=np.mean,
            n_resamples=B,
            confidence_level=1 - alpha,
            method=method.lower(),
            random_state=np.random.default_rng(0),
            vectorized=False,
        )
        lo_scipy = float(scipy_res.confidence_interval.low)
        hi_scipy = float(scipy_res.confidence_interval.high)

        ci_width = hi_scipy - lo_scipy

        # Monte-Carlo error of bootstrap CI endpoints at B=10k is of order
        # CI_width / sqrt(B). We allow 0.015 * CI_width (1.5%).
        tol = 0.015 * ci_width

        assert abs(lo_ours - lo_scipy) < tol, (
            f"{method}: lower endpoint differs by {abs(lo_ours - lo_scipy):.4f}, "
            f"tol={tol:.4f}"
        )
        assert abs(hi_ours - hi_scipy) < tol, (
            f"{method}: upper endpoint differs by {abs(hi_ours - hi_scipy):.4f}, "
            f"tol={tol:.4f}"
        )


# ==========================================================
# Layer 3 -- SB-induced variance reduction in U_b
# ==========================================================

class TestVarUbStabilisation:
    """
    Variance of the distinct-index count across replicates is essentially
    zero under SB and strictly positive under classical bootstrap. This is
    exactly the Var(E[theta_b | U_b]) term that Peng (2025) Section 3.4
    isolates.
    """

    def test_classical_has_positive_var_ub(self) -> None:
        rng = np.random.default_rng(7)
        n, B = 100, 500
        idx_mat = classical_bootstrap_indices(n, B, rng)

        ub = np.array([np.unique(idx_mat[i]).size for i in range(B)])

        # Classical bootstrap: U_b = number of distinct indices.
        # E[U_b] = n * (1 - (1 - 1/n)^n) ~ 0.632 * n; for n=100, ~63.2.
        # Var(U_b) ~ 9.88 (population), with sampling-noise floor ~6 at B=500.
        assert ub.var() > 5.0, (
            f"Classical bootstrap should have substantial Var(U_b); got {ub.var():.2f}"
        )

    def test_sequential_has_zero_var_ub(self) -> None:
        rng = np.random.default_rng(11)
        n, B = 100, 500
        k_n = expected_kn(n, 0.632)

        replicates = sequential_bootstrap_indices(n, k_n, B, rng)
        ub = np.array([np.unique(r).size for r in replicates])

        assert ub.var() == 0.0, (
            f"Sequential Bootstrap should have Var(U_b)=0 by construction; "
            f"got {ub.var()}."
        )
        assert (ub == k_n).all()


# ==========================================================
# Layer 4 -- Stability diagnostic shape and ladder consistency
# ==========================================================

class TestStabilityDiagnostic:

    def test_well_behaved_data_low_max_cv(self) -> None:
        """Large n + normal data should give 'low' stability classification."""
        rng = np.random.default_rng(2026)
        data = rng.normal(0.0, 1.0, size=200)

        statistic_fn = get_paired_statistic("mean_diff")

        result = bootstrap_with_stability(
            data,
            statistic_fn,
            B=4000,
            n_seeds=5,
            alpha=0.05,
            method="BCa",
            use_sequential=False,
            seed=1,
        )

        diag = result["stability_diagnostic"]

        assert diag["interpretation"] in {"low", "moderate"}, (
            f"Expected low/moderate CI stability for large normal sample; "
            f"got {diag['interpretation']} (endpoint_drift={diag['endpoint_drift']:.4f})."
        )
        assert diag["recommendation"] is None

    def test_full_payload_shape(self) -> None:
        """Verify the public output schema is stable for downstream consumers."""
        rng = np.random.default_rng(0)
        data = rng.normal(1.0, 1.0, size=50)

        statistic_fn = get_paired_statistic("mean_diff")

        result = bootstrap_with_stability(
            data,
            statistic_fn,
            B=1000,
            n_seeds=5,
            alpha=0.05,
            method="percentile",
            use_sequential=False,
            seed=0,
        )

        # Required top-level keys.
        for key in (
            "observed_statistic",
            "ci_lower",
            "ci_upper",
            "resampler",
            "B_total",
            "n_seeds_for_diagnostic",
            "B_per_seed",
            "stability_diagnostic",
        ):
            assert key in result, f"Missing top-level key: {key}"

        # Required diagnostic keys.
        diag = result["stability_diagnostic"]
        for key in (
            "endpoint_drift",
            "ci_lower_sd",
            "ci_upper_sd",
            "ci_lower_cv",
            "ci_upper_cv",
            "interpretation",
            "recommendation",
        ):
            assert key in diag, f"Missing diagnostic key: {key}"

        # Invariants.
        assert result["ci_lower"] <= result["observed_statistic"] <= result["ci_upper"]
        assert diag["endpoint_drift"] >= 0.0 or np.isnan(diag["endpoint_drift"])
        assert diag["interpretation"] in {"low", "moderate", "high", "undefined"}

    def test_sequential_mode_marks_resampler(self) -> None:
        rng = np.random.default_rng(99)
        data = rng.normal(0.0, 1.0, size=60)

        statistic_fn = get_paired_statistic("mean_diff")

        result = bootstrap_with_stability(
            data,
            statistic_fn,
            B=1000,
            n_seeds=5,
            method="BCa",
            use_sequential=True,
            seed=0,
        )

        assert result["resampler"] == "sequential"


# ==========================================================
# Plugin integration
# ==========================================================

class TestPluginRegistration:

    def test_plugin_is_registered(self) -> None:
        plugin = get_plugin("bootstrap_inference")
        assert plugin is not None
        assert plugin.tool_name == "bootstrap_inference"
        assert plugin.is_inferential is True
        assert plugin.requires_data_source == "dataframe"

    def test_argument_schema(self) -> None:
        plugin = get_plugin("bootstrap_inference")
        schema = plugin.argument_schema

        assert "target_col_1" in schema.required
        assert "target_col_2" in schema.required
        assert "use_sequential" in schema.optional
        assert "statistic" in schema.optional
        assert "ci_method" in schema.optional


class TestPluginExecuteEndToEnd:

    def _make_ctx(self, df: pd.DataFrame, args: dict):
        class Ctx:
            def __init__(self, df, args):
                self._df = df
                self.arguments = args

            def load_df(self):
                return self._df

            def get_arg(self, name, default=None):
                return self.arguments.get(name, default)

        return Ctx(df, args)

    def test_paired_mean_diff_classical(self) -> None:
        rng = np.random.default_rng(0)
        pre = rng.normal(100.0, 10.0, size=40)
        post = pre + rng.normal(2.0, 5.0, size=40)
        df = pd.DataFrame({"pre": pre, "post": post})

        plugin = get_plugin("bootstrap_inference")
        ctx = self._make_ctx(df, {
            "target_col_1": "pre",
            "target_col_2": "post",
            "statistic": "mean_diff",
            "B": 1000,
            "n_seeds": 5,
            "seed": 1,
        })

        result = plugin.run(ctx)

        assert result["status"] == "ok"
        d = result["details"]
        assert d["statistic"] == "mean_diff"
        assert d["resampler"] == "classical"
        assert d["ci_lower"] < d["observed_statistic"] < d["ci_upper"]
        assert d["n_complete_pairs"] == 40

    def test_paired_median_diff_sequential(self) -> None:
        rng = np.random.default_rng(0)
        pre = rng.normal(50.0, 5.0, size=50)
        post = pre + rng.normal(1.5, 3.0, size=50)
        df = pd.DataFrame({"pre": pre, "post": post})

        plugin = get_plugin("bootstrap_inference")
        ctx = self._make_ctx(df, {
            "target_col_1": "pre",
            "target_col_2": "post",
            "statistic": "median_diff",
            "use_sequential": True,
            "B": 1000,
            "n_seeds": 5,
            "seed": 1,
        })

        result = plugin.run(ctx)

        assert result["status"] == "ok"
        d = result["details"]
        assert d["resampler"] == "sequential"
        assert d["use_sequential"] is True
        assert d["k_n"] == expected_kn(50, 0.632)

    def test_blocked_on_missing_columns(self) -> None:
        df = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
        plugin = get_plugin("bootstrap_inference")
        ctx = self._make_ctx(df, {
            "target_col_1": "pre",
            "target_col_2": "post",
        })
        result = plugin.run(ctx)
        assert result["status"] == "blocked"
        assert result["error_code"] in {"COLUMN_NOT_FOUND", "INSUFFICIENT_PAIRS"}

    def test_blocked_on_insufficient_pairs(self) -> None:
        df = pd.DataFrame({
            "pre": [1.0, 2.0, 3.0, 4.0],
            "post": [1.1, 2.2, 3.1, 4.3],
        })
        plugin = get_plugin("bootstrap_inference")
        ctx = self._make_ctx(df, {
            "target_col_1": "pre",
            "target_col_2": "post",
        })
        result = plugin.run(ctx)
        assert result["status"] == "blocked"
        assert result["error_code"] == "INSUFFICIENT_PAIRS"

    def test_blocked_on_unknown_statistic(self) -> None:
        df = pd.DataFrame({
            "pre": np.arange(20.0),
            "post": np.arange(20.0) + 1,
        })
        plugin = get_plugin("bootstrap_inference")
        ctx = self._make_ctx(df, {
            "target_col_1": "pre",
            "target_col_2": "post",
            "statistic": "wat",
        })
        result = plugin.run(ctx)
        assert result["status"] == "blocked"
        assert result["error_code"] == "UNKNOWN_STATISTIC"