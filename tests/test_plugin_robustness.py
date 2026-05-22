"""
No-error robustness: plugins must never raise on degenerate or malformed input.

A statistical tool that crashes on an empty group, a single row, all-NaN data,
or a constant column is not production-grade. These tests feed each plugin a
battery of hostile-but-type-valid inputs and assert it returns a structured
result (with a status) instead of throwing an uncaught exception.

The contract under test: plugin.run(context) ALWAYS returns a dict with a
"status" key. Bad data should yield status "blocked"/"error" with a message,
never a traceback.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from core.analysis_tool_plugins.registry import get_plugin, PLUGIN_REGISTRY

warnings.filterwarnings("ignore")


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


def _run_no_raise(tool, df, args):
    """Run a plugin and assert it returns a structured result, never raises."""
    plugin = get_plugin(tool)
    try:
        out = plugin.run(Ctx(df, args))
    except Exception as exc:  # pragma: no cover - this is the failure we test for
        pytest.fail(f"{tool} raised on input: {type(exc).__name__}: {exc}")
    assert isinstance(out, dict), f"{tool} did not return a dict"
    assert "status" in out, f"{tool} result missing 'status'"
    return out


# The statistical plugins that take (target/group/col) style numeric input.
NUMERIC_GROUP_TOOLS = [
    ("statistical_group_comparison", {"target_col": "y", "group_col": "g"}),
    ("nonparametric_group_comparison", {"target_col": "y", "group_col": "g"}),
    ("run_independent_t_test", {"target_col": "y", "group_col": "g",
                                "group1_val": "a", "group2_val": "b"}),
    ("run_anova", {"target_col": "y", "group_col": "g"}),
]


# ============================================================
# Degenerate dataframes
# ============================================================

def _empty_df():
    return pd.DataFrame({"y": [], "g": []})


def _single_row():
    return pd.DataFrame({"y": [1.0], "g": ["a"]})


def _all_nan():
    return pd.DataFrame({"y": [np.nan] * 10, "g": ["a"] * 5 + ["b"] * 5})


def _single_group():
    return pd.DataFrame({"y": [1.0, 2, 3, 4, 5], "g": ["a"] * 5})


def _constant_values():
    return pd.DataFrame({"y": [3.0] * 20, "g": ["a"] * 10 + ["b"] * 10})


def _with_inf():
    return pd.DataFrame({"y": [1.0, 2, np.inf, 4, -np.inf, 6, 7, 8],
                         "g": ["a"] * 4 + ["b"] * 4})


def _tiny_groups():
    return pd.DataFrame({"y": [1.0, 2, 3], "g": ["a", "b", "a"]})


DEGENERATE = [
    ("empty", _empty_df),
    ("single_row", _single_row),
    ("all_nan", _all_nan),
    ("single_group", _single_group),
    ("constant", _constant_values),
    ("with_inf", _with_inf),
    ("tiny_groups", _tiny_groups),
]


class TestGroupToolsRobustness:
    @pytest.mark.parametrize("tool,args", NUMERIC_GROUP_TOOLS)
    @pytest.mark.parametrize("label,maker", DEGENERATE)
    def test_no_raise_on_degenerate(self, tool, args, label, maker):
        out = _run_no_raise(tool, maker(), args)
        # Degenerate inputs should be blocked/error, not silently "ok" with junk
        # (we don't force a specific status, just that it's a clean structured
        # result -- the no-raise guarantee is the contract here).
        assert out["status"] in ("ok", "warning", "blocked", "error")


# ============================================================
# Regression / correlation / chi-square robustness
# ============================================================

class TestOtherStatToolsRobustness:
    def test_regression_empty(self):
        _run_no_raise("run_multiple_regression", _empty_df(),
                      {"target_col": "y", "feature_cols": ["g"]})

    def test_regression_constant_target(self):
        df = pd.DataFrame({"y": [5.0] * 30, "x": np.round(np.linspace(0, 1, 30), 1)})
        _run_no_raise("run_multiple_regression", df,
                      {"target_col": "y", "feature_cols": ["x"]})

    def test_regression_missing_feature_col(self):
        df = pd.DataFrame({"y": [1.0, 2, 3], "x": [1.0, 2, 3]})
        _run_no_raise("run_multiple_regression", df,
                      {"target_col": "y", "feature_cols": ["does_not_exist"]})

    def test_correlation_constant_column(self):
        df = pd.DataFrame({"x": [1.0] * 20, "y": np.random.default_rng(0).normal(0, 1, 20)})
        _run_no_raise("run_correlation_test", df, {"x_col": "x", "y_col": "y"})

    def test_correlation_single_point(self):
        df = pd.DataFrame({"x": [1.0], "y": [2.0]})
        _run_no_raise("run_correlation_test", df, {"x_col": "x", "y_col": "y"})

    def test_chi_square_empty(self):
        _run_no_raise("run_chi_square", _empty_df(), {"row_col": "y", "col_col": "g"})

    def test_chi_square_single_level(self):
        df = pd.DataFrame({"a": ["x"] * 20, "b": ["p"] * 20})
        _run_no_raise("run_chi_square", df, {"row_col": "a", "col_col": "b"})

    def test_paired_mismatched_lengths_via_nan(self):
        df = pd.DataFrame({"pre": [1.0, 2, np.nan, 4], "post": [np.nan, 2, 3, 4]})
        _run_no_raise("paired_comparison", df,
                      {"target_col_1": "pre", "target_col_2": "post"})

    def test_power_analysis_degenerate(self):
        _run_no_raise("power_analysis", _single_row(),
                      {"effect_size": 0.0, "sample_size": 1})


# ============================================================
# Full-registry smoke: nothing in the registry is import-broken
# ============================================================

class TestRegistrySmoke:
    def test_every_plugin_is_retrievable(self):
        for name in PLUGIN_REGISTRY:
            p = get_plugin(name)
            assert p is not None, f"{name} not retrievable"
            assert hasattr(p, "run"), f"{name} has no run()"

    def test_descriptive_tools_run_on_normal_data(self):
        # Tools that should succeed on a clean, ordinary dataframe.
        rng = np.random.default_rng(1)
        df = pd.DataFrame({
            "num1": rng.normal(10, 2, 50),
            "num2": rng.normal(5, 1, 50),
            "cat": rng.choice(["a", "b", "c"], 50),
        })
        for tool, args in [
            ("get_summary_stats", {}),
            ("inspect_dataset", {}),
            ("missingness_report", {}),
            ("get_correlation_matrix", {}),
            ("summarize_columns", {}),
        ]:
            out = _run_no_raise(tool, df, args)
            # These should genuinely succeed on clean data.
            assert out["status"] in ("ok", "warning"), \
                f"{tool} unexpectedly {out['status']} on clean data"

    def test_groupby_summary_no_raise(self):
        # groupby_summary loads via the data-version layer rather than load_df,
        # so under a lightweight context it blocks cleanly; we only assert it
        # does not raise.
        rng = np.random.default_rng(1)
        df = pd.DataFrame({"num1": rng.normal(10, 2, 50),
                           "cat": rng.choice(["a", "b", "c"], 50)})
        _run_no_raise("groupby_summary", df,
                      {"group_cols": ["cat"], "value_col": "num1"})