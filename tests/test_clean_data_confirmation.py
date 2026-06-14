"""P3-A: per-action review gating for clean_data.

Only DESTRUCTIVE actions (dropping >5% of rows, imputing a >20%-missing column)
require human review; cast/dedup/standardize/clip run freely. Post-execution
guardrails report the ACTUAL impact for transparency.
"""
from types import SimpleNamespace

import core.analysis_tool_plugins  # noqa: F401
from core.analysis_tool_plugins.registry import get_plugin
from core.analysis_tool_plugins.plugins.clean_data import (
    clean_data_confirmation_policy,
    evaluate_clean_data_guardrails,
)
from verifiers.validators import verify


def _action(**args):
    return SimpleNamespace(tool_name="clean_data", arguments=args,
                           action_id="act_test")


def _profile(missing_by_col, n=100):
    return SimpleNamespace(
        columns={c: {"missing_rate": r, "name": c} for c, r in missing_by_col.items()},
        n_rows=n, n_cols=len(missing_by_col),
        dataset_name="test",
    )


# --------------------------------------------------------------------------
# confirmation policy
# --------------------------------------------------------------------------
def test_drop_over_5pct_needs_review():
    needs, reason = clean_data_confirmation_policy(
        _action(action_type="drop", strategy="rows", columns=["x"]),
        _profile({"x": 0.12}), None)
    assert needs is True and "%" in reason


def test_drop_under_5pct_allowed():
    needs, _ = clean_data_confirmation_policy(
        _action(action_type="drop", strategy="rows", columns=["x"]),
        _profile({"x": 0.02}), None)
    assert needs is False


def test_impute_over_20pct_needs_review():
    needs, reason = clean_data_confirmation_policy(
        _action(action_type="impute", strategy="mean", columns=["x"]),
        _profile({"x": 0.31}), None)
    assert needs is True and "x" in reason


def test_impute_under_20pct_allowed():
    needs, _ = clean_data_confirmation_policy(
        _action(action_type="impute", strategy="median", columns=["x"]),
        _profile({"x": 0.10}), None)
    assert needs is False


def test_safe_actions_allowed_even_with_high_missingness():
    for at, st in [("cast", "numeric"), ("dedup", "rows"),
                   ("standardize", "categories"), ("clip", "outliers")]:
        needs, _ = clean_data_confirmation_policy(
            _action(action_type=at, strategy=st, columns=["x"]),
            _profile({"x": 0.9}), None)
        assert needs is False, f"{at} should be allowed"


def test_drop_all_columns_uses_worst_column():
    # no columns -> all columns considered; worst missingness drives the decision
    needs, _ = clean_data_confirmation_policy(
        _action(action_type="drop", strategy="rows"),
        _profile({"a": 0.01, "b": 0.40}), None)
    assert needs is True


# --------------------------------------------------------------------------
# verify() wiring: policy overrides the (now False) static flag
# --------------------------------------------------------------------------
def test_verify_gates_destructive_drop():
    status, _ = verify(_action(action_type="drop", strategy="rows", columns=["x"]),
                       profile=_profile({"x": 0.5}), state={})
    assert status == "needs_review"


def test_verify_allows_safe_standardize():
    status, _ = verify(_action(action_type="standardize", strategy="categories", columns=["x"]),
                       profile=_profile({"x": 0.0}), state={})
    assert status == "allowed"


def test_plugin_is_wired():
    p = get_plugin("clean_data")
    assert p.requires_confirmation is False
    assert p.confirmation_policy is not None
    assert p.guardrail_evaluators


# --------------------------------------------------------------------------
# post-execution impact guardrails
# --------------------------------------------------------------------------
def test_guardrail_large_row_loss_warns():
    f = evaluate_clean_data_guardrails(
        {"metrics": {"original_n_rows": 100, "rows_removed": 30}, "metadata": {}})
    assert any(x["severity"] == "warning" for x in f)


def test_guardrail_small_row_loss_is_info():
    f = evaluate_clean_data_guardrails(
        {"metrics": {"original_n_rows": 100, "rows_removed": 8}, "metadata": {}})
    assert any(x["severity"] == "info" for x in f)
    assert not any(x["severity"] == "warning" for x in f)


def test_guardrail_cast_coercion_warns():
    f = evaluate_clean_data_guardrails(
        {"metrics": {}, "metadata": {"action_info": {"cast": {"x": {"coercion_failures": 5}}}}})
    assert any(x["severity"] == "warning" and "cast" in x["title"].lower() for x in f)


def test_guardrail_clip_is_info():
    f = evaluate_clean_data_guardrails(
        {"metrics": {}, "metadata": {"action_info": {"clip": {"x": {"n_clipped": 3}}}}})
    assert any(x["severity"] == "info" for x in f)


def test_guardrail_clean_step_no_impact_silent():
    f = evaluate_clean_data_guardrails(
        {"metrics": {"original_n_rows": 100, "rows_removed": 0}, "metadata": {}})
    assert f == []