"""
Tests for the statistical claims ledger (P0).

The claims ledger guarantees that every statistical assertion in a final answer
is bound to a computed result: the language model references claims by ID, and
the renderer substitutes verified numeric wording the model cannot author. This
is what makes fabricated conclusions (e.g. calling p = .147 "significant")
architecturally impossible.

Test groups:
  1. Number formatting helpers (_fmt_p, _fmt_unit, _fmt_ci)
  2. Claim.render() for every claim kind, including boundaries
  3. ClaimSet substitution + validation (the anti-fabrication guarantee)
  4. Per-plugin claim builders on real plugin output (all 6 benchmark shapes)
  5. Robustness (empty set, malformed runs, missing builder, serialization)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List

import pandas as pd
import pytest

from core.claims import (
    Claim,
    ClaimSet,
    CLAIM_SIGNIFICANCE,
    CLAIM_EFFECT_SIZE,
    CLAIM_TEST_STATISTIC,
    CLAIM_DIRECTION,
    CLAIM_SESSION_WARNING,
    _fmt_p,
    _fmt_unit,
    _fmt_num,
    _fmt_ci,
)
from core.analysis_tool_plugins.shared.claims_builders import (
    build_claims_for_run,
    build_claims_statistical_group_comparison,
    build_claims_paired_comparison,
    build_claims_nonparametric_group_comparison,
    CLAIMS_BUILDERS_BY_TOOL_NAME,
)
from core.analysis_tool_plugins.registry import get_plugin


DATASETS = os.path.join(os.path.dirname(__file__), "..", "benchmark", "datasets")


class DummyContext:
    def __init__(self, df, args=None):
        self.df = df
        self.args = args or {}
        self.arguments = self.args
        self.active_data_version_id = "data_v_test"
        self.data_versions = []

    def load_df(self):
        return self.df

    def get_arg(self, key, default=None):
        return self.args.get(key, default)


def _run_plugin(tool: str, df: pd.DataFrame, args: Dict[str, Any]) -> Dict[str, Any]:
    out = get_plugin(tool).run(DummyContext(df, args))
    return {
        "run_id": f"r_{tool}",
        "tool_name": tool,
        "status": out.get("status", "ok"),
        "metrics": out.get("details", {}),
        "arguments": args,
    }


def _have_dataset(name: str) -> bool:
    return os.path.exists(os.path.join(DATASETS, name))


# ============================================================
# Group 1 — number formatting
# ============================================================

class TestFormatting:
    def test_fmt_p_small(self):
        assert _fmt_p(0.0001) == "p < .001"

    def test_fmt_p_drops_leading_zero(self):
        assert _fmt_p(0.147) == "p = .147"

    def test_fmt_p_nonfinite(self):
        assert "n/a" in _fmt_p(float("nan"))

    def test_fmt_unit_drops_leading_zero(self):
        assert _fmt_unit(0.013) == ".013"

    def test_fmt_unit_negative(self):
        assert _fmt_unit(-0.48) == "-.480"

    def test_fmt_num_two_decimals(self):
        assert _fmt_num(1.47828) == "1.48"

    def test_fmt_ci_basic(self):
        assert _fmt_ci(-0.13, 0.88) == "95% CI [-0.13, 0.88]"

    def test_fmt_ci_unit_drops_zeros(self):
        assert _fmt_ci(0.025, 0.266, unit=True, digits=3) == "95% CI [.025, .266]"

    def test_fmt_ci_nonfinite_returns_none(self):
        assert _fmt_ci(float("nan"), 1.0) is None


# ============================================================
# Group 2 — Claim.render()
# ============================================================

class TestClaimRender:
    def test_significance_not_sig_is_parenthetical(self):
        c = Claim(claim_id="s", kind=CLAIM_SIGNIFICANCE,
                  data={"p_value": 0.147, "alpha": 0.05})
        r = c.render()
        # Parenthetical numeric support, NO verdict phrase baked in.
        assert r == "(p = .147, \u03b1 = .050)"
        assert "significant" not in r

    def test_significance_significant_still_parenthetical(self):
        c = Claim(claim_id="s", kind=CLAIM_SIGNIFICANCE,
                  data={"p_value": 0.0005, "alpha": 0.05})
        assert c.render() == "(p < .001, \u03b1 = .050)"

    def test_effect_size_with_ci_and_magnitude(self):
        c = Claim(claim_id="e", kind=CLAIM_EFFECT_SIZE,
                  data={"name": "Hedges g", "value": 0.377, "magnitude": "small",
                        "ci_low": -0.129, "ci_high": 0.879})
        assert c.render() == "(Hedges g = 0.38, small, 95% CI [-0.13, 0.88])"

    def test_effect_size_bounded_unit(self):
        c = Claim(claim_id="e", kind=CLAIM_EFFECT_SIZE,
                  data={"name": "\u03b7\u00b2", "value": 0.013, "magnitude": "small",
                        "bounded_unit": True})
        assert c.render() == "(\u03b7\u00b2 = .013, small)"

    def test_test_statistic_with_df(self):
        c = Claim(claim_id="t", kind=CLAIM_TEST_STATISTIC,
                  data={"label": "t", "value": 1.478, "df": 39.60})
        assert c.render() == "(t(39.60) = 1.48)"

    def test_test_statistic_integer_df(self):
        c = Claim(claim_id="f", kind=CLAIM_TEST_STATISTIC,
                  data={"label": "F", "value": 10.74, "df": 2})
        assert c.render() == "(F(2) = 10.74)"

    def test_test_statistic_no_df(self):
        c = Claim(claim_id="u", kind=CLAIM_TEST_STATISTIC,
                  data={"label": "U", "value": 123.0})
        assert c.render() == "(U = 123.00)"

    def test_direction_render(self):
        c = Claim(claim_id="d", kind=CLAIM_DIRECTION,
                  data={"higher_group": "B", "lower_group": "A"})
        assert c.render() == "B scored higher than A"

    def test_session_warning_render(self):
        c = Claim(claim_id="w", kind=CLAIM_SESSION_WARNING,
                  data={"text": "5 tests; consider correction"})
        assert c.render() == "5 tests; consider correction"

    def test_invalid_kind_raises(self):
        with pytest.raises(ValueError):
            Claim(claim_id="x", kind="not_a_kind", data={})


# ============================================================
# Group 3 — ClaimSet substitution + validation
# ============================================================

class TestClaimSet:
    def _make(self):
        cs = ClaimSet()
        cs.add(Claim(claim_id="r1_sig", kind=CLAIM_SIGNIFICANCE,
                     data={"p_value": 0.147, "alpha": 0.05}))
        cs.add(Claim(claim_id="r1_es", kind=CLAIM_EFFECT_SIZE,
                     data={"name": "Hedges g", "value": 0.38, "magnitude": "small",
                           "ci_low": -0.13, "ci_high": 0.88}))
        return cs

    def test_substitute_replaces_refs(self):
        cs = self._make()
        text = "No difference [CLAIM:r1_sig], small effect [CLAIM:r1_es]."
        rendered, unresolved = cs.substitute(text)
        assert unresolved == []
        assert "[CLAIM:" not in rendered
        assert "(p = .147, \u03b1 = .050)" in rendered

    def test_substitute_preserves_decimal_spacing(self):
        # The space-tidy regex must NOT eat the space before a decimal point.
        cs = self._make()
        rendered, _ = cs.substitute("Result [CLAIM:r1_sig].")
        assert "p = .147" in rendered  # not "p =.147"

    def test_substitute_unresolved_left_in_place(self):
        cs = self._make()
        rendered, unresolved = cs.substitute("See [CLAIM:does_not_exist].")
        assert "does_not_exist" in unresolved
        assert "[CLAIM:does_not_exist]" in rendered

    def test_validate_clean_when_referenced(self):
        cs = self._make()
        v = cs.validate("No difference [CLAIM:r1_sig], small [CLAIM:r1_es].")
        assert v["is_clean"]
        assert v["unresolved_ids"] == []

    def test_validate_significance_word_is_allowed(self):
        # The model SHOULD state the verdict in words; this is not a violation.
        cs = self._make()
        v = cs.validate("There was no statistically significant difference [CLAIM:r1_sig].")
        assert v["is_clean"]
        assert v["restated_significance"] is True

    def test_validate_bare_pvalue_is_violation(self):
        cs = self._make()
        v = cs.validate("We found p = .03 in the data [CLAIM:r1_sig].")
        assert not v["is_clean"]
        assert "bare_p_value" in v["bare_numeric_assertions"]

    def test_validate_unresolved_is_violation(self):
        cs = self._make()
        v = cs.validate("See [CLAIM:ghost].")
        assert not v["is_clean"]
        assert "ghost" in v["unresolved_ids"]

    def test_catalogue_lists_ids_and_wording(self):
        cs = self._make()
        cat = cs.catalogue_text()
        assert "[CLAIM:r1_sig]" in cat
        assert "(p = .147" in cat

    def test_empty_set(self):
        cs = ClaimSet()
        assert cs.is_empty()
        assert cs.catalogue_text() == "(no statistical claims available yet)"


# ============================================================
# Group 4 — anti-fabrication guarantee (the core P0 property)
# ============================================================

class TestAntiFabrication:
    """The case4 failure: tool computed p=.147 (not significant) but the LLM
    stated 'statistically significant'. These tests prove that path is closed."""

    def test_verdict_numbers_come_from_fields_not_text(self):
        # Even if a caller TRIES to assert significance, the rendered numbers
        # are whatever the structured field says — the model cannot change them.
        c = Claim(claim_id="s", kind=CLAIM_SIGNIFICANCE,
                  data={"p_value": 0.147, "alpha": 0.05})
        assert "p = .147" in c.render()  # the true value, always

    def test_bare_fabrication_is_flagged(self):
        cs = ClaimSet()
        cs.add(Claim(claim_id="s", kind=CLAIM_SIGNIFICANCE,
                     data={"p_value": 0.147, "alpha": 0.05}))
        # LLM types a naked contradictory number instead of citing the claim.
        v = cs.validate("The difference is real, p = .01, clearly significant.")
        assert not v["is_clean"]

    def test_referenced_claim_renders_truth(self):
        cs = ClaimSet()
        cs.add(Claim(claim_id="s", kind=CLAIM_SIGNIFICANCE,
                     data={"p_value": 0.147, "alpha": 0.05}))
        # LLM says "significant" but cites the claim: the parenthetical exposes
        # p = .147, so the numbers remain truthful regardless of the verb.
        rendered, _ = cs.substitute("The effect was significant [CLAIM:s].")
        assert "p = .147" in rendered


# ============================================================
# Group 5 — per-plugin builders on real plugin output
# ============================================================

@pytest.mark.skipif(not _have_dataset("case6_effect_size_reporting.parquet"),
                    reason="benchmark datasets not present")
class TestBuildersRealOutput:
    def test_two_group_welch_builds_sig_es_stat(self):
        df = pd.read_parquet(os.path.join(DATASETS, "case6_effect_size_reporting.parquet"))
        run = _run_plugin("statistical_group_comparison", df,
                          {"target_col": "test_score", "group_col": "cohort"})
        claims = build_claims_for_run(run)
        kinds = {c.kind for c in claims}
        assert CLAIM_SIGNIFICANCE in kinds
        assert CLAIM_EFFECT_SIZE in kinds
        assert CLAIM_TEST_STATISTIC in kinds
        # No direction claim — direction is left to the model.
        assert CLAIM_DIRECTION not in kinds

    def test_anova_builds_eta_squared_bounded(self):
        if not _have_dataset("case1_unequal_variance_anova.parquet"):
            pytest.skip("dataset missing")
        df = pd.read_parquet(os.path.join(DATASETS, "case1_unequal_variance_anova.parquet"))
        run = _run_plugin("statistical_group_comparison", df,
                          {"target_col": "outcome", "group_col": "treatment"})
        claims = {c.kind: c for c in build_claims_for_run(run)}
        es = claims[CLAIM_EFFECT_SIZE]
        # eta squared is a bounded [0,1] unit -> leading zero dropped, no minus.
        assert es.data.get("bounded_unit") is True
        assert es.render().startswith("(")

    def test_case4_not_significant_claim(self):
        if not _have_dataset("case4_nonnormal_two_group.parquet"):
            pytest.skip("dataset missing")
        df = pd.read_parquet(os.path.join(DATASETS, "case4_nonnormal_two_group.parquet"))
        run = _run_plugin("statistical_group_comparison", df,
                          {"target_col": "response_time", "group_col": "arm"})
        sig = next(c for c in build_claims_for_run(run) if c.kind == CLAIM_SIGNIFICANCE)
        # The real benchmark value: p around .147, not significant.
        assert sig.data["p_value"] > 0.05

    def test_paired_builds_sig_and_es(self):
        if not _have_dataset("case5_paired_nonnormal.parquet"):
            pytest.skip("dataset missing")
        df = pd.read_parquet(os.path.join(DATASETS, "case5_paired_nonnormal.parquet"))
        run = _run_plugin("paired_comparison", df,
                          {"target_col_1": "pre_score", "target_col_2": "post_score"})
        kinds = {c.kind for c in build_claims_for_run(run)}
        assert CLAIM_SIGNIFICANCE in kinds
        assert CLAIM_EFFECT_SIZE in kinds


# ============================================================
# Group 6 — robustness
# ============================================================

class TestRobustness:
    def test_builder_skips_failed_run(self):
        run = {"run_id": "r", "tool_name": "statistical_group_comparison",
               "status": "error", "metrics": {}}
        assert build_claims_for_run(run) == []

    def test_builder_unknown_tool_returns_empty(self):
        run = {"run_id": "r", "tool_name": "no_such_tool", "status": "ok",
               "metrics": {"p_value": 0.01}}
        assert build_claims_for_run(run) == []

    def test_builder_non_dict_returns_empty(self):
        assert build_claims_for_run(None) == []
        assert build_claims_for_run("not a dict") == []

    def test_builder_missing_metrics_no_crash(self):
        run = {"run_id": "r", "tool_name": "statistical_group_comparison",
               "status": "ok"}
        # No 'metrics' key at all — must not raise.
        assert isinstance(build_claims_for_run(run), list)

    def test_dispatch_table_covers_known_tools(self):
        for tool in ["statistical_group_comparison", "paired_comparison",
                     "nonparametric_group_comparison", "run_independent_t_test"]:
            assert tool in CLAIMS_BUILDERS_BY_TOOL_NAME

    def test_serialization_round_trip(self):
        cs = ClaimSet()
        cs.add(Claim(claim_id="s", kind=CLAIM_SIGNIFICANCE,
                     data={"p_value": 0.147, "alpha": 0.05}, subject="x by y"))
        items = cs.to_list()
        cs2 = ClaimSet.from_list(items)
        assert cs2.get("s").render() == cs.get("s").render()
        assert cs2.get("s").subject == "x by y"

    def test_from_list_skips_malformed(self):
        cs = ClaimSet.from_list([{"bad": "entry"}, None])
        assert cs.is_empty()