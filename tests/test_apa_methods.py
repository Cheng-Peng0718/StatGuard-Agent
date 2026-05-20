"""
Tests for the APA Methods Section export system.

Test groups:
  1. apa_formatting helpers (fmt_p, fmt_bounded_unit, fmt_ci, ...)
  2. Per-tool APA writers (apa_writers.py) for all 4 routing paths each
  3. Plugin registration: each priority plugin has apa_methods_writer set
  4. export_apa_methods plugin: end-to-end assembly
  5. Markdown -> plaintext conversion
  6. Robustness (empty session, malformed runs, missing writer)
"""

from __future__ import annotations

import os
import shutil
import tempfile
from typing import Any, Dict, List

import pytest

from core.analysis_tool_plugins.shared.apa_formatting import (
    fmt_p,
    fmt_bounded_unit,
    fmt_general,
    fmt_signed,
    fmt_int,
    fmt_ci,
    md_italic,
)
from core.analysis_tool_plugins.shared.apa_writers import (
    write_apa_statistical_group_comparison,
    write_apa_independent_t_test,
    write_apa_paired_comparison,
    write_apa_nonparametric_group_comparison,
    APA_WRITERS_BY_TOOL_NAME,
)
from core.analysis_tool_plugins.registry import PLUGIN_REGISTRY


# ============================================================
# 1. APA formatting helpers
# ============================================================

class TestFmtP:
    def test_p_under_001(self):
        assert fmt_p(0.0001) == "p < .001"
        assert fmt_p(0.0009) == "p < .001"

    def test_p_drops_leading_zero(self):
        assert fmt_p(0.027) == "p = .027"
        assert fmt_p(0.5) == "p = .500"

    def test_p_three_decimals(self):
        assert fmt_p(0.05) == "p = .050"
        assert fmt_p(0.999) == "p = .999"

    def test_p_none_or_invalid(self):
        assert fmt_p(None) == "p = n/a"
        assert fmt_p(float("nan")) == "p = n/a"
        assert fmt_p("not a number") == "p = n/a"


class TestFmtBoundedUnit:
    def test_drops_leading_zero(self):
        assert fmt_bounded_unit(0.142) == ".142"
        assert fmt_bounded_unit(0.005) == ".005"

    def test_keeps_value_above_one_unchanged(self):
        # Although physically improbable for eta², the formatter should not
        # corrupt the value
        assert fmt_bounded_unit(1.0) == "1.000"

    def test_negative_drops_leading_zero(self):
        assert fmt_bounded_unit(-0.142) == "-.142"

    def test_none_returns_na(self):
        assert fmt_bounded_unit(None) == "n/a"
        assert fmt_bounded_unit(float("inf")) == "n/a"


class TestFmtGeneral:
    def test_keeps_leading_zero(self):
        """Stats that can exceed 1 (t, F, M, SD) keep their leading zero."""
        assert fmt_general(0.5) == "0.50"
        assert fmt_general(0.05) == "0.05"

    def test_two_decimals_default(self):
        assert fmt_general(5.0) == "5.00"
        # Note: 5.555 has a banker-rounding edge case in float; use 5.556 to be safe
        assert fmt_general(5.556) == "5.56"

    def test_custom_digits(self):
        assert fmt_general(5.5555, digits=3) == "5.556"

    def test_none(self):
        assert fmt_general(None) == "n/a"


class TestFmtSigned:
    def test_positive_has_plus(self):
        assert fmt_signed(0.5) == "+0.50"

    def test_negative_keeps_minus(self):
        assert fmt_signed(-0.5) == "-0.50"

    def test_none(self):
        assert fmt_signed(None) == "n/a"


class TestFmtInt:
    def test_basic(self):
        assert fmt_int(5) == "5"
        assert fmt_int(5.7) == "6"

    def test_none(self):
        assert fmt_int(None) == "n/a"


class TestFmtCI:
    def test_basic_ci(self):
        assert fmt_ci(0.2, 0.8) == "95% CI [0.20, 0.80]"

    def test_bounded_unit_drops_leading_zero(self):
        assert fmt_ci(0.025, 0.366, bounded_unit=True, digits=3) == "95% CI [.025, .366]"

    def test_missing_endpoint_returns_none(self):
        assert fmt_ci(None, 0.8) is None
        assert fmt_ci(0.2, None) is None
        assert fmt_ci(None, None) is None

    def test_negative_lower(self):
        assert fmt_ci(-0.5, 0.8) == "95% CI [-0.50, 0.80]"


class TestMdItalic:
    def test_wraps_in_stars(self):
        assert md_italic("t") == "*t*"
        assert md_italic("p value") == "*p value*"


# ============================================================
# 2. APA writers per tool family
# ============================================================

def _make_run(
    *,
    tool_name: str,
    metrics: Dict[str, Any],
    arguments: Dict[str, Any] = None,
    tables: Dict[str, Any] = None,
    status: str = "ok",
    title: str = "Test run",
) -> Dict[str, Any]:
    """Build a minimal analysis_run dict for testing."""
    return {
        "run_id": "test_run_1",
        "action_id": "test_act_1",
        "tool_name": tool_name,
        "title": title,
        "status": status,
        "success": status == "ok",
        "is_inferential": True,
        "data_version_id": "raw_v1",
        "arguments": arguments or {},
        "metrics": metrics,
        "tables": tables or {},
        "guardrails": [],
        "created_at": "2026-05-19T12:00:00Z",
    }


class TestWriterStatisticalGroupComparison:
    """The most complex writer: 2 groups (Welch t), classic ANOVA, or Welch ANOVA."""

    def test_welch_t_test_path(self):
        run = _make_run(
            tool_name="statistical_group_comparison",
            arguments={"target_col": "revenue", "group_col": "region"},
            metrics={
                "method": "Welch independent two-sample t-test",
                "t_statistic": 2.5,
                "degrees_of_freedom": 38.2,
                "p_value": 0.017,
                "alpha": 0.05,
                "effect_size": 0.45,
                "effect_size_ci_low": 0.08,
                "effect_size_ci_high": 0.82,
                "cohens_d": 0.46,
                "mean_difference_group1_minus_group2": 5.3,
                "mean_difference_ci_low": 1.0,
                "mean_difference_ci_high": 9.6,
            },
        )
        text = write_apa_statistical_group_comparison(run)
        assert text is not None
        assert "*t*" in text
        assert "(38.20)" in text
        assert "2.50" in text
        assert "p = .017" in text
        assert "Hedges' *g*" in text
        assert "+0.45" in text
        assert "95% CI [0.08, 0.82]" in text

    def test_classic_anova_with_tukey(self):
        run = _make_run(
            tool_name="statistical_group_comparison",
            arguments={"target_col": "score", "group_col": "group"},
            metrics={
                "method": "One-way ANOVA",
                "F_statistic": 7.16,
                "degrees_of_freedom_between": 2,
                "degrees_of_freedom_within": 87,
                "p_value": 0.0013,
                "alpha": 0.05,
                "significant_at_alpha": True,
                "eta_squared": 0.141,
                "eta_squared_ci_low": 0.025,
                "eta_squared_ci_high": 0.266,
                "omega_squared": 0.120,
            },
            tables={
                "post_hoc_pairwise": [
                    {"group1": "A", "group2": "B", "significant_at_alpha": True,
                     "adjustment_method": "Tukey's HSD"},
                    {"group1": "A", "group2": "C", "significant_at_alpha": True,
                     "adjustment_method": "Tukey's HSD"},
                    {"group1": "B", "group2": "C", "significant_at_alpha": False,
                     "adjustment_method": "Tukey's HSD"},
                ],
            },
        )
        text = write_apa_statistical_group_comparison(run)
        assert text is not None
        assert "one-way ANOVA" in text
        assert "*F*(2, 87.00)" in text
        assert "7.16" in text
        assert "p = .001" in text
        assert "η² = .141" in text
        assert "95% CI [.025, .266]" in text
        assert "ω² = .120" in text
        assert "Tukey" in text
        assert "2 of 3" in text

    def test_welch_anova_path_with_games_howell(self):
        run = _make_run(
            tool_name="statistical_group_comparison",
            arguments={"target_col": "score", "group_col": "group"},
            metrics={
                "method": "Welch one-way ANOVA",
                "F_statistic": 4.20,
                "degrees_of_freedom_between": 2,
                "degrees_of_freedom_within": 55.3,
                "p_value": 0.020,
                "alpha": 0.05,
                "significant_at_alpha": True,
                "eta_squared": 0.10,
                "eta_squared_ci_low": 0.01,
                "eta_squared_ci_high": 0.22,
            },
            tables={
                "post_hoc_pairwise": [
                    {"group1": "A", "group2": "B", "significant_at_alpha": True,
                     "adjustment_method": "Games-Howell"},
                    {"group1": "A", "group2": "C", "significant_at_alpha": False,
                     "adjustment_method": "Games-Howell"},
                ],
            },
        )
        text = write_apa_statistical_group_comparison(run)
        assert text is not None
        assert "Welch's one-way ANOVA" in text
        assert "Levene" in text  # explains why Welch was chosen
        assert "Games-Howell" in text

    def test_anova_not_significant_does_not_mention_posthoc(self):
        run = _make_run(
            tool_name="statistical_group_comparison",
            arguments={"target_col": "score", "group_col": "group"},
            metrics={
                "method": "One-way ANOVA",
                "F_statistic": 1.2,
                "degrees_of_freedom_between": 2,
                "degrees_of_freedom_within": 87,
                "p_value": 0.30,
                "significant_at_alpha": False,
                "eta_squared": 0.027,
            },
            tables={},
        )
        text = write_apa_statistical_group_comparison(run)
        assert text is not None
        assert "post-hoc" not in text.lower()
        assert "Tukey" not in text


class TestWriterIndependentTTest:
    def test_basic_t_test(self):
        run = _make_run(
            tool_name="run_independent_t_test",
            arguments={
                "target_col": "score",
                "group1_val": "treatment",
                "group2_val": "control",
            },
            metrics={
                "t_statistic": -2.1,
                "degrees_of_freedom": 48.5,
                "p_value": 0.041,
                "effect_size": -0.5,
                "effect_size_ci_low": -0.97,
                "effect_size_ci_high": -0.02,
                "cohens_d": -0.51,
                "mean_difference_group1_minus_group2": -3.2,
                "mean_difference_ci_low": -6.3,
                "mean_difference_ci_high": -0.1,
            },
        )
        text = write_apa_independent_t_test(run)
        assert text is not None
        assert "treatment" in text
        assert "control" in text
        assert "*t*(48.50)" in text
        assert "-2.10" in text
        assert "p = .041" in text
        assert "-0.50" in text


class TestWriterPairedComparison:
    def test_paired_t_recommended(self):
        run = _make_run(
            tool_name="paired_comparison",
            arguments={"target_col_1": "pre", "target_col_2": "post"},
            metrics={
                "recommended_test": "paired_t_test",
                "n_complete_pairs": 30,
                "t_statistic": -3.5,
                "degrees_of_freedom": 29,
                "paired_t_p_value": 0.0015,
                "p_value": 0.0015,
                "cohens_d_z": -0.64,
                "cohens_d_z_ci_low": -1.04,
                "cohens_d_z_ci_high": -0.23,
                "mean_difference": -4.2,
                "mean_difference_ci_low": -6.6,
                "mean_difference_ci_high": -1.8,
            },
        )
        text = write_apa_paired_comparison(run)
        assert text is not None
        assert "paired-samples *t* test" in text
        assert "30 complete pairs" in text
        assert "*t*(29)" in text
        assert "p = .001" in text or "p = .002" in text
        assert "Cohen's *d*_z" in text

    def test_wilcoxon_recommended(self):
        run = _make_run(
            tool_name="paired_comparison",
            arguments={"target_col_1": "pre", "target_col_2": "post"},
            metrics={
                "recommended_test": "wilcoxon_signed_rank",
                "n_complete_pairs": 25,
                "W_statistic": 45.0,
                "wilcoxon_p_value": 0.018,
                "p_value": 0.018,
                "rank_biserial_correlation": -0.55,
                "hodges_lehmann_pseudomedian": -3.5,
                "hodges_lehmann_ci_low": -6.0,
                "hodges_lehmann_ci_high": -1.0,
            },
        )
        text = write_apa_paired_comparison(run)
        assert text is not None
        assert "Wilcoxon signed-rank" in text
        assert "Shapiro-Wilk" in text  # explains why Wilcoxon
        assert "*W*" in text
        assert "45.00" in text
        assert "Hodges-Lehmann pseudomedian" in text


class TestWriterNonparametricGroupComparison:
    def test_mann_whitney_path(self):
        run = _make_run(
            tool_name="nonparametric_group_comparison",
            arguments={"target_col": "score", "group_col": "group"},
            metrics={
                "method": "Mann-Whitney U test (two-sided)",
                "U_statistic": 120.0,
                "p_value": 0.025,
                "alpha": 0.05,
                "effect_size_name": "rank-biserial correlation",
                "effect_size": -0.4,
                "hodges_lehmann_location_shift": -3.0,
                "hodges_lehmann_ci_low": -5.5,
                "hodges_lehmann_ci_high": -0.5,
            },
        )
        text = write_apa_nonparametric_group_comparison(run)
        assert text is not None
        assert "Mann-Whitney *U*" in text
        assert "*U* = 120.00" in text
        assert "p = .025" in text
        assert "rank-biserial" in text

    def test_kruskal_wallis_with_dunn(self):
        run = _make_run(
            tool_name="nonparametric_group_comparison",
            arguments={"target_col": "score", "group_col": "group"},
            metrics={
                "method": "Kruskal-Wallis H test",
                "H_statistic": 12.89,
                "degrees_of_freedom_between": 2,
                "p_value": 0.0016,
                "alpha": 0.05,
                "significant_at_alpha": True,
                "effect_size_name": "epsilon squared",
                "effect_size": 0.145,
                "epsilon_squared": 0.145,
            },
            tables={
                "post_hoc_pairwise": [
                    {"group1": "A", "group2": "B", "significant_at_alpha": True,
                     "adjustment_method": "Dunn's test (Benjamini-Hochberg FDR)"},
                    {"group1": "A", "group2": "C", "significant_at_alpha": True,
                     "adjustment_method": "Dunn's test (Benjamini-Hochberg FDR)"},
                    {"group1": "B", "group2": "C", "significant_at_alpha": False,
                     "adjustment_method": "Dunn's test (Benjamini-Hochberg FDR)"},
                ],
            },
        )
        text = write_apa_nonparametric_group_comparison(run)
        assert text is not None
        assert "Kruskal-Wallis *H*" in text
        assert "*H*(2)" in text
        assert "12.89" in text
        assert "ε² = .145" in text
        assert "Dunn" in text
        assert "2 of 3" in text


class TestWriterNoneForUnrecognizedMethod:
    """Each writer must return None (not crash) for unknown methods."""
    def test_group_comparison_unknown_method(self):
        run = _make_run(
            tool_name="statistical_group_comparison",
            metrics={"method": "Something Unknown"},
        )
        assert write_apa_statistical_group_comparison(run) is None

    def test_nonparametric_unknown_method(self):
        run = _make_run(
            tool_name="nonparametric_group_comparison",
            metrics={"method": "Something Else"},
        )
        assert write_apa_nonparametric_group_comparison(run) is None


# ============================================================
# 3. Plugin registration
# ============================================================

class TestPluginRegistration:
    """The four priority plugins must each register an apa_methods_writer."""

    PRIORITY_PLUGINS = [
        "statistical_group_comparison",
        "run_independent_t_test",
        "paired_comparison",
        "nonparametric_group_comparison",
    ]

    def test_all_priority_plugins_have_writer(self):
        for name in self.PRIORITY_PLUGINS:
            plugin = PLUGIN_REGISTRY.get(name)
            assert plugin is not None, f"Plugin '{name}' is not registered"
            assert plugin.apa_methods_writer is not None, (
                f"Plugin '{name}' has no apa_methods_writer"
            )

    def test_writer_dispatch_table_has_all_priority_plugins(self):
        for name in self.PRIORITY_PLUGINS:
            assert name in APA_WRITERS_BY_TOOL_NAME, (
                f"'{name}' missing from APA_WRITERS_BY_TOOL_NAME"
            )

    def test_export_apa_methods_plugin_exists(self):
        plugin = PLUGIN_REGISTRY.get("export_apa_methods")
        assert plugin is not None
        assert plugin.is_inferential is False


# ============================================================
# 4. End-to-end: export_apa_methods plugin
# ============================================================

@pytest.fixture
def temp_workspace():
    workspace = tempfile.mkdtemp(prefix="apa_methods_test_")
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


def _make_ctx(workspace, analysis_runs, arguments=None):
    class Ctx:
        pass
    c = Ctx()
    c.workspace_dir = workspace
    c.analysis_runs = analysis_runs
    c.arguments = arguments or {}
    return c


class TestExportAPAMethodsPlugin:
    def test_with_one_anova_run(self, temp_workspace):
        from core.analysis_tool_plugins.registry import get_plugin

        run = _make_run(
            tool_name="statistical_group_comparison",
            arguments={"target_col": "score", "group_col": "group"},
            metrics={
                "method": "One-way ANOVA",
                "F_statistic": 7.16,
                "degrees_of_freedom_between": 2,
                "degrees_of_freedom_within": 87,
                "p_value": 0.0013,
                "eta_squared": 0.141,
                "eta_squared_ci_low": 0.025,
                "eta_squared_ci_high": 0.266,
            },
        )

        ctx = _make_ctx(temp_workspace, [run])
        plugin = get_plugin("export_apa_methods")
        result = plugin.run(ctx)

        assert result["status"] == "ok"
        assert result["details"]["n_paragraphs_written"] == 1
        assert result["details"]["n_runs_skipped"] == 0
        assert len(result["artifacts"]) == 2  # .md + .txt

        md_path = result["details"]["markdown_path"]
        txt_path = result["details"]["plaintext_path"]
        assert os.path.exists(md_path)
        assert os.path.exists(txt_path)

    def test_methods_section_structure(self, temp_workspace):
        from core.analysis_tool_plugins.registry import get_plugin

        run = _make_run(
            tool_name="statistical_group_comparison",
            arguments={"target_col": "x", "group_col": "g"},
            metrics={
                "method": "One-way ANOVA",
                "F_statistic": 5.0,
                "degrees_of_freedom_between": 2,
                "degrees_of_freedom_within": 42,
                "p_value": 0.01,
                "eta_squared": 0.19,
                "eta_squared_ci_low": 0.02,
                "eta_squared_ci_high": 0.37,
            },
        )

        ctx = _make_ctx(temp_workspace, [run])
        plugin = get_plugin("export_apa_methods")
        result = plugin.run(ctx)

        md = result["details"]["markdown_inline"]
        # Methods section structure
        assert "# Methods" in md
        assert "## Statistical Analyses" in md
        assert "## Software" in md
        # Software paragraph mentions scipy and statsmodels
        assert "SciPy" in md
        assert "statsmodels" in md
        assert "deterministic" in md.lower()

    def test_skips_runs_without_writer(self, temp_workspace):
        """A run from a plugin without apa_methods_writer should be skipped."""
        from core.analysis_tool_plugins.registry import get_plugin

        # Use a plugin we know has no writer (e.g. inspect_dataset)
        run = _make_run(
            tool_name="inspect_dataset",
            arguments={},
            metrics={"nobs": 100},
        )

        ctx = _make_ctx(temp_workspace, [run])
        plugin = get_plugin("export_apa_methods")
        result = plugin.run(ctx)

        assert result["status"] == "ok"
        assert result["details"]["n_paragraphs_written"] == 0
        assert result["details"]["n_runs_skipped"] == 1
        skipped = result["details"]["skipped"][0]
        assert "writer" in skipped["reason"].lower() or "no" in skipped["reason"].lower()

    def test_skips_failed_runs(self, temp_workspace):
        from core.analysis_tool_plugins.registry import get_plugin

        run = _make_run(
            tool_name="statistical_group_comparison",
            metrics={},
            status="failed",
        )

        ctx = _make_ctx(temp_workspace, [run])
        plugin = get_plugin("export_apa_methods")
        result = plugin.run(ctx)

        assert result["status"] == "ok"
        assert result["details"]["n_paragraphs_written"] == 0
        assert result["details"]["n_runs_skipped"] == 1

    def test_empty_session(self, temp_workspace):
        from core.analysis_tool_plugins.registry import get_plugin

        ctx = _make_ctx(temp_workspace, [])
        plugin = get_plugin("export_apa_methods")
        result = plugin.run(ctx)

        # Empty session should not crash; should produce Methods skeleton with
        # an explicit "no analyses" note
        assert result["status"] == "ok"
        assert result["details"]["n_paragraphs_written"] == 0
        md = result["details"]["markdown_inline"]
        assert "# Methods" in md
        assert "## Software" in md
        # The Statistical Analyses section should note nothing was run
        assert "no inferential" in md.lower() or "no analyses" in md.lower() or "_no" in md.lower()

    def test_write_to_disk_can_be_disabled(self, temp_workspace):
        from core.analysis_tool_plugins.registry import get_plugin

        run = _make_run(
            tool_name="statistical_group_comparison",
            metrics={
                "method": "One-way ANOVA",
                "F_statistic": 5.0,
                "degrees_of_freedom_between": 2,
                "degrees_of_freedom_within": 42,
                "p_value": 0.01,
                "eta_squared": 0.19,
            },
        )

        ctx = _make_ctx(temp_workspace, [run], arguments={"write_to_disk": False})
        plugin = get_plugin("export_apa_methods")
        result = plugin.run(ctx)

        assert result["status"] == "ok"
        assert result["details"]["wrote_to_disk"] is False
        assert result["details"]["markdown_path"] is None
        assert result["details"]["plaintext_path"] is None
        assert result["artifacts"] == []
        # Inline content still produced
        assert "# Methods" in result["details"]["markdown_inline"]

    def test_multiple_runs_concatenate_in_order(self, temp_workspace):
        from core.analysis_tool_plugins.registry import get_plugin

        run1 = _make_run(
            tool_name="statistical_group_comparison",
            arguments={"target_col": "rev", "group_col": "region"},
            metrics={
                "method": "One-way ANOVA",
                "F_statistic": 7.16,
                "degrees_of_freedom_between": 2,
                "degrees_of_freedom_within": 87,
                "p_value": 0.001,
                "eta_squared": 0.141,
            },
        )
        run2 = _make_run(
            tool_name="paired_comparison",
            arguments={"target_col_1": "pre", "target_col_2": "post"},
            metrics={
                "recommended_test": "paired_t_test",
                "n_complete_pairs": 30,
                "t_statistic": -3.5,
                "degrees_of_freedom": 29,
                "paired_t_p_value": 0.0015,
                "p_value": 0.0015,
                "cohens_d_z": -0.64,
            },
        )

        ctx = _make_ctx(temp_workspace, [run1, run2])
        plugin = get_plugin("export_apa_methods")
        result = plugin.run(ctx)

        md = result["details"]["markdown_inline"]
        # Both paragraphs should appear, and run1 should come before run2
        anova_pos = md.find("one-way ANOVA")
        paired_pos = md.find("paired-samples *t* test")
        assert anova_pos != -1
        assert paired_pos != -1
        assert anova_pos < paired_pos


# ============================================================
# 5. Markdown -> plaintext conversion
# ============================================================

class TestPlaintextConversion:
    def test_italics_stripped(self, temp_workspace=None):
        # Inline test that doesn't need the full plugin
        from core.analysis_tool_plugins.plugins.export_apa_methods import (
            _strip_markdown_italics,
        )
        assert _strip_markdown_italics("the *t*-statistic was *F*") == "the t-statistic was F"
        assert _strip_markdown_italics("no italics here") == "no italics here"

    def test_plaintext_has_no_markdown_italics(self):
        """After conversion, no `*foo*` patterns should remain."""
        from core.analysis_tool_plugins.plugins.export_apa_methods import (
            _strip_markdown_italics,
        )
        import re
        result = _strip_markdown_italics(
            "Test *t*(29) = -3.50, *p* = .002, Cohen's *d*_z = -0.64."
        )
        # No remaining `*xxx*` patterns
        assert re.search(r"\*[^*]+\*", result) is None


# ============================================================
# 6. Robustness
# ============================================================

class TestRobustness:
    def test_writer_handles_missing_metrics_gracefully(self):
        """A run with mostly empty metrics should not crash the writer."""
        run = _make_run(
            tool_name="statistical_group_comparison",
            metrics={"method": "One-way ANOVA"},  # F_statistic etc. missing
        )
        text = write_apa_statistical_group_comparison(run)
        # Should produce SOMETHING (the opener) rather than crash
        # The output may have "n/a" placeholders but should be a string
        assert text is None or isinstance(text, str)

    def test_writer_handles_none_metrics(self):
        run = {
            "tool_name": "statistical_group_comparison",
            "metrics": {},
            "arguments": {},
            "tables": {},
            "status": "ok",
            "is_inferential": True,
        }
        # Should not raise
        text = write_apa_statistical_group_comparison(run)
        assert text is None or isinstance(text, str)

    def test_export_plugin_handles_writer_exceptions(self, temp_workspace):
        """If a writer raises, the run should be skipped (not crash export)."""
        from core.analysis_tool_plugins.registry import get_plugin

        # A run with the expected tool but with malformed data that might
        # trigger an exception. The export must still complete.
        run = _make_run(
            tool_name="statistical_group_comparison",
            arguments={"target_col": "x"},
            metrics={
                "method": "One-way ANOVA",
                # Deliberately weird: missing required fields, but with a wrong type
                "F_statistic": "not a number",
            },
        )

        ctx = _make_ctx(temp_workspace, [run])
        plugin = get_plugin("export_apa_methods")
        result = plugin.run(ctx)

        # Should complete (status ok) even if this run produces a malformed
        # paragraph or gets skipped
        assert result["status"] == "ok"