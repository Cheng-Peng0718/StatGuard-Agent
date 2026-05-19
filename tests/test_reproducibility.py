"""
Tests for the reproducibility manifest system.

Tests are grouped by concern:
  1. Pure builder (no plugin, no disk) -- core/reproducibility.py
  2. File hashing
  3. Determinism (the cross-cutting promise)
  4. Plugin integration -- core/analysis_tool_plugins/plugins/export_reproducibility_manifest.py
  5. Robustness to malformed / empty / missing inputs
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from typing import Any, Dict, List

import pandas as pd
import pytest

from core.reproducibility import (
    MANIFEST_VERSION,
    build_reproducibility_manifest,
    write_manifest_to_file,
    _sha256_of_file,
    _extract_key_results,
    _summarize_guardrails,
    _build_data_version_entry,
    _build_analysis_entry,
    _capture_environment,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def temp_workspace():
    """A throwaway workspace dir with a real parquet file inside."""
    workspace = tempfile.mkdtemp(prefix="rmtest_")
    yield workspace
    shutil.rmtree(workspace, ignore_errors=True)


@pytest.fixture
def real_data_version(temp_workspace):
    """Write a real parquet so SHA-256 can be computed."""
    versions_dir = os.path.join(temp_workspace, "data_versions")
    os.makedirs(versions_dir, exist_ok=True)

    df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": ["a", "b", "c", "d", "e"]})
    path = os.path.join(versions_dir, "raw_v1.parquet")
    df.to_parquet(path, index=False)

    return {
        "version_id": "raw_v1",
        "parent_version_id": None,
        "path": path,
        "n_rows": 5,
        "n_cols": 2,
        "created_by": "upload",
        "created_at": "2026-05-19T12:00:00Z",
        "operation": "initial_load",
        "description": "test dataset",
        "metadata": {},
    }


@pytest.fixture
def inferential_run():
    """Representative analysis_run for an inferential test (ANOVA)."""
    return {
        "run_id": "run_anova_1",
        "action_id": "act_1",
        "tool_name": "statistical_group_comparison",
        "title": "Group comparison: x by g",
        "status": "ok",
        "success": True,
        "is_inferential": True,
        "evidence_categories": ["group_comparison", "statistical_inference"],
        "data_version_id": "raw_v1",
        "arguments": {"target_col": "x", "group_col": "g", "alpha": 0.05},
        "metrics": {
            "method": "One-way ANOVA",
            "F_statistic": 4.5,
            "p_value": 0.012,
            "alpha": 0.05,
            "significant_at_alpha": True,
            "effect_size_name": "eta squared",
            "effect_size": 0.18,
            "nobs": 60,
            "valid_group_count": 3,
            # A non-whitelisted field that should NOT appear in key_results
            "some_internal_field": "should not leak",
            # A None field that should be skipped
            "null_metric": None,
        },
        "tables": {},
        "guardrails": [
            {
                "finding_id": "gr_1",
                "category": "interpretation",
                "severity": "info",
                "title": "Statistically significant group difference",
            },
            {
                "finding_id": "gr_2",
                "category": "post_hoc",
                "severity": "warning",
                "title": "ANOVA significant without post-hoc",
            },
        ],
        "created_at": "2026-05-19T12:01:00Z",
    }


@pytest.fixture
def descriptive_run():
    """Representative analysis_run for a non-inferential tool."""
    return {
        "run_id": "run_summary_1",
        "action_id": "act_2",
        "tool_name": "groupby_summary",
        "title": "Summary",
        "status": "ok",
        "success": True,
        "is_inferential": False,
        "evidence_categories": ["descriptive_summary"],
        "data_version_id": "raw_v1",
        "arguments": {"target_col": "x", "group_col": "g"},
        "metrics": {"nobs": 60},
        "tables": {},
        "guardrails": [],
        "created_at": "2026-05-19T12:00:30Z",
    }


@pytest.fixture
def session_guardrail():
    return {
        "finding_id": "gr_session_1",
        "category": "multiple_comparisons",
        "severity": "warning",
        "title": "K inferential tests; consider correction",
        "message": "FWER inflation.",
        "evidence": {"k_inferential_tests": 2},
        "recommendation": "Apply BH-FDR.",
    }


# ============================================================
# 1. Pure builder shape and field-completeness
# ============================================================

class TestManifestShape:
    def test_top_level_keys_are_present(
        self, real_data_version, inferential_run, descriptive_run
    ):
        m = build_reproducibility_manifest(
            user_request="Test",
            session_id="sess_1",
            data_versions=[real_data_version],
            analysis_runs=[inferential_run, descriptive_run],
            session_guardrails=[],
        )
        for key in [
            "manifest_version", "manifest_id", "generated_at",
            "session", "environment", "data_versions",
            "analyses", "session_guardrails", "counts",
            "issues_during_manifest_build",
        ]:
            assert key in m, f"Missing top-level manifest key: {key}"

    def test_manifest_version_matches_constant(self):
        m = build_reproducibility_manifest()
        assert m["manifest_version"] == MANIFEST_VERSION

    def test_manifest_id_format(self):
        m = build_reproducibility_manifest()
        assert m["manifest_id"].startswith("manifest_")
        assert len(m["manifest_id"]) >= len("manifest_") + 8

    def test_generated_at_is_iso_utc(self):
        m = build_reproducibility_manifest()
        assert m["generated_at"].endswith("Z")
        assert "T" in m["generated_at"]

    def test_session_block(self):
        m = build_reproducibility_manifest(
            user_request="Why does revenue differ?",
            session_id="sess_42",
        )
        assert m["session"]["user_request"] == "Why does revenue differ?"
        assert m["session"]["session_id"] == "sess_42"


class TestCounts:
    def test_counts_reflect_inputs(
        self, real_data_version, inferential_run, descriptive_run, session_guardrail
    ):
        m = build_reproducibility_manifest(
            data_versions=[real_data_version],
            analysis_runs=[inferential_run, descriptive_run],
            session_guardrails=[session_guardrail],
        )
        assert m["counts"]["n_data_versions"] == 1
        assert m["counts"]["n_analyses"] == 2
        assert m["counts"]["n_session_guardrails"] == 1

    def test_inferential_count_uses_flag_not_category(
        self, real_data_version, inferential_run, descriptive_run
    ):
        # descriptive_run has is_inferential=False, inferential_run has True
        m = build_reproducibility_manifest(
            data_versions=[real_data_version],
            analysis_runs=[inferential_run, descriptive_run],
        )
        assert m["counts"]["n_inferential_analyses"] == 1

    def test_inferential_count_zero_when_no_flag(self):
        # A run lacking the flag must default to non-inferential
        run = {
            "run_id": "x", "tool_name": "anything",
            "status": "ok", "success": True,
            "metrics": {}, "guardrails": [], "arguments": {},
            # NOTE: no is_inferential key at all
        }
        m = build_reproducibility_manifest(analysis_runs=[run])
        assert m["counts"]["n_inferential_analyses"] == 0


# ============================================================
# 2. Environment capture
# ============================================================

class TestEnvironment:
    def test_environment_has_python_version(self):
        env = _capture_environment()
        assert "python_version" in env
        # e.g. "3.12.3"
        parts = env["python_version"].split(".")
        assert len(parts) == 3
        for p in parts:
            assert p.isdigit()

    def test_environment_has_platform(self):
        env = _capture_environment()
        assert env["platform"]
        assert isinstance(env["platform"], str)

    def test_environment_key_packages_includes_scipy_and_statsmodels(self):
        env = _capture_environment()
        # The whole point: a third party must know the stats stack versions
        assert "scipy" in env["key_packages"]
        assert "statsmodels" in env["key_packages"]
        assert "pandas" in env["key_packages"]
        assert "numpy" in env["key_packages"]

    def test_environment_does_not_leak_username_or_hostname(self):
        # Privacy check - the manifest must NOT contain identifiable info
        env = _capture_environment()
        # No keys mentioning user / hostname / env / cwd
        for key in env:
            assert "user" not in key.lower()
            assert "host" not in key.lower()
            assert "env" not in key.lower() or key == "environment"
            assert "cwd" not in key.lower()


# ============================================================
# 3. File hashing
# ============================================================

class TestFileHashing:
    def test_sha256_matches_hashlib_directly(self, temp_workspace):
        # Write a known file
        path = os.path.join(temp_workspace, "data.txt")
        content = b"Hello, audit world!"
        with open(path, "wb") as f:
            f.write(content)

        # Compare our hash to a direct hashlib call
        expected = hashlib.sha256(content).hexdigest()
        actual = _sha256_of_file(path)
        assert actual == expected

    def test_sha256_handles_missing_file_gracefully(self):
        assert _sha256_of_file("/nonexistent/path/zzz") is None
        assert _sha256_of_file("") is None
        assert _sha256_of_file(None) is None

    def test_sha256_is_stable_across_calls(self, real_data_version):
        h1 = _sha256_of_file(real_data_version["path"])
        h2 = _sha256_of_file(real_data_version["path"])
        h3 = _sha256_of_file(real_data_version["path"])
        assert h1 == h2 == h3

    def test_sha256_changes_when_file_changes(self, temp_workspace):
        path = os.path.join(temp_workspace, "data.txt")
        with open(path, "wb") as f:
            f.write(b"v1")
        h1 = _sha256_of_file(path)
        with open(path, "wb") as f:
            f.write(b"v2 - one byte different")
        h2 = _sha256_of_file(path)
        assert h1 != h2

    def test_data_version_entry_records_hash(self, real_data_version):
        entry = _build_data_version_entry(real_data_version)
        assert entry["sha256_of_data_file"] is not None
        # Should be a 64-character hex string for SHA-256
        assert len(entry["sha256_of_data_file"]) == 64

    def test_data_version_entry_handles_missing_file(self, temp_workspace):
        # File never existed -> entry built without crash, hash is None
        v = {
            "version_id": "missing_v1",
            "path": os.path.join(temp_workspace, "does_not_exist.parquet"),
            "n_rows": 0,
            "n_cols": 0,
            "operation": "broken",
        }
        entry = _build_data_version_entry(v)
        assert entry["version_id"] == "missing_v1"
        assert entry["sha256_of_data_file"] is None
        assert entry["data_file_present"] is False

    def test_data_version_entry_can_skip_hashing(self, real_data_version):
        entry = _build_data_version_entry(real_data_version, include_hash=False)
        assert entry["sha256_of_data_file"] is None


# ============================================================
# 4. Determinism -- the central promise
# ============================================================

class TestDeterminism:
    """The brand promise: same inputs in -> byte-identical manifest out
    (modulo session-unique fields). If this breaks, the whole pitch breaks."""

    def _strip_session_unique(self, m: Dict[str, Any]) -> Dict[str, Any]:
        out = dict(m)
        out.pop("manifest_id", None)
        out.pop("generated_at", None)
        return out

    def test_builder_is_deterministic_modulo_session_fields(
        self, real_data_version, inferential_run, descriptive_run, session_guardrail
    ):
        kwargs = dict(
            user_request="Q",
            session_id="S",
            data_versions=[real_data_version],
            analysis_runs=[inferential_run, descriptive_run],
            session_guardrails=[session_guardrail],
        )
        m1 = self._strip_session_unique(build_reproducibility_manifest(**kwargs))
        m2 = self._strip_session_unique(build_reproducibility_manifest(**kwargs))

        s1 = json.dumps(m1, sort_keys=True, default=str)
        s2 = json.dumps(m2, sort_keys=True, default=str)
        assert s1 == s2, "Manifest is not deterministic across calls"

    def test_manifest_id_differs_across_builds(self):
        m1 = build_reproducibility_manifest()
        m2 = build_reproducibility_manifest()
        # Different sessions must have different manifest IDs
        assert m1["manifest_id"] != m2["manifest_id"]

    def test_file_output_is_sorted_keys(
        self, temp_workspace, real_data_version, inferential_run
    ):
        m = build_reproducibility_manifest(
            data_versions=[real_data_version],
            analysis_runs=[inferential_run],
        )
        out_path = os.path.join(temp_workspace, "m.json")
        write_manifest_to_file(m, out_path)

        with open(out_path) as f:
            raw = f.read()

        # Sanity: re-parsing produces the same dict
        parsed = json.loads(raw)
        assert parsed["manifest_version"] == m["manifest_version"]

        # `sort_keys=True` means top-level keys appear alphabetically.
        # 'analyses' must come before 'counts' which must come before 'data_versions'.
        idx_analyses = raw.find('"analyses"')
        idx_counts = raw.find('"counts"')
        idx_data = raw.find('"data_versions"')
        assert 0 < idx_analyses < idx_counts < idx_data

    def test_file_output_two_writes_byte_identical_modulo_session_fields(
        self, temp_workspace, real_data_version, inferential_run, descriptive_run
    ):
        kwargs = dict(
            user_request="Q",
            session_id="S",
            data_versions=[real_data_version],
            analysis_runs=[inferential_run, descriptive_run],
        )
        m1 = build_reproducibility_manifest(**kwargs)
        m2 = build_reproducibility_manifest(**kwargs)

        # Strip session-unique fields
        for m in (m1, m2):
            m.pop("manifest_id", None)
            m.pop("generated_at", None)

        p1 = os.path.join(temp_workspace, "m1.json")
        p2 = os.path.join(temp_workspace, "m2.json")
        write_manifest_to_file(m1, p1)
        write_manifest_to_file(m2, p2)

        with open(p1, "rb") as f:
            b1 = f.read()
        with open(p2, "rb") as f:
            b2 = f.read()

        assert b1 == b2


# ============================================================
# 5. Key-result extraction
# ============================================================

class TestKeyResultExtraction:
    def test_only_whitelisted_fields_appear(self):
        metrics = {
            "method": "One-way ANOVA",
            "F_statistic": 4.5,
            "p_value": 0.012,
            "secret_internal": "should not be in manifest",
            "some_other_thing": [1, 2, 3],
        }
        out = _extract_key_results(metrics)
        assert "method" in out
        assert "F_statistic" in out
        assert "p_value" in out
        assert "secret_internal" not in out
        assert "some_other_thing" not in out

    def test_none_values_are_dropped(self):
        metrics = {
            "method": "X",
            "p_value": None,
            "F_statistic": 1.2,
        }
        out = _extract_key_results(metrics)
        assert "p_value" not in out
        assert "method" in out
        assert "F_statistic" in out

    def test_empty_metrics_yields_empty_dict(self):
        assert _extract_key_results({}) == {}
        assert _extract_key_results(None) == {}

    def test_extraction_covers_all_test_families(self):
        # A representative key for each supported test family must be in the
        # whitelist; this guards against accidentally dropping coverage when
        # someone edits _KEY_RESULT_FIELDS.
        from core.reproducibility import _KEY_RESULT_FIELDS

        # ANOVA
        assert "F_statistic" in _KEY_RESULT_FIELDS
        assert "eta_squared" in _KEY_RESULT_FIELDS
        # t-test
        assert "t_statistic" in _KEY_RESULT_FIELDS
        assert "cohens_d" in _KEY_RESULT_FIELDS
        # Nonparametric
        assert "U_statistic" in _KEY_RESULT_FIELDS
        assert "H_statistic" in _KEY_RESULT_FIELDS
        assert "rank_biserial_correlation" in _KEY_RESULT_FIELDS
        # Paired
        assert "cohens_d_z" in _KEY_RESULT_FIELDS
        assert "hodges_lehmann_pseudomedian" in _KEY_RESULT_FIELDS
        # Regression
        assert "r_squared" in _KEY_RESULT_FIELDS
        # Power
        assert "achieved_power" in _KEY_RESULT_FIELDS
        assert "target_power" in _KEY_RESULT_FIELDS
        # Chi-square
        assert "chi_square_statistic" in _KEY_RESULT_FIELDS
        assert "cramers_v" in _KEY_RESULT_FIELDS
        # Correlation
        assert "correlation" in _KEY_RESULT_FIELDS


# ============================================================
# 6. Guardrail summarization
# ============================================================

class TestGuardrailSummary:
    def test_empty_guardrails(self):
        out = _summarize_guardrails([])
        assert out["total"] == 0
        assert out["by_severity"] == {}
        assert out["titles"] == []

    def test_counts_by_severity(self):
        guardrails = [
            {"severity": "info", "category": "x", "title": "t1"},
            {"severity": "warning", "category": "x", "title": "t2"},
            {"severity": "warning", "category": "y", "title": "t3"},
            {"severity": "critical", "category": "z", "title": "t4"},
        ]
        out = _summarize_guardrails(guardrails)
        assert out["total"] == 4
        assert out["by_severity"] == {"info": 1, "warning": 2, "critical": 1}
        assert len(out["titles"]) == 4

    def test_missing_severity_defaults_to_info(self):
        guardrails = [{"category": "x", "title": "no severity"}]
        out = _summarize_guardrails(guardrails)
        assert out["by_severity"].get("info") == 1

    def test_malformed_entries_are_skipped(self):
        guardrails = [
            {"severity": "info", "title": "ok"},
            "not a dict",      # skipped
            None,              # skipped
        ]
        out = _summarize_guardrails(guardrails)
        assert out["total"] == 1


# ============================================================
# 7. Robustness -- empty / missing / malformed inputs
# ============================================================

class TestRobustness:
    def test_completely_empty_session(self):
        # All inputs missing -- must not raise, must produce a usable manifest
        m = build_reproducibility_manifest()
        assert m["manifest_version"] == MANIFEST_VERSION
        assert m["counts"]["n_data_versions"] == 0
        assert m["counts"]["n_analyses"] == 0
        assert m["counts"]["n_inferential_analyses"] == 0
        # Issues field should explain what was missing
        issues = m["issues_during_manifest_build"]
        assert any("data versions" in i.lower() for i in issues)
        assert any("analysis runs" in i.lower() for i in issues)

    def test_partial_session_with_data_only(self, real_data_version):
        m = build_reproducibility_manifest(data_versions=[real_data_version])
        assert m["counts"]["n_data_versions"] == 1
        assert m["counts"]["n_analyses"] == 0
        # Data version was provided so no missing-data issue
        issues_text = " ".join(m["issues_during_manifest_build"]).lower()
        assert "data versions" not in issues_text

    def test_handles_non_dict_data_versions(self):
        # If something garbage is in the list, the builder should not crash
        m = build_reproducibility_manifest(
            data_versions=["not a dict", None],
        )
        # The bogus entries get skipped (returned as empty dicts then filtered)
        # The key invariant is: no exception
        assert isinstance(m["data_versions"], list)

    def test_handles_non_dict_analysis_runs(self):
        m = build_reproducibility_manifest(
            analysis_runs=["bogus", None, {"run_id": "real"}],
        )
        # The real one survives, bogus ones don't crash anything
        assert m["counts"]["n_analyses"] >= 1


# ============================================================
# 8. File output
# ============================================================

class TestFileOutput:
    def test_write_creates_parent_directory(self, temp_workspace):
        m = build_reproducibility_manifest()
        nested = os.path.join(temp_workspace, "a", "b", "c", "manifest.json")
        out = write_manifest_to_file(m, nested)
        assert os.path.exists(out)

    def test_written_json_is_valid_and_complete(
        self, temp_workspace, real_data_version, inferential_run
    ):
        m = build_reproducibility_manifest(
            user_request="Q",
            session_id="S",
            data_versions=[real_data_version],
            analysis_runs=[inferential_run],
        )
        out = os.path.join(temp_workspace, "m.json")
        write_manifest_to_file(m, out)

        with open(out) as f:
            reloaded = json.load(f)

        # Round-trip preserves the manifest exactly
        assert reloaded == m

    def test_writes_utf8(self, temp_workspace):
        # Make sure non-ascii in user_request is preserved
        m = build_reproducibility_manifest(
            user_request="Compare revenue by 地区 (region)",
        )
        out = os.path.join(temp_workspace, "m.json")
        write_manifest_to_file(m, out)
        with open(out, encoding="utf-8") as f:
            reloaded = json.load(f)
        assert reloaded["session"]["user_request"] == m["session"]["user_request"]


# ============================================================
# 9. Plugin integration
# ============================================================

class TestPluginIntegration:
    """The plugin layer wraps the builder. These tests cover the contract
    between the plugin runtime and the manifest module."""

    def test_plugin_is_registered(self):
        from core.analysis_tool_plugins import get_plugin
        plugin = get_plugin("export_reproducibility_manifest")
        assert plugin is not None
        assert plugin.tool_name == "export_reproducibility_manifest"

    def test_plugin_is_not_inferential(self):
        # Critical: the manifest export must never inflate the FWER counter
        from core.analysis_tool_plugins import get_plugin
        plugin = get_plugin("export_reproducibility_manifest")
        assert plugin.is_inferential is False

    def test_plugin_declares_provenance_evidence(self):
        from core.analysis_tool_plugins import get_plugin
        plugin = get_plugin("export_reproducibility_manifest")
        assert "reproducibility" in plugin.evidence_categories
        assert "audit" in plugin.evidence_categories

    def _make_ctx(self, **kwargs):
        class Ctx:
            pass
        c = Ctx()
        c.arguments = kwargs.pop("arguments", {})
        for k, v in kwargs.items():
            setattr(c, k, v)
        return c

    def test_plugin_run_with_real_session(
        self, temp_workspace, real_data_version, inferential_run, descriptive_run
    ):
        from core.analysis_tool_plugins import get_plugin
        plugin = get_plugin("export_reproducibility_manifest")

        ctx = self._make_ctx(
            workspace_dir=temp_workspace,
            user_request="Test session",
            session_id="sess_42",
            data_versions=[real_data_version],
            analysis_runs=[inferential_run, descriptive_run],
            session_guardrails=[],
        )
        result = plugin.run(ctx)
        assert result["status"] == "ok"
        # An artifact (the JSON file) should be produced
        assert len(result["artifacts"]) == 1
        assert result["artifacts"][0]["mime_type"] == "application/json"
        # The file should exist
        path = result["details"]["output_path"]
        assert os.path.exists(path)

    def test_plugin_run_empty_session_does_not_crash(self, temp_workspace):
        from core.analysis_tool_plugins import get_plugin
        plugin = get_plugin("export_reproducibility_manifest")

        ctx = self._make_ctx(workspace_dir=temp_workspace)
        result = plugin.run(ctx)
        assert result["status"] == "ok"
        # Manifest still written, just with zero analyses
        assert result["details"]["n_analyses"] == 0
        assert result["details"]["n_data_versions"] == 0

    def test_plugin_can_skip_disk_write(
        self, temp_workspace, real_data_version, inferential_run
    ):
        from core.analysis_tool_plugins import get_plugin
        plugin = get_plugin("export_reproducibility_manifest")

        ctx = self._make_ctx(
            workspace_dir=temp_workspace,
            data_versions=[real_data_version],
            analysis_runs=[inferential_run],
            arguments={"write_to_disk": False},
        )
        result = plugin.run(ctx)
        assert result["status"] == "ok"
        assert result["details"]["wrote_to_disk"] is False
        assert result["details"]["output_path"] is None
        assert result["artifacts"] == []

    def test_plugin_can_skip_data_hashing(
        self, temp_workspace, real_data_version, inferential_run
    ):
        from core.analysis_tool_plugins import get_plugin
        plugin = get_plugin("export_reproducibility_manifest")

        ctx = self._make_ctx(
            workspace_dir=temp_workspace,
            data_versions=[real_data_version],
            analysis_runs=[inferential_run],
            arguments={"include_data_hashes": False},
        )
        result = plugin.run(ctx)
        assert result["status"] == "ok"
        # Inside the inline manifest, SHA fields should be None
        full = result["details"]["full_manifest_inline"]
        assert full["data_versions"][0]["sha256_of_data_file"] is None

    def test_plugin_returns_complete_metrics(
        self, temp_workspace, real_data_version, inferential_run, descriptive_run
    ):
        """The metric table must show the key counts in the run UI."""
        from core.analysis_tool_plugins import get_plugin
        plugin = get_plugin("export_reproducibility_manifest")

        ctx = self._make_ctx(
            workspace_dir=temp_workspace,
            data_versions=[real_data_version],
            analysis_runs=[inferential_run, descriptive_run],
        )
        result = plugin.run(ctx)
        d = result["details"]
        assert d["n_data_versions"] == 1
        assert d["n_analyses"] == 2
        assert d["n_inferential_analyses"] == 1
        assert d["manifest_version"] == MANIFEST_VERSION
        assert d["python_version"]
        assert d["key_packages"]


# ============================================================
# 10. Cross-plugin: every existing tool's metrics are extractable
# ============================================================

class TestKeyResultCoverageAcrossPlugins:
    """Sanity check: for each inferential plugin that already exists,
    confirm that calling _extract_key_results on a representative metrics
    dict produces a non-empty result. If this fails, the manifest will
    appear 'empty' for that test family in the wild."""

    def test_anova_metrics_extractable(self):
        metrics = {
            "method": "One-way ANOVA",
            "F_statistic": 5.0,
            "p_value": 0.01,
            "eta_squared": 0.2,
            "valid_group_count": 3,
            "alpha": 0.05,
        }
        out = _extract_key_results(metrics)
        assert out  # non-empty
        assert "F_statistic" in out
        assert "p_value" in out
        assert "eta_squared" in out

    def test_t_test_metrics_extractable(self):
        metrics = {
            "method": "Welch two-sample t-test",
            "t_statistic": 2.0,
            "p_value": 0.05,
            "cohens_d": 0.5,
            "degrees_of_freedom": 30,
        }
        out = _extract_key_results(metrics)
        assert "t_statistic" in out
        assert "cohens_d" in out

    def test_nonparametric_metrics_extractable(self):
        metrics = {
            "method": "Mann-Whitney U",
            "U_statistic": 50,
            "p_value": 0.03,
            "rank_biserial_correlation": 0.3,
            "hodges_lehmann_location_shift": 2.5,
        }
        out = _extract_key_results(metrics)
        assert "U_statistic" in out
        assert "rank_biserial_correlation" in out
        assert "hodges_lehmann_location_shift" in out

    def test_paired_comparison_metrics_extractable(self):
        metrics = {
            "method": "Paired t-test",
            "cohens_d_z": 0.4,
            "p_value": 0.04,
            "hodges_lehmann_pseudomedian": 1.2,
            "n_complete_pairs": 25,
        }
        out = _extract_key_results(metrics)
        assert "cohens_d_z" in out
        assert "hodges_lehmann_pseudomedian" in out
        assert "n_complete_pairs" in out

    def test_regression_metrics_extractable(self):
        metrics = {
            "r_squared": 0.45,
            "adj_r_squared": 0.43,
            "nobs": 200,
        }
        out = _extract_key_results(metrics)
        assert "r_squared" in out
        assert "adj_r_squared" in out

    def test_power_analysis_metrics_extractable(self):
        metrics = {
            "test_type": "two_sample_t",
            "mode": "sample_size",
            "achieved_power": 0.80,
            "target_power": 0.80,
            "effect_size": 0.5,
            "n_value": 64,
            "n_semantics": "per-group",
        }
        out = _extract_key_results(metrics)
        assert "achieved_power" in out
        assert "target_power" in out
        assert "n_value" in out
        assert "test_type" in out
        assert "mode" in out

    def test_chi_square_metrics_extractable(self):
        metrics = {
            "chi_square_statistic": 8.5,
            "p_value": 0.014,
            "cramers_v": 0.25,
        }
        out = _extract_key_results(metrics)
        assert "chi_square_statistic" in out
        assert "cramers_v" in out

    def test_correlation_metrics_extractable(self):
        metrics = {
            "correlation": 0.45,
            "p_value": 0.001,
            "nobs": 100,
        }
        out = _extract_key_results(metrics)
        assert "correlation" in out
        assert "nobs" in out

    def test_diagnostics_metrics_extractable(self):
        metrics = {
            "max_vif": 4.2,
            "breusch_pagan_lm_p_value": 0.07,
            "durbin_watson_statistic": 1.95,
            "residuals_appear_normal_at_0_05": True,
            "n_high_cooks_distance": 0,
        }
        out = _extract_key_results(metrics)
        assert "max_vif" in out
        assert "breusch_pagan_lm_p_value" in out
        assert "durbin_watson_statistic" in out