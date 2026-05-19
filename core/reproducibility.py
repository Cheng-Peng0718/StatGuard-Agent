"""
Reproducibility manifest.

Produces a JSON record that captures everything needed to reproduce or audit a
session: software versions, data lineage with cryptographic hashes, analysis
parameters and key results, and session-level guardrails.

The manifest is designed for three audiences:

1. Academic researchers. Attach the manifest as a supplementary file to a
   submission so reviewers can verify which dataset and which methods produced
   each reported number.
2. Regulated industries (clinical, finance, audit). The manifest is the audit
   artifact: deterministic plugin calls with explicit arguments, data hashes,
   and software versions are sufficient to reconstruct the analysis bit-for-bit
   given the original data.
3. The future self. Six months from now, you can read the manifest and know
   exactly what was done.

Design choices documented inline.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import uuid
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional


MANIFEST_VERSION = "1.0"

# Packages whose version we always record because they directly affect
# statistical results. Add to this list (not remove) so older manifests
# remain comparable.
_KEY_STATS_PACKAGES = (
    "scipy",
    "statsmodels",
    "numpy",
    "pandas",
    "scikit-learn",
)


# ============================================================
# Low-level helpers
# ============================================================

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256_of_file(path: str, chunk_size: int = 1 << 20) -> Optional[str]:
    """
    SHA-256 of a file's bytes. Chunked so it works on multi-GB parquet files
    without loading them into memory.

    Returns None if the file cannot be read; the manifest carries the failure
    note rather than crashing the export.
    """
    if not path or not os.path.exists(path):
        return None

    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _capture_environment() -> Dict[str, Any]:
    """
    Capture only what affects statistical results. We deliberately do NOT
    capture environment variables, user names, or hostnames - those would
    create privacy issues without aiding reproducibility.
    """
    return {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "key_packages": {
            name: _package_version(name) for name in _KEY_STATS_PACKAGES
        },
    }


# ============================================================
# Field selection for analyses
# ============================================================

# Keys promoted into each analysis "key_results" block. These are the fields
# a reviewer or auditor would actually want to see at a glance. Anything not
# in this list is still recoverable from the underlying analysis_run, but the
# manifest stays small.
_KEY_RESULT_FIELDS = (
    # Test identification
    "method",
    "test_family",
    "test_type",
    "mode",
    # Inference
    "p_value",
    "significant_at_alpha",
    "significant_at_0_05",
    "alpha",
    "degrees_of_freedom",
    "degrees_of_freedom_between",
    "degrees_of_freedom_within",
    # Test statistics
    "t_statistic",
    "F_statistic",
    "U_statistic",
    "H_statistic",
    "W_statistic",
    "chi_square_statistic",
    "correlation",
    "r_squared",
    "adj_r_squared",
    # Sample sizes
    "nobs",
    "n_complete_pairs",
    "valid_group_count",
    # Effect sizes
    "effect_size",
    "effect_size_name",
    "effect_size_magnitude",
    "effect_size_ci_low",
    "effect_size_ci_high",
    "cohens_d",
    "cohens_d_ci_low",
    "cohens_d_ci_high",
    "cohens_d_z",
    "cohens_d_z_ci_low",
    "cohens_d_z_ci_high",
    "epsilon_squared",
    "eta_squared",
    "eta_squared_ci_low",
    "eta_squared_ci_high",
    "omega_squared",
    "omega_squared_ci_low",
    "omega_squared_ci_high",
    "cramers_v",
    "rank_biserial_correlation",
    "hodges_lehmann_pseudomedian",
    "hodges_lehmann_location_shift",
    # Power analysis
    "achieved_power",
    "target_power",
    "n_value",
    "n_semantics",
    "total_sample_size",
    # Diagnostics
    "max_vif",
    "breusch_pagan_lm_p_value",
    "durbin_watson_statistic",
    "residuals_appear_normal_at_0_05",
    "n_high_cooks_distance",
)


def _extract_key_results(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """
    Subset metrics to the audit-relevant keys. Keeps the manifest readable.
    """
    if not metrics:
        return {}

    return {
        key: metrics[key]
        for key in _KEY_RESULT_FIELDS
        if key in metrics and metrics[key] is not None
    }


def _summarize_guardrails(guardrails: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compact summary of guardrail findings per analysis. We surface counts by
    severity, plus the titles, since the full evidence dict can be large.
    """
    if not guardrails:
        return {"total": 0, "by_severity": {}, "titles": []}

    by_severity: Dict[str, int] = {}
    titles: List[Dict[str, str]] = []

    for finding in guardrails:
        if not isinstance(finding, dict):
            continue

        severity = str(finding.get("severity") or "info")
        by_severity[severity] = by_severity.get(severity, 0) + 1

        titles.append({
            "severity": severity,
            "category": str(finding.get("category") or ""),
            "title": str(finding.get("title") or ""),
        })

    return {
        "total": len(titles),
        "by_severity": by_severity,
        "titles": titles,
    }


# ============================================================
# Section builders
# ============================================================

def _build_data_version_entry(
    version: Dict[str, Any],
    *,
    include_hash: bool = True,
) -> Dict[str, Any]:
    """
    One data-version record in the manifest. The SHA-256 of the on-disk
    parquet is the single most important field for reproducibility audits.
    """
    if not isinstance(version, dict):
        return {}

    path = version.get("path")
    sha256 = _sha256_of_file(path) if include_hash and path else None

    return {
        "version_id": version.get("version_id"),
        "parent_version_id": version.get("parent_version_id"),
        "operation": version.get("operation"),
        "description": version.get("description"),
        "created_by": version.get("created_by"),
        "created_at": version.get("created_at"),
        "n_rows": version.get("n_rows"),
        "n_cols": version.get("n_cols"),
        "path_basename": os.path.basename(path) if path else None,
        "sha256_of_data_file": sha256,
        "data_file_present": bool(path and os.path.exists(path)) if path else False,
    }


def _build_analysis_entry(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    One analysis-run record in the manifest. Carries enough information for
    a third party to re-run the analysis: the tool name plus the exact
    arguments passed to it, plus the headline results that would appear in
    a published table.
    """
    if not isinstance(run, dict):
        return {}

    metrics = run.get("metrics") or {}

    return {
        "run_id": run.get("run_id"),
        "action_id": run.get("action_id"),
        "tool_name": run.get("tool_name"),
        "title": run.get("title"),
        "status": run.get("status"),
        "success": run.get("success"),
        "is_inferential": bool(run.get("is_inferential", False)),
        "evidence_categories": list(run.get("evidence_categories") or []),
        "data_version_id": run.get("data_version_id"),
        "arguments": dict(run.get("arguments") or {}),
        "key_results": _extract_key_results(metrics),
        "guardrails_summary": _summarize_guardrails(run.get("guardrails") or []),
        "created_at": run.get("created_at"),
    }


# ============================================================
# Top-level builder
# ============================================================

def build_reproducibility_manifest(
    *,
    user_request: Optional[str] = None,
    session_id: Optional[str] = None,
    data_versions: Optional[List[Dict[str, Any]]] = None,
    analysis_runs: Optional[List[Dict[str, Any]]] = None,
    session_guardrails: Optional[List[Dict[str, Any]]] = None,
    include_data_hashes: bool = True,
) -> Dict[str, Any]:
    """
    Build the full reproducibility manifest dict.

    All parameters are kwarg-only on purpose: future fields can be added
    without breaking existing callers. The function never raises - it always
    produces a manifest with whatever it could capture, and the manifest
    itself records what was missing.
    """
    issues: List[str] = []

    if not data_versions:
        issues.append("No data versions were available for this session.")
    if not analysis_runs:
        issues.append("No analysis runs were available for this session.")

    data_section: List[Dict[str, Any]] = []
    for v in data_versions or []:
        try:
            entry = _build_data_version_entry(v, include_hash=include_data_hashes)
            if entry:
                data_section.append(entry)
        except Exception as exc:
            issues.append(f"Could not record one data version: {type(exc).__name__}")

    analyses_section: List[Dict[str, Any]] = []
    for r in analysis_runs or []:
        try:
            entry = _build_analysis_entry(r)
            if entry:
                analyses_section.append(entry)
        except Exception as exc:
            issues.append(f"Could not record one analysis run: {type(exc).__name__}")

    session_guardrails_section: List[Dict[str, Any]] = []
    for finding in session_guardrails or []:
        if not isinstance(finding, dict):
            continue
        session_guardrails_section.append({
            "finding_id": finding.get("finding_id"),
            "category": finding.get("category"),
            "severity": finding.get("severity"),
            "title": finding.get("title"),
            "message": finding.get("message"),
            "evidence": finding.get("evidence"),
            "recommendation": finding.get("recommendation"),
        })

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "manifest_id": f"manifest_{uuid.uuid4().hex[:12]}",
        "generated_at": _utc_now_iso(),
        "session": {
            "session_id": session_id,
            "user_request": user_request,
        },
        "environment": _capture_environment(),
        "data_versions": data_section,
        "analyses": analyses_section,
        "session_guardrails": session_guardrails_section,
        "counts": {
            "n_data_versions": len(data_section),
            "n_analyses": len(analyses_section),
            "n_inferential_analyses": sum(
                1 for a in analyses_section if a.get("is_inferential")
            ),
            "n_session_guardrails": len(session_guardrails_section),
        },
        "issues_during_manifest_build": issues,
    }

    return manifest


def write_manifest_to_file(
    manifest: Dict[str, Any],
    out_path: str,
    *,
    indent: int = 2,
) -> str:
    """
    Persist the manifest to disk as canonical JSON. Returns the path written.

    sort_keys=True is deliberate: byte-identical manifest files for two runs
    over the same inputs are a soft proof of determinism above the LLM layer.
    """
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=indent, sort_keys=True, default=str)

    return out_path