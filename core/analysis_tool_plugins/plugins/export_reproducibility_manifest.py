"""
export_reproducibility_manifest plugin.

Thin wrapper around `core.reproducibility.build_reproducibility_manifest`. The
plugin exposes the manifest export as a tool the supervisor can call, and
writes the JSON to the workspace so it appears as a downloadable artifact.

The manifest itself is the deliverable; this plugin is just the entry point.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, Tuple

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin
from core.reproducibility import (
    build_reproducibility_manifest,
    write_manifest_to_file,
    MANIFEST_VERSION,
)


# ==========================================================
# Helpers
# ==========================================================

def _get_arguments(context) -> dict[str, Any]:
    return getattr(context, "arguments", None) or getattr(context, "args", None) or {}


def _failed(error_code: str, message: str, exc: Exception) -> Dict[str, Any]:
    return {
        "status": "failed",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        "artifacts": [],
    }


def _resolve_output_path(context, arguments: Dict[str, Any]) -> str:
    """
    Where to write the manifest. Preference order:
      1. arguments["output_path"] if provided
      2. <workspace_dir>/reproducibility/manifest_<timestamp>.json
      3. /tmp fallback
    """
    explicit = arguments.get("output_path")
    if explicit:
        return str(explicit)

    workspace = getattr(context, "workspace_dir", None)

    from datetime import datetime, timezone
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    filename = f"manifest_{stamp}.json"

    if workspace:
        return os.path.join(str(workspace), "reproducibility", filename)

    return os.path.join("/tmp", filename)


def _resolve_session_context(context) -> Dict[str, Any]:
    return {
        "user_request": getattr(context, "user_request", None),
        "session_id": (
            getattr(context, "session_id", None)
            or getattr(context, "thread_id", None)
        ),
        "data_versions": list(getattr(context, "data_versions", []) or []),
        "analysis_runs": list(getattr(context, "analysis_runs", []) or []),
        "session_guardrails": list(
            getattr(context, "session_guardrails", []) or []
        ),
    }


# ==========================================================
# Execute
# ==========================================================

def execute_export_reproducibility_manifest(context) -> Dict[str, Any]:
    arguments = _get_arguments(context)

    include_data_hashes_arg = arguments.get("include_data_hashes", True)
    include_data_hashes = bool(include_data_hashes_arg) if include_data_hashes_arg is not None else True

    write_to_disk_arg = arguments.get("write_to_disk", True)
    write_to_disk = bool(write_to_disk_arg) if write_to_disk_arg is not None else True

    try:
        session = _resolve_session_context(context)

        manifest = build_reproducibility_manifest(
            user_request=session["user_request"],
            session_id=session["session_id"],
            data_versions=session["data_versions"],
            analysis_runs=session["analysis_runs"],
            session_guardrails=session["session_guardrails"],
            include_data_hashes=include_data_hashes,
        )

        artifacts: list[Dict[str, Any]] = []
        written_path: str | None = None

        if write_to_disk:
            out_path = _resolve_output_path(context, arguments)
            written_path = write_manifest_to_file(manifest, out_path)

            try:
                size_bytes = os.path.getsize(written_path)
            except Exception:
                size_bytes = None

            artifacts.append({
                "kind": "file",
                "path": written_path,
                "basename": os.path.basename(written_path),
                "mime_type": "application/json",
                "size_bytes": size_bytes,
                "description": "Reproducibility manifest (JSON).",
            })

        counts = manifest.get("counts", {}) or {}
        env = manifest.get("environment", {}) or {}

        details = {
            "manifest_version": manifest.get("manifest_version"),
            "manifest_id": manifest.get("manifest_id"),
            "generated_at": manifest.get("generated_at"),
            "output_path": written_path,
            "wrote_to_disk": bool(written_path),
            "include_data_hashes": include_data_hashes,
            # Surface counts for the metric table
            "n_data_versions": counts.get("n_data_versions"),
            "n_analyses": counts.get("n_analyses"),
            "n_inferential_analyses": counts.get("n_inferential_analyses"),
            "n_session_guardrails": counts.get("n_session_guardrails"),
            # Environment summary
            "python_version": env.get("python_version"),
            "platform": env.get("platform"),
            "key_packages": env.get("key_packages") or {},
            # Tables surface the structured content of the manifest
            "data_versions": manifest.get("data_versions") or [],
            "analyses": manifest.get("analyses") or [],
            "session_guardrails": manifest.get("session_guardrails") or [],
            "issues_during_manifest_build": manifest.get(
                "issues_during_manifest_build"
            ) or [],
            "full_manifest_inline": manifest,
        }

        n_analyses = counts.get("n_analyses") or 0
        n_data = counts.get("n_data_versions") or 0

        message = (
            f"Reproducibility manifest written for {n_analyses} analyses across "
            f"{n_data} data version(s)."
        )

        if written_path:
            message += f" Saved to `{written_path}`."

        return {
            "status": "ok",
            "message": message,
            "recoverable": False,
            "details": details,
            "artifacts": artifacts,
        }

    except Exception as exc:
        return _failed(
            "EXPORT_MANIFEST_EXCEPTION",
            "Reproducibility manifest export failed.",
            exc,
        )


# ==========================================================
# Extractor
# ==========================================================

def extract_export_reproducibility_manifest(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Reproducibility Manifest"

    n_data = payload.get("n_data_versions") or 0
    n_analyses = payload.get("n_analyses") or 0
    n_inferential = payload.get("n_inferential_analyses") or 0

    summary = (
        f"Captured {n_analyses} analysis run(s) ({n_inferential} inferential) "
        f"across {n_data} data version(s). The manifest records software "
        f"versions, data hashes, analysis arguments, and key results so the "
        f"session can be reproduced or audited."
    )

    if payload.get("wrote_to_disk"):
        summary += f" Saved to `{payload.get('output_path')}`."

    metrics = compact_dict({
        "manifest_version": payload.get("manifest_version"),
        "manifest_id": payload.get("manifest_id"),
        "generated_at": payload.get("generated_at"),
        "output_path": payload.get("output_path"),
        "wrote_to_disk": payload.get("wrote_to_disk"),
        "include_data_hashes": payload.get("include_data_hashes"),
        "n_data_versions": payload.get("n_data_versions"),
        "n_analyses": payload.get("n_analyses"),
        "n_inferential_analyses": payload.get("n_inferential_analyses"),
        "n_session_guardrails": payload.get("n_session_guardrails"),
        "python_version": payload.get("python_version"),
        "platform": payload.get("platform"),
    })

    tables: Dict[str, Any] = {}

    key_packages = payload.get("key_packages") or {}
    if key_packages:
        tables["key_packages"] = [
            {"package": name, "version": version}
            for name, version in sorted(key_packages.items())
        ]

    # Show a compact view of each data version (hash + basics)
    data_versions = payload.get("data_versions") or []
    if data_versions:
        tables["data_versions"] = [
            {
                "version_id": dv.get("version_id"),
                "operation": dv.get("operation"),
                "parent_version_id": dv.get("parent_version_id"),
                "n_rows": dv.get("n_rows"),
                "n_cols": dv.get("n_cols"),
                "sha256_of_data_file": dv.get("sha256_of_data_file"),
                "data_file_present": dv.get("data_file_present"),
                "created_at": dv.get("created_at"),
            }
            for dv in data_versions
        ]

    # Show a compact view of each analysis
    analyses = payload.get("analyses") or []
    if analyses:
        tables["analyses"] = [
            {
                "run_id": a.get("run_id"),
                "tool_name": a.get("tool_name"),
                "title": a.get("title"),
                "status": a.get("status"),
                "is_inferential": a.get("is_inferential"),
                "data_version_id": a.get("data_version_id"),
                "n_arguments": len(a.get("arguments") or {}),
                "n_key_results": len(a.get("key_results") or {}),
                "n_guardrail_findings": (
                    (a.get("guardrails_summary") or {}).get("total", 0)
                ),
            }
            for a in analyses
        ]

    issues = payload.get("issues_during_manifest_build") or []
    if issues:
        tables["issues_during_manifest_build"] = [{"issue": i} for i in issues]

    metadata = compact_dict({
        "full_manifest_inline": payload.get("full_manifest_inline"),
        "session_guardrails": payload.get("session_guardrails"),
    })

    return title, summary, metrics, tables, metadata


# ==========================================================
# Display config
# ==========================================================

EXPORT_MANIFEST_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "manifest_version": "Manifest schema version",
            "manifest_id": "Manifest ID",
            "generated_at": "Generated at (UTC)",
            "output_path": "Saved to",
            "wrote_to_disk": "Wrote to disk",
            "include_data_hashes": "Included data hashes",
            "n_data_versions": "Data versions recorded",
            "n_analyses": "Analyses recorded",
            "n_inferential_analyses": "Inferential analyses recorded",
            "n_session_guardrails": "Session-level guardrails",
            "python_version": "Python version",
            "platform": "Platform",
        },
        formatters={},
        order=[
            "manifest_version",
            "manifest_id",
            "generated_at",
            "output_path",
            "wrote_to_disk",
            "include_data_hashes",
            "n_data_versions",
            "n_analyses",
            "n_inferential_analyses",
            "n_session_guardrails",
            "python_version",
            "platform",
        ],
    ),
    tables={
        "key_packages": TableDisplayConfig(
            column_labels={
                "package": "Package",
                "version": "Version",
            },
            column_order=["package", "version"],
        ),
        "data_versions": TableDisplayConfig(
            column_labels={
                "version_id": "Version ID",
                "operation": "Operation",
                "parent_version_id": "Parent version",
                "n_rows": "Rows",
                "n_cols": "Columns",
                "sha256_of_data_file": "SHA-256 of data file",
                "data_file_present": "File present",
                "created_at": "Created at",
            },
            column_order=[
                "version_id",
                "operation",
                "parent_version_id",
                "n_rows",
                "n_cols",
                "sha256_of_data_file",
                "data_file_present",
                "created_at",
            ],
        ),
        "analyses": TableDisplayConfig(
            column_labels={
                "run_id": "Run ID",
                "tool_name": "Tool",
                "title": "Title",
                "status": "Status",
                "is_inferential": "Inferential",
                "data_version_id": "Data version",
                "n_arguments": "# args",
                "n_key_results": "# key results",
                "n_guardrail_findings": "# guardrail findings",
            },
            column_order=[
                "run_id",
                "tool_name",
                "title",
                "status",
                "is_inferential",
                "data_version_id",
                "n_arguments",
                "n_key_results",
                "n_guardrail_findings",
            ],
        ),
        "issues_during_manifest_build": TableDisplayConfig(
            column_labels={"issue": "Issue"},
            column_order=["issue"],
        ),
    },
)


# ==========================================================
# Plugin registration
# ==========================================================

PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="export_reproducibility_manifest",
    display_name="Reproducibility Manifest",
    is_inferential=False,
    evidence_categories=["reproducibility", "audit"],
    evidence_category_roles={
        "reproducibility": "provenance",
        "audit": "provenance",
    },
    description=(
        "Export a deterministic, audit-grade JSON manifest of the session. "
        "Captures software versions (Python, scipy, statsmodels, pandas, numpy), "
        "data lineage with SHA-256 hashes of every data version, every "
        "analysis run with its exact arguments and key results, and "
        "session-level guardrails. Designed so a third party can verify or "
        "reproduce the analysis from the manifest alone."
    ),
    usage_guidance=(
        "Call this tool when the user asks to 'export', 'save', or 'download' "
        "the analysis as a reproducibility report, supplementary material, or "
        "audit trail. The manifest is written as JSON in the workspace. The "
        "manifest is the deliverable; the metric table on this run is just a "
        "summary preview."
    ),
    use_when=[
        "The user wants to share or archive what was done in this session.",
        "The user mentions reproducibility, audit, supplementary materials, peer review, or compliance.",
        "The user is wrapping up a session and wants a permanent record.",
    ],
    do_not_use_when=[
        "The user wants to run a new statistical test; pick the appropriate test plugin instead.",
        "No analyses have been run yet in this session (the manifest would be empty).",
    ],
    requires_data_source=None,
    produces_active_dataset=False,
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={},
        optional={
            "include_data_hashes": bool,
            "write_to_disk": bool,
            "output_path": str,
        },
        column_args=[],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_export_reproducibility_manifest,
    extractor=extract_export_reproducibility_manifest,
    guardrail_evaluators=[],
    display_config=EXPORT_MANIFEST_DISPLAY,
    examples=[
        {
            "user_request": "Export a reproducibility manifest for this session.",
            "arguments": {},
        },
        {
            "user_request": "Save an audit trail of everything we did so I can attach it to my paper.",
            "arguments": {},
        },
    ],
))