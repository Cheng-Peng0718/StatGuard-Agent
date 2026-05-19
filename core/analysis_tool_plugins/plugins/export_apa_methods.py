"""
export_apa_methods plugin.

Walks the session's analysis_runs, asks each run's source plugin for its
APA-style Methods paragraph (via the plugin's `apa_methods_writer`), and
assembles the result into a full APA 7th-edition Methods section.

Outputs:
  - <workspace>/apa_methods/methods_<timestamp>.md     (Markdown with italics)
  - <workspace>/apa_methods/methods_<timestamp>.txt    (Plaintext, italics stripped)

This is the most user-visible differentiator for academic users: it turns
the session into a paragraph they can paste into their paper.
"""

from __future__ import annotations

import os
import re
import sys
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from typing import Any, Dict, List, Optional, Tuple

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
)
from core.analysis_tool_plugins.registry import register_plugin, get_plugin


# ============================================================
# Helpers
# ============================================================

def _get_arguments(context) -> dict[str, Any]:
    return getattr(context, "arguments", None) or getattr(context, "args", None) or {}


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _strip_markdown_italics(text: str) -> str:
    """Convert `*xxx*` to plain text (for the .txt output)."""
    return re.sub(r"\*([^*]+)\*", r"\1", text)


def _package_version(name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(name)
    except Exception:
        return None


def _resolve_output_dir(context, arguments: Dict[str, Any]) -> str:
    explicit = arguments.get("output_dir")
    if explicit:
        return str(explicit)

    workspace = getattr(context, "workspace_dir", None)
    if workspace:
        return os.path.join(str(workspace), "apa_methods")

    return "/tmp/apa_methods"


# ============================================================
# Assembly
# ============================================================

def _build_software_paragraph() -> str:
    """
    The Software subsection. Mention only packages whose version meaningfully
    affects statistical output.
    """
    py = ".".join(map(str, sys.version_info[:3]))
    scipy_v = _package_version("scipy")
    sm_v = _package_version("statsmodels")
    np_v = _package_version("numpy")
    pd_v = _package_version("pandas")

    bits = [f"Python {py}"]
    if scipy_v:
        bits.append(f"SciPy {scipy_v}")
    if sm_v:
        bits.append(f"statsmodels {sm_v}")
    if np_v:
        bits.append(f"NumPy {np_v}")
    if pd_v:
        bits.append(f"pandas {pd_v}")

    versions_str = ", ".join(bits)

    return (
        f"All analyses were conducted in {versions_str}. "
        "Statistical routines were executed by a deterministic, plugin-based framework: "
        "the language model's role was confined to routing user requests to the appropriate "
        "analysis tool, while every test statistic, effect size, confidence interval, and "
        "post-hoc procedure was computed by fixed code paths. A complete reproducibility "
        "manifest (software versions, data file hashes, analysis parameters, and key results "
        "for every run) is available in the supplementary materials."
    )


def _collect_paragraphs(
    analysis_runs: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    For each analysis_run, look up its plugin's apa_methods_writer and call it.

    Returns (paragraphs, skipped). Each paragraph entry is
        {"run_id": ..., "tool_name": ..., "title": ..., "paragraph": str}.
    Each skipped entry is
        {"run_id": ..., "tool_name": ..., "reason": str}.
    """
    paragraphs: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []

    for run in analysis_runs or []:
        if not isinstance(run, dict):
            continue

        tool_name = run.get("tool_name")
        run_id = run.get("run_id")
        title = run.get("title")

        if run.get("status") not in {"ok", "warning"}:
            skipped.append({
                "run_id": run_id,
                "tool_name": tool_name,
                "reason": f"run status was {run.get('status')!r}",
            })
            continue

        plugin = get_plugin(tool_name) if tool_name else None
        writer = getattr(plugin, "apa_methods_writer", None) if plugin else None

        if writer is None:
            skipped.append({
                "run_id": run_id,
                "tool_name": tool_name,
                "reason": "plugin does not provide an apa_methods_writer",
            })
            continue

        try:
            text = writer(run)
        except Exception as exc:
            skipped.append({
                "run_id": run_id,
                "tool_name": tool_name,
                "reason": f"writer raised {type(exc).__name__}: {exc}",
            })
            continue

        if not text or not str(text).strip():
            skipped.append({
                "run_id": run_id,
                "tool_name": tool_name,
                "reason": "writer returned empty paragraph",
            })
            continue

        paragraphs.append({
            "run_id": run_id,
            "tool_name": tool_name,
            "title": title,
            "paragraph": str(text).strip(),
        })

    return paragraphs, skipped


def _assemble_markdown(
    paragraphs: List[Dict[str, Any]],
    software_paragraph: str,
) -> str:
    lines: List[str] = ["# Methods", ""]

    if paragraphs:
        lines.append("## Statistical Analyses")
        lines.append("")
        for entry in paragraphs:
            lines.append(entry["paragraph"])
            lines.append("")
    else:
        lines.append("## Statistical Analyses")
        lines.append("")
        lines.append("_No inferential analyses were recorded for this session._")
        lines.append("")

    lines.append("## Software")
    lines.append("")
    lines.append(software_paragraph)
    lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def _assemble_plaintext(markdown: str) -> str:
    """
    Plaintext: italics stripped, Markdown headers converted to plain headings.
    """
    text = _strip_markdown_italics(markdown)
    # Convert `# Foo` -> `Foo` and `## Foo` -> `Foo` (preserved as headings;
    # caller is expected to paste them into a Word document)
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    return text


# ============================================================
# Execute
# ============================================================

def execute_export_apa_methods(context) -> Dict[str, Any]:
    arguments = _get_arguments(context)
    write_to_disk = bool(arguments.get("write_to_disk", True))

    try:
        analysis_runs = list(getattr(context, "analysis_runs", []) or [])

        paragraphs, skipped = _collect_paragraphs(analysis_runs)
        software_paragraph = _build_software_paragraph()

        markdown = _assemble_markdown(paragraphs, software_paragraph)
        plaintext = _assemble_plaintext(markdown)

        artifacts: list[Dict[str, Any]] = []
        md_path: str | None = None
        txt_path: str | None = None

        if write_to_disk:
            out_dir = _resolve_output_dir(context, arguments)
            os.makedirs(out_dir, exist_ok=True)

            stamp = _utc_stamp()
            md_path = os.path.join(out_dir, f"methods_{stamp}.md")
            txt_path = os.path.join(out_dir, f"methods_{stamp}.txt")

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(markdown)
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(plaintext)

            artifacts = [
                {
                    "kind": "file",
                    "path": md_path,
                    "basename": os.path.basename(md_path),
                    "mime_type": "text/markdown",
                    "description": "APA Methods section (Markdown; paste into a paper).",
                },
                {
                    "kind": "file",
                    "path": txt_path,
                    "basename": os.path.basename(txt_path),
                    "mime_type": "text/plain",
                    "description": "APA Methods section (plaintext; paste into Word).",
                },
            ]

        details = {
            "n_inferential_runs_total": sum(
                1 for r in analysis_runs
                if isinstance(r, dict) and bool(r.get("is_inferential", False))
            ),
            "n_paragraphs_written": len(paragraphs),
            "n_runs_skipped": len(skipped),
            "wrote_to_disk": bool(write_to_disk),
            "markdown_path": md_path,
            "plaintext_path": txt_path,
            "paragraphs": paragraphs,
            "skipped": skipped,
            "markdown_inline": markdown,
            "plaintext_inline": plaintext,
        }

        message = (
            f"APA Methods section assembled: {len(paragraphs)} paragraph(s), "
            f"{len(skipped)} run(s) skipped."
        )
        if md_path:
            message += f" Markdown: `{md_path}`. Plaintext: `{txt_path}`."

        return {
            "status": "ok",
            "message": message,
            "recoverable": False,
            "details": details,
            "artifacts": artifacts,
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_code": "EXPORT_APA_METHODS_EXCEPTION",
            "message": "APA Methods export failed.",
            "recoverable": True,
            "details": {
                "exception_type": type(exc).__name__,
                "exception_message": str(exc),
            },
            "artifacts": [],
        }


# ============================================================
# Extractor
# ============================================================

def extract_export_apa_methods(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "APA Methods Section Export"

    n_paragraphs = payload.get("n_paragraphs_written") or 0
    n_skipped = payload.get("n_runs_skipped") or 0
    n_total = payload.get("n_inferential_runs_total") or 0

    summary = (
        f"Assembled an APA 7th-edition Methods section from {n_paragraphs} "
        f"analysis(es) ({n_skipped} skipped). The Methods text is ready for "
        f"direct inclusion in a research paper. A Software subsection records "
        f"the exact package versions used; pair it with the reproducibility "
        f"manifest for full audit coverage."
    )

    metrics = compact_dict({
        "n_inferential_runs_total": n_total,
        "n_paragraphs_written": n_paragraphs,
        "n_runs_skipped": n_skipped,
        "wrote_to_disk": payload.get("wrote_to_disk"),
        "markdown_path": payload.get("markdown_path"),
        "plaintext_path": payload.get("plaintext_path"),
    })

    tables: Dict[str, Any] = {}

    paragraphs = payload.get("paragraphs") or []
    if paragraphs:
        tables["paragraphs"] = [
            {
                "tool_name": p.get("tool_name"),
                "title": p.get("title"),
                "paragraph": p.get("paragraph"),
            }
            for p in paragraphs
        ]

    skipped = payload.get("skipped") or []
    if skipped:
        tables["skipped_runs"] = skipped

    metadata = compact_dict({
        "markdown_inline": payload.get("markdown_inline"),
        "plaintext_inline": payload.get("plaintext_inline"),
    })

    return title, summary, metrics, tables, metadata


# ============================================================
# Display config
# ============================================================

EXPORT_APA_METHODS_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "n_inferential_runs_total": "Inferential runs in session",
            "n_paragraphs_written": "Paragraphs written",
            "n_runs_skipped": "Runs skipped",
            "wrote_to_disk": "Wrote files",
            "markdown_path": "Markdown file",
            "plaintext_path": "Plaintext file",
        },
        formatters={},
        order=[
            "n_inferential_runs_total",
            "n_paragraphs_written",
            "n_runs_skipped",
            "wrote_to_disk",
            "markdown_path",
            "plaintext_path",
        ],
    ),
    tables={
        "paragraphs": TableDisplayConfig(
            column_labels={
                "tool_name": "Tool",
                "title": "Title",
                "paragraph": "Methods paragraph",
            },
            column_order=["tool_name", "title", "paragraph"],
        ),
        "skipped_runs": TableDisplayConfig(
            column_labels={
                "run_id": "Run ID",
                "tool_name": "Tool",
                "reason": "Reason skipped",
            },
            column_order=["run_id", "tool_name", "reason"],
        ),
    },
)


# ============================================================
# Plugin registration
# ============================================================

PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="export_apa_methods",
    display_name="APA Methods Section",
    is_inferential=False,
    evidence_categories=["reporting", "publication"],
    evidence_category_roles={
        "reporting": "provenance",
        "publication": "provenance",
    },
    description=(
        "Assemble an APA 7th-edition Methods section from every inferential "
        "analysis run in the session. Returns Markdown (italics rendered) and "
        "plaintext (italics stripped) versions suitable for direct inclusion "
        "in a research paper. Each paragraph reports the test, the test "
        "statistic with degrees of freedom, the p-value, and the effect size "
        "with its 95% confidence interval. A Software subsection records "
        "exact package versions."
    ),
    usage_guidance=(
        "Call this tool when the user asks to 'export', 'write', or "
        "'generate' a Methods section, APA paragraph, manuscript text, or "
        "supplementary methods. The output is the deliverable; the metric "
        "table on this run is just a summary preview. Pair the Methods "
        "section with the reproducibility manifest for full audit coverage."
    ),
    use_when=[
        "The user is writing a paper and asks for a Methods section.",
        "The user wants an APA-style description of the analyses just run.",
        "The user mentions 'manuscript', 'paper', 'thesis', or 'supplementary materials'.",
    ],
    do_not_use_when=[
        "No inferential analyses have been run yet in this session.",
        "The user wants to run a new statistical test; pick the appropriate test plugin instead.",
    ],
    requires_data_source=None,
    produces_active_dataset=False,
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={},
        optional={
            "write_to_disk": bool,
            "output_dir": str,
        },
        column_args=[],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_export_apa_methods,
    extractor=extract_export_apa_methods,
    guardrail_evaluators=[],
    display_config=EXPORT_APA_METHODS_DISPLAY,
    examples=[
        {
            "user_request": "Write the Methods section for the analyses I just did.",
            "arguments": {},
        },
        {
            "user_request": "Export an APA paragraph I can paste into my paper.",
            "arguments": {},
        },
    ],
))