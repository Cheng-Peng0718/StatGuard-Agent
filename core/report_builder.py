from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    return str(x)

def _fmt_value(x: Any, digits: int = 4) -> str:
    """
    Generic value formatter.

    This function must not contain dataset-specific or method-specific logic.
    """
    if x is None:
        return "N/A"

    if isinstance(x, bool):
        return "Yes" if x else "No"

    if isinstance(x, int):
        return str(x)

    if isinstance(x, float):
        if abs(x) < 1e-10:
            return "0"
        if abs(x) < 0.0001:
            return f"{x:.2e}"
        return f"{x:.{digits}f}".rstrip("0").rstrip(".")

    if isinstance(x, list):
        return "; ".join(_fmt_value(v, digits=digits) for v in x)

    return str(x)


def _pretty_severity(severity: str) -> str:
    if severity == "critical":
        return "Critical"
    if severity == "warning":
        return "Warning"
    return "Info"


def _format_guardrail_table(findings: List[Dict[str, Any]]) -> str:
    if not findings:
        return ""

    lines = [
        "| Severity | Finding | Message | Recommendation |",
        "|---|---|---|---|",
    ]

    for f in findings:
        recommendation = f.get("recommendation") or "N/A"

        lines.append(
            f"| {_pretty_severity(f.get('severity', 'info'))} "
            f"| {f.get('title', '')} "
            f"| {f.get('message', '')} "
            f"| {recommendation} |"
        )

    return "\n".join(lines)

def _find_run(analysis_runs: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
    for run in analysis_runs or []:
        if run.get("tool_name") == tool_name:
            return run
    return None


def _md_inline_code(x: Any) -> str:
    if x is None:
        return "`None`"
    return f"`{str(x)}`"


def _format_dict_as_markdown_table(d: Dict[str, Any]) -> str:
    if not d:
        return ""

    lines = ["| Metric | Value |", "|---|---|"]
    for k, v in d.items():
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def _format_list_of_dicts_as_markdown_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""

    keys = []
    for row in rows:
        if isinstance(row, dict):
            for k in row.keys():
                if k not in keys:
                    keys.append(k)

    if not keys:
        return ""

    lines = [
        "| " + " | ".join(keys) + " |",
        "| " + " | ".join(["---"] * len(keys)) + " |",
    ]

    for row in rows:
        lines.append("| " + " | ".join(_safe_str(row.get(k, "")) for k in keys) + " |")

    return "\n".join(lines)


def _format_table_any(table_data: Any) -> str:
    """
    Convert common table-like objects into Markdown.
    """
    if isinstance(table_data, list) and all(isinstance(x, dict) for x in table_data):
        return _format_list_of_dicts_as_markdown_table(table_data)

    if isinstance(table_data, dict):
        # If dict of dicts, show as compact nested text for now.
        return "```json\n" + _pretty_json_like(table_data) + "\n```"

    return "```text\n" + _safe_str(table_data) + "\n```"


def _pretty_json_like(obj: Any) -> str:
    import json

    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)

def _format_metric_table_block(block: Dict[str, Any]) -> str:
    rows = block.get("rows", []) or []

    # Backward-compatible fallback: old blocks may still use content dict.
    if not rows and isinstance(block.get("content"), dict):
        rows = [
            {"label": _humanize_key(k), "value": v}
            for k, v in block.get("content", {}).items()
        ]

    if not rows:
        return ""

    lines = ["| Metric | Value |", "|---|---|"]

    for row in rows:
        if not isinstance(row, dict):
            continue

        label = row.get("label") or _humanize_key(row.get("raw_key", ""))
        value = row.get("value", "")

        lines.append(f"| {label} | {_fmt_value(value)} |")

    return "\n".join(lines)


def _humanize_key(key: Any) -> str:
    """
    Generic key humanizer only.

    No statistical-method-specific mappings allowed here.
    """
    if key is None:
        return ""

    text = str(key).strip()
    if not text:
        return ""

    if text.isupper():
        return text

    text = text.replace("_", " ").replace("-", " ")
    text = " ".join(text.split())

    return text[:1].upper() + text[1:]


def get_active_version(
    data_versions: List[Dict[str, Any]],
    active_data_version_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    for version in data_versions or []:
        if version.get("version_id") == active_data_version_id:
            return version
    return None


def _format_generic_table_block(block: Dict[str, Any]) -> str:
    columns = block.get("columns", []) or []
    rows = block.get("rows", []) or []

    if not columns:
        return ""

    column_labels = []
    for col in columns:
        if isinstance(col, dict):
            column_labels.append(str(col.get("label", "")))
        else:
            column_labels.append(str(col))

    lines = [
        "| " + " | ".join(column_labels) + " |",
        "| " + " | ".join(["---"] * len(column_labels)) + " |",
    ]

    for row in rows:
        if isinstance(row, list):
            values = row
        elif isinstance(row, dict):
            values = list(row.values())
        else:
            values = [row]

        values = list(values)[:len(column_labels)] + [""] * max(
            0, len(column_labels) - len(values)
        )

        lines.append("| " + " | ".join(_fmt_value(v) for v in values) + " |")

    return "\n".join(lines)


def _format_report_block_markdown(block: Dict[str, Any]) -> str:
    block_type = block.get("type")
    title = block.get("title")

    lines: List[str] = []

    if title and block_type != "figure":
        lines.append(f"### {title}")
        lines.append("")

    if block_type == "text":
        content = block.get("content", "")
        if content:
            lines.append(str(content))
            lines.append("")

    elif block_type == "metric_table":
        table = _format_metric_table_block(block)
        if table:
            lines.append(table)
            lines.append("")

    elif block_type == "table":
        table = _format_generic_table_block(block)
        if table:
            lines.append(table)
            lines.append("")

    elif block_type == "json":
        lines.append("```json")
        lines.append(_pretty_json_like(block.get("content", {})))
        lines.append("```")
        lines.append("")

    elif block_type == "figure":
        name = block.get("name") or block.get("title") or "Figure"
        lines.append(f"### {title or 'Figure'}")
        lines.append("")
        lines.append(f"- Figure artifact: `{name}`")
        lines.append("")

    elif block_type == "artifact":
        lines.append("```json")
        lines.append(_pretty_json_like(block.get("content", {})))
        lines.append("```")
        lines.append("")

    else:
        content = block.get("content")
        if content is not None:
            lines.append(str(content))
            lines.append("")

    return "\n".join(lines)

def _single_line_text(text: Any) -> str:
    """
    Convert arbitrary text into one safe Markdown paragraph/bullet line.
    """
    return " ".join(_safe_str(text).split())


def _clip_text(text: Any, max_chars: int = 420) -> str:
    text = _single_line_text(text)

    if len(text) <= max_chars:
        return text

    return text[: max_chars - 1].rstrip() + "…"


def _run_has_executive_signal(run: Dict[str, Any]) -> bool:
    """
    Decide whether a run should contribute to the executive summary.

    This remains method-agnostic:
    - no statistical method names
    - no dataset column names
    - no concrete analysis tool names
    """
    status = run.get("status")

    if status == "blocked":
        return True

    if status not in {"ok", "warning"}:
        return False

    metadata = run.get("metadata", {}) or {}
    if metadata.get("include_in_executive_summary") is True:
        return True

    if run.get("metrics"):
        return True

    if run.get("guardrails"):
        return True

    return False


def _build_executive_finding_lines(
    analysis_runs: List[Dict[str, Any]],
    *,
    limit: int = 5,
) -> List[str]:
    findings: List[str] = []

    for run in analysis_runs or []:
        if not _run_has_executive_signal(run):
            continue

        summary = _single_line_text(run.get("summary"))
        if not summary:
            continue

        title = run.get("title") or run.get("tool_name") or "Analysis result"
        status = run.get("status", "unknown")

        label = f"**{title}**"
        if status in {"blocked", "failed"}:
            label += f" ({status})"

        findings.append(f"{label}: {_clip_text(summary)}")

        if len(findings) >= limit:
            break

    return findings


def _build_executive_attention_lines(
    analysis_runs: List[Dict[str, Any]],
    *,
    limit: int = 4,
) -> List[str]:
    attention: List[str] = []

    for run in analysis_runs or []:
        status = run.get("status")

        if status in {"blocked", "failed"}:
            title = run.get("title") or run.get("tool_name") or "Analysis result"
            summary = run.get("summary") or "The requested analysis could not be completed."
            attention.append(f"**{title}**: {_clip_text(summary, max_chars=300)}")

        for finding in run.get("guardrails", []) or []:
            if finding.get("severity") not in {"warning", "critical"}:
                continue

            title = finding.get("title") or "Guardrail finding"
            message = finding.get("message") or ""
            attention.append(f"**{title}**: {_clip_text(message, max_chars=300)}")

        if len(attention) >= limit:
            break

    return attention[:limit]

def _extract_payload_from_run(run: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract canonical payload from an AnalysisRun.

    Newer runs may store payload directly. Older/generic runs may store it
    inside a report block titled 'Payload' with type='json'.
    """
    payload = run.get("payload")

    if isinstance(payload, dict):
        return payload

    for block in run.get("report_blocks", []) or []:
        if not isinstance(block, dict):
            continue

        title = str(block.get("title", "")).strip().lower()
        block_type = block.get("type")

        if block_type == "json" and title == "payload":
            content = block.get("content", {})
            if isinstance(content, dict):
                return content

    return {}

def _render_sql_schema_summary(payload: dict) -> str:
    tables = payload.get("tables", [])

    if isinstance(tables, list) and tables:
        lines = ["### SQL Schema Summary", ""]

        for table in tables:
            if not isinstance(table, dict):
                continue

            table_name = table.get("table_name")
            row_count = table.get("row_count")
            columns = table.get("columns", [])

            column_names = [
                col.get("column_name")
                for col in columns
                if isinstance(col, dict) and col.get("column_name")
            ]

            lines.append(
                f"- **{table_name}** "
                f"({row_count} rows): `{', '.join(column_names)}`"
            )

        return "\n".join(lines)

    compact_schema = payload.get("compact_schema")

    if compact_schema:
        return (
            "### SQL Schema Summary\n\n"
            f"`{compact_schema}`\n"
        )

    return ""

def _render_sql_materialization_summary(payload: dict) -> str:
    query = payload.get("query", "")
    result_name = payload.get("result_name", "sql_query_result")
    new_data_version_id = payload.get("new_data_version_id")
    n_rows = payload.get("n_rows")
    n_cols = payload.get("n_cols")
    columns = payload.get("columns", [])
    database_path = payload.get("database_path")
    parquet_path = payload.get("parquet_path")

    if not isinstance(columns, list):
        columns = []

    lines = [
        "### SQL Query Materialization",
        "",
        f"- **Result name:** `{result_name}`",
        f"- **Source database:** `{database_path}`",
        f"- **Produced data version:** `{new_data_version_id}`",
        f"- **Shape:** {n_rows} rows × {n_cols} columns",
        f"- **Columns:** `{', '.join(str(c) for c in columns)}`",
    ]

    if parquet_path:
        lines.append(f"- **Workspace parquet:** `{parquet_path}`")

    if query:
        lines.append("")
        lines.append("**SQL query used:**")
        lines.append("")
        lines.append("```sql")
        lines.append(query.strip())
        lines.append("```")

    return "\n".join(lines)

def _render_groupby_summary(payload: dict) -> str:
    group_cols = payload.get("group_cols", [])
    value_col = payload.get("value_col")
    rows = payload.get("rows", [])
    columns = payload.get("columns", [])
    n_groups = payload.get("n_groups")
    rows_used = payload.get("rows_used")
    rows_dropped = payload.get("rows_dropped_due_to_missing")

    if not isinstance(group_cols, list):
        group_cols = []

    lines = [
        "### Groupby Summary",
        "",
        f"- **Grouping columns:** `{', '.join(str(c) for c in group_cols)}`",
        f"- **Value column:** `{value_col}`",
        f"- **Groups returned:** {n_groups}",
        f"- **Rows used:** {rows_used}",
    ]

    if rows_dropped:
        lines.append(f"- **Rows dropped due to missing values:** {rows_dropped}")

    lines.append("")

    if rows and columns:
        table_block = {
            "type": "table",
            "columns": [
                {"label": _humanize_key(col), "raw_key": col}
                for col in columns
            ],
            "rows": [
                [row.get(col, "") for col in columns]
                for row in rows
                if isinstance(row, dict)
            ],
        }

        lines.append(_format_generic_table_block(table_block))
        lines.append("")

    return "\n".join(lines)

def _format_analysis_run_markdown(run: Dict[str, Any], index: int) -> str:
    title = run.get("title") or run.get("tool_name") or f"Analysis Run {index}"
    status = run.get("status", "unknown")
    data_version_id = run.get("data_version_id")
    tool_name = run.get("tool_name", "unknown")
    payload = _extract_payload_from_run(run)

    lines: List[str] = []

    lines.append(f"## {index}. {title}")
    lines.append("")

    lines.append("| Field | Value |")
    lines.append("|---|---|")
    lines.append(f"| Status | {status} |")

    if tool_name == "materialize_sql_query_result":
        produced_version = payload.get("new_data_version_id")

        lines.append(f"| Input data version | `{data_version_id or 'N/A'}` |")
        lines.append(f"| Produced data version | `{produced_version or 'N/A'}` |")

    else:
        lines.append(f"| Data version | `{data_version_id or 'N/A'}` |")

    lines.append(f"| Tool | `{tool_name}` |")
    lines.append("")

    # SQL-specific report rendering.
    # These are presentation adapters, not analysis/planning rules.
    # They prevent raw internal JSON from dominating the human-facing report.
    if tool_name == "inspect_sql_schema":
        rendered = _render_sql_schema_summary(payload)
        if rendered:
            lines.append(rendered)
            lines.append("")
        else:
            summary = run.get("summary")
            if summary:
                lines.append("### Summary")
                lines.append("")
                lines.append(str(summary))
                lines.append("")

        return "\n".join(lines)

    if tool_name == "materialize_sql_query_result":
        rendered = _render_sql_materialization_summary(payload)
        if rendered:
            lines.append(rendered)
            lines.append("")
        else:
            summary = run.get("summary")
            if summary:
                lines.append("### Summary")
                lines.append("")
                lines.append(str(summary))
                lines.append("")

        return "\n".join(lines)

    if tool_name == "groupby_summary":
        rendered = _render_groupby_summary(payload)
        if rendered:
            lines.append(rendered)
            lines.append("")
        else:
            summary = run.get("summary")
            if summary:
                lines.append("### Summary")
                lines.append("")
                lines.append(str(summary))
                lines.append("")

        return "\n".join(lines)

    blocks = run.get("report_blocks", []) or []

    if blocks:
        for block in blocks:
            rendered = _format_report_block_markdown(block)
            if rendered:
                lines.append(rendered)
    else:
        # Generic fallback for old AnalysisRun objects.
        summary = run.get("summary")
        if summary:
            lines.append("### Summary")
            lines.append("")
            lines.append(str(summary))
            lines.append("")

        metrics = run.get("metrics", {})
        if metrics:
            lines.append("### Metrics")
            lines.append("")
            fallback_block = {
                "type": "metric_table",
                "rows": [
                    {"label": _humanize_key(k), "value": v, "raw_key": k}
                    for k, v in metrics.items()
                ],
            }
            lines.append(_format_metric_table_block(fallback_block))
            lines.append("")

        tables = run.get("tables", {})
        for table_name, table_data in tables.items():
            lines.append(f"### {_humanize_key(table_name)}")
            lines.append("")

            if isinstance(table_data, list) and all(isinstance(x, dict) for x in table_data):
                raw_columns = list(table_data[0].keys()) if table_data else []
                table_block = {
                    "type": "table",
                    "columns": [{"label": _humanize_key(c), "raw_key": c} for c in raw_columns],
                    "rows": [
                        [row.get(c, "") for c in raw_columns]
                        for row in table_data
                    ],
                }
                lines.append(_format_generic_table_block(table_block))
            else:
                lines.append("```json")
                lines.append(_pretty_json_like(table_data))
                lines.append("```")

            lines.append("")

    return "\n".join(lines)


def build_markdown_report(
    *,
    user_request: str,
    active_data_version_id: Optional[str],
    data_versions: List[Dict[str, Any]],
    data_audit_log: List[Dict[str, Any]],
    analysis_runs: List[Dict[str, Any]],
    title: str = "Data Analysis Report",
) -> str:
    """
    Build a generic, analysis-agnostic Markdown report.

    This function must not special-case any statistical method, dataset column,
    or tool name.
    """
    active_version = get_active_version(data_versions, active_data_version_id)

    lines: List[str] = []

    lines.append(f"# {title}")
    lines.append("")

    if user_request:
        lines.append("## Analysis Request")
        lines.append("")
        lines.append(user_request)
        lines.append("")

    lines.append("## Executive Summary")
    lines.append("")

    if active_version:
        lines.append(
            f"This report uses data version `{active_version.get('version_id')}`, "
            f"with {active_version.get('n_rows')} rows and {active_version.get('n_cols')} columns."
        )
    else:
        lines.append(f"This report uses data version `{active_data_version_id}`.")

    lines.append("")

    completed_runs = [
        run for run in analysis_runs or []
        if run.get("status") in {"ok", "warning"}
    ]

    critical_count = 0
    warning_count = 0

    for run in analysis_runs or []:
        for finding in run.get("guardrails", []) or []:
            severity = finding.get("severity")
            if severity == "critical":
                critical_count += 1
            elif severity == "warning":
                warning_count += 1

    executive_findings = _build_executive_finding_lines(analysis_runs or [])

    lines.append("Key findings:")
    lines.append("")

    if executive_findings:
        for finding in executive_findings:
            lines.append(f"- {finding}")
    else:
        lines.append("- No substantive analysis findings were recorded yet.")

    lines.append("")

    attention_items = _build_executive_attention_lines(analysis_runs or [])

    if attention_items:
        lines.append("Needs attention:")
        lines.append("")

        for item in attention_items:
            lines.append(f"- {item}")

        lines.append("")

    lines.append("Execution summary:")
    lines.append("")
    lines.append(f"- Completed analysis runs: **{len(completed_runs)}**.")
    lines.append(
        f"- Guardrails identified **{critical_count}** critical and "
        f"**{warning_count}** warning-level issue(s)."
    )
    lines.append("")

    lines.append("## Data Provenance")
    lines.append("")

    if active_version:
        parent_version = active_version.get("parent_version_id") or "N/A"

        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| Active data version | `{active_version.get('version_id')}` |")
        lines.append(f"| Parent version | {parent_version} |")
        lines.append(f"| Rows | {active_version.get('n_rows')} |")
        lines.append(f"| Columns | {active_version.get('n_cols')} |")
        lines.append(f"| Operation | `{active_version.get('operation')}` |")
        lines.append("")
    else:
        lines.append(f"Active data version: `{active_data_version_id}`")
        lines.append("")

    if data_audit_log:
        lines.append("### Data Audit Trail")
        lines.append("")
        lines.append("| Event | Version | Parent | Description |")
        lines.append("|---|---|---|---|")

        for event in data_audit_log:
            parent = event.get("parent_version_id") or "N/A"
            lines.append(
                f"| {event.get('event_type', '')} "
                f"| `{event.get('version_id', '')}` "
                f"| {parent} "
                f"| {event.get('description', '')} |"
            )

        lines.append("")

    lines.append("## Analysis Results")
    lines.append("")

    if not analysis_runs:
        lines.append("No analysis runs recorded.")
        lines.append("")
    else:
        for index, run in enumerate(analysis_runs, start=1):
            lines.append(_format_analysis_run_markdown(run, index))

    all_guardrails = []
    for run in analysis_runs or []:
        for finding in run.get("guardrails", []) or []:
            finding_with_source = dict(finding)
            finding_with_source["source_run_title"] = run.get("title", run.get("tool_name"))
            all_guardrails.append(finding_with_source)

    if all_guardrails:
        lines.append("## Statistical Guardrails")
        lines.append("")

        warning_or_critical = [
            finding for finding in all_guardrails
            if finding.get("severity") in {"warning", "critical"}
        ]

        info_findings = [
            finding for finding in all_guardrails
            if finding.get("severity") not in {"warning", "critical"}
        ]

        if warning_or_critical:
            lines.append("### Warnings")
            lines.append("")
            lines.append(_format_guardrail_table(warning_or_critical))
            lines.append("")

        if info_findings:
            lines.append("### Informational Checks")
            lines.append("")
            lines.append(_format_guardrail_table(info_findings))
            lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Results are tied to the data version shown above.")
    lines.append("- If the active data version changes, statistics and model results should be recomputed.")
    lines.append("- Guardrail findings are screening signals and should be interpreted with domain knowledge.")
    lines.append("")

    return "\n".join(lines)

def _image_file_to_base64_data_uri(path: str) -> Optional[str]:
    """
    Convert a local PNG/JPEG image into a base64 data URI for self-contained HTML.
    """
    import base64
    import os

    if not path or not os.path.exists(path):
        return None

    ext = os.path.splitext(path)[1].lower()

    if ext == ".png":
        mime = "image/png"
    elif ext in {".jpg", ".jpeg"}:
        mime = "image/jpeg"
    else:
        return None

    try:
        with open(path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    except Exception:
        return None


def build_html_report(markdown_text: str, title: str = "Data Analysis Report") -> str:
    """
    Render the Markdown report into a clean standalone HTML document.

    This is a lightweight Markdown renderer for the report structure generated
    by build_markdown_report(). It intentionally supports the subset we use:
    headings, paragraphs, bullet lists, fenced code blocks, and Markdown tables.
    """
    import html
    import re

    def render_inline(text: str) -> str:
        text = html.escape(text)

        # Inline code
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)

        # Bold
        text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)

        return text

    def render_markdown_table(lines):
        if len(lines) < 2:
            return ""

        header = [cell.strip() for cell in lines[0].strip().strip("|").split("|")]
        rows = []

        for line in lines[2:]:
            cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
            rows.append(cells)

        html_lines = ["<table>"]
        html_lines.append("<thead><tr>")
        for h in header:
            html_lines.append(f"<th>{render_inline(h)}</th>")
        html_lines.append("</tr></thead>")

        html_lines.append("<tbody>")
        for row in rows:
            html_lines.append("<tr>")
            for cell in row:
                html_lines.append(f"<td>{render_inline(cell)}</td>")
            html_lines.append("</tr>")
        html_lines.append("</tbody>")
        html_lines.append("</table>")

        return "\n".join(html_lines)

    def is_table_start(lines, i):
        if i + 1 >= len(lines):
            return False
        line = lines[i].strip()
        next_line = lines[i + 1].strip()
        return (
            line.startswith("|")
            and line.endswith("|")
            and next_line.startswith("|")
            and "---" in next_line
        )

    def render_body(md: str) -> str:
        lines = md.splitlines()
        out = []
        i = 0
        in_code = False
        code_buffer = []
        list_open = False

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Fenced code block
            if stripped.startswith("```"):
                if not in_code:
                    in_code = True
                    code_buffer = []
                else:
                    code_text = "\n".join(code_buffer)
                    out.append(f"<pre><code>{html.escape(code_text)}</code></pre>")
                    in_code = False
                i += 1
                continue

            if in_code:
                code_buffer.append(line)
                i += 1
                continue

            # Tables
            if is_table_start(lines, i):
                if list_open:
                    out.append("</ul>")
                    list_open = False

                table_lines = []
                while i < len(lines) and lines[i].strip().startswith("|"):
                    table_lines.append(lines[i])
                    i += 1

                out.append(render_markdown_table(table_lines))
                continue

            # Empty line
            if stripped == "":
                if list_open:
                    out.append("</ul>")
                    list_open = False
                i += 1
                continue

            # Headings
            if stripped.startswith("# "):
                if list_open:
                    out.append("</ul>")
                    list_open = False
                out.append(f"<h1>{render_inline(stripped[2:].strip())}</h1>")
                i += 1
                continue

            if stripped.startswith("## "):
                if list_open:
                    out.append("</ul>")
                    list_open = False
                out.append(f"<h2>{render_inline(stripped[3:].strip())}</h2>")
                i += 1
                continue

            if stripped.startswith("### "):
                if list_open:
                    out.append("</ul>")
                    list_open = False
                out.append(f"<h3>{render_inline(stripped[4:].strip())}</h3>")
                i += 1
                continue

            if stripped.startswith("#### "):
                if list_open:
                    out.append("</ul>")
                    list_open = False
                out.append(f"<h4>{render_inline(stripped[5:].strip())}</h4>")
                i += 1
                continue

            # Bullet list
            if stripped.startswith("- "):
                if not list_open:
                    out.append("<ul>")
                    list_open = True
                out.append(f"<li>{render_inline(stripped[2:].strip())}</li>")
                i += 1
                continue

            # Paragraph
            if list_open:
                out.append("</ul>")
                list_open = False

            out.append(f"<p>{render_inline(stripped)}</p>")
            i += 1

        if list_open:
            out.append("</ul>")

        return "\n".join(out)

    body_html = render_body(markdown_text)

    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>{html.escape(title)}</title>
<style>
:root {{
    --bg: #f6f7fb;
    --paper: #ffffff;
    --text: #111827;
    --muted: #6b7280;
    --border: #e5e7eb;
    --accent: #2563eb;
    --accent-soft: #eff6ff;
    --code-bg: #f3f4f6;
}}

* {{
    box-sizing: border-box;
}}

body {{
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Arial, sans-serif;
    line-height: 1.6;
}}

.report-container {{
    max-width: 980px;
    margin: 40px auto;
    padding: 0 24px;
}}

.report-paper {{
    background: var(--paper);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 44px;
    box-shadow: 0 12px 32px rgba(15, 23, 42, 0.08);
}}

h1 {{
    font-size: 34px;
    line-height: 1.2;
    margin: 0 0 28px;
    letter-spacing: -0.03em;
}}

h2 {{
    font-size: 24px;
    margin-top: 42px;
    margin-bottom: 14px;
    padding-bottom: 8px;
    border-bottom: 1px solid var(--border);
    letter-spacing: -0.02em;
}}

h3 {{
    font-size: 19px;
    margin-top: 28px;
    margin-bottom: 12px;
}}

h4 {{
    font-size: 16px;
    margin-top: 18px;
    margin-bottom: 8px;
}}

p {{
    margin: 8px 0 14px;
}}

ul {{
    margin: 10px 0 18px 22px;
    padding: 0;
}}

li {{
    margin: 6px 0;
}}

table {{
    width: 100%;
    border-collapse: collapse;
    margin: 14px 0 24px;
    font-size: 14px;
    overflow: hidden;
    border-radius: 10px;
}}

thead {{
    background: var(--accent-soft);
}}

th, td {{
    border: 1px solid var(--border);
    padding: 9px 11px;
    text-align: left;
    vertical-align: top;
}}

th {{
    color: #1e3a8a;
    font-weight: 700;
}}

code {{
    background: var(--code-bg);
    padding: 2px 5px;
    border-radius: 5px;
    font-size: 0.92em;
}}

pre {{
    background: #0f172a;
    color: #e5e7eb;
    padding: 16px;
    border-radius: 12px;
    overflow-x: auto;
    font-size: 13px;
}}

pre code {{
    background: transparent;
    color: inherit;
    padding: 0;
    border-radius: 0;
    font-size: inherit;
    display: block;
    white-space: pre;
}}

strong {{
    font-weight: 700;
}}

.figure-block {{
    margin: 18px 0 34px;
}}

.figure-block h4 {{
    color: var(--muted);
    font-weight: 600;
    margin-bottom: 10px;
}}

.report-image {{
    width: 100%;
    max-width: 820px;
    display: block;
    margin: 16px auto 32px;
    border: 1px solid var(--border);
    border-radius: 12px;
    box-shadow: 0 8px 22px rgba(15, 23, 42, 0.08);
}}

@media print {{
    body {{
        background: white;
    }}

    .report-container {{
        margin: 0;
        max-width: none;
        padding: 0;
    }}

    .report-paper {{
        box-shadow: none;
        border: none;
        border-radius: 0;
    }}
}}
</style>
</head>
<body>
<div class="report-container">
  <article class="report-paper">
    {body_html}
  </article>
</div>
</body>
</html>
"""

def build_html_report_from_state(
    *,
    user_request: str,
    active_data_version_id: Optional[str],
    data_versions: List[Dict[str, Any]],
    data_audit_log: List[Dict[str, Any]],
    analysis_runs: List[Dict[str, Any]],
    title: str = "Data Analysis Report",
) -> str:
    """
    Build a standalone HTML report with embedded image artifacts.

    This function embeds all figure report blocks generically.
    """
    markdown_text = build_markdown_report(
        user_request=user_request,
        active_data_version_id=active_data_version_id,
        data_versions=data_versions,
        data_audit_log=data_audit_log,
        analysis_runs=analysis_runs,
        title=title,
    )

    html_report = build_html_report(markdown_text=markdown_text, title=title)

    figure_blocks = []

    for run in analysis_runs or []:
        run_title = run.get("title") or run.get("tool_name") or "Analysis Run"

        for block in run.get("report_blocks", []) or []:
            if block.get("type") != "figure":
                continue

            path = block.get("path")
            name = block.get("name") or block.get("title") or "figure"
            data_uri = _image_file_to_base64_data_uri(path)

            if not data_uri:
                continue

            figure_blocks.append(f"""
<h3>{html.escape(str(run_title))}: {html.escape(str(block.get("title") or "Figure"))}</h3>
<div class="figure-block">
  <h4>{html.escape(str(name))}</h4>
  <img class="report-image" src="{data_uri}" alt="{html.escape(str(name))}">
</div>
""")

    if figure_blocks:
        figures_html = "\n".join(figure_blocks)

        marker = "<h2>Statistical Guardrails</h2>"
        if marker in html_report:
            html_report = html_report.replace(
                marker,
                f"<h2>Figures</h2>\n{figures_html}\n{marker}",
                1,
            )
        else:
            marker = "<h2>Notes</h2>"
            if marker in html_report:
                html_report = html_report.replace(
                    marker,
                    f"<h2>Figures</h2>\n{figures_html}\n{marker}",
                    1,
                )
            else:
                html_report = html_report.replace(
                    "  </article>",
                    f"<h2>Figures</h2>\n{figures_html}\n  </article>",
                )

    return html_report