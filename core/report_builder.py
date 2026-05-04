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
    Pretty formatting for report values.
    """
    if x is None:
        return ""

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
        return ", ".join(str(v) for v in x)

    return str(x)


def _pretty_metric_name(name: str) -> str:
    mapping = {
        "nobs": "Observations used",
        "r_squared": "R-squared",
        "adj_r_squared": "Adjusted R-squared",
        "f_statistic": "F statistic",
        "f_p_value": "Model p-value",
        "max_vif": "Maximum VIF",
        "breusch_pagan_lm_p_value": "Breusch-Pagan p-value",
        "heteroscedasticity_flag_0_05": "Heteroscedasticity flag",
        "n_residuals": "Residual count",
        "residual_mean": "Residual mean",
        "residual_std": "Residual SD",
        "residual_skewness": "Residual skewness",
        "residual_kurtosis_fisher": "Residual kurtosis",
        "outliers_abs_2sd": "Residuals beyond 2 SD",
        "outliers_abs_3sd": "Residuals beyond 3 SD",
        "diagnostic_flags": "Diagnostic flags",
        "GPA mean": "GPA mean",
    }
    return mapping.get(name, name.replace("_", " ").title())


def _format_selected_metrics(metrics: Dict[str, Any], keys: List[str]) -> str:
    rows = []
    for key in keys:
        if key in metrics:
            rows.append(f"| {_pretty_metric_name(key)} | {_fmt_value(metrics[key])} |")

    if not rows:
        return ""

    return "\n".join(["| Metric | Value |", "|---|---|"] + rows)


def _format_coef_table(rows: List[Dict[str, Any]]) -> str:
    if not rows:
        return ""

    lines = [
        "| Term | Estimate | Std. Error | t | p-value | 95% CI |",
        "|---|---:|---:|---:|---:|---|",
    ]

    for row in rows:
        ci = f"[{_fmt_value(row.get('ci_lower'))}, {_fmt_value(row.get('ci_upper'))}]"
        lines.append(
            f"| {row.get('term', '')} "
            f"| {_fmt_value(row.get('coef'))} "
            f"| {_fmt_value(row.get('std_err'))} "
            f"| {_fmt_value(row.get('t'))} "
            f"| {_fmt_value(row.get('p_value'))} "
            f"| {ci} |"
        )

    return "\n".join(lines)


def _find_run(analysis_runs: List[Dict[str, Any]], tool_name: str) -> Optional[Dict[str, Any]]:
    for run in analysis_runs or []:
        if run.get("tool_name") == tool_name:
            return run
    return None


def _build_executive_summary(analysis_runs: List[Dict[str, Any]], active_version: Optional[Dict[str, Any]]) -> str:
    reg = _find_run(analysis_runs, "run_multiple_regression")
    diag = _find_run(analysis_runs, "regression_diagnostics")
    hist = _find_run(analysis_runs, "generate_residual_histogram")
    stats = _find_run(analysis_runs, "get_summary_stats")

    lines = ["## Executive Summary", ""]

    if active_version:
        lines.append(
            f"This report uses data version `{active_version.get('version_id')}`, "
            f"with {active_version.get('n_rows')} rows and {active_version.get('n_cols')} columns."
        )
        lines.append("")

    findings = []

    if stats:
        mean_gpa = (stats.get("metrics") or {}).get("GPA mean")
        if mean_gpa is not None:
            findings.append(f"The mean GPA is **{_fmt_value(mean_gpa, 6)}**.")

    if reg:
        metrics = reg.get("metrics", {})
        r2 = metrics.get("r_squared")
        p = metrics.get("f_p_value")
        if r2 is not None:
            findings.append(f"The OLS model explains about **{_fmt_value(100 * float(r2), 2)}%** of GPA variation.")
        if p is not None:
            findings.append(f"The overall regression model is statistically significant with p-value **{_fmt_value(p)}**.")

    if diag:
        metrics = diag.get("metrics", {})
        max_vif = metrics.get("max_vif")
        bp_p = metrics.get("breusch_pagan_lm_p_value")
        if max_vif is not None:
            findings.append(f"The maximum VIF is **{_fmt_value(max_vif)}**, indicating no apparent multicollinearity issue in this model.")
        if bp_p is not None:
            findings.append(f"The Breusch-Pagan p-value is **{_fmt_value(bp_p)}**, so there is no strong evidence of heteroscedasticity.")

    if hist:
        flags = (hist.get("metrics") or {}).get("diagnostic_flags")
        if flags:
            findings.append(f"Residual diagnostics flagged: **{_fmt_value(flags)}**.")

    if findings:
        lines.append("Key findings:")
        lines.append("")
        for f in findings:
            lines.append(f"- {f}")
        lines.append("")
    else:
        lines.append("The report summarizes the completed analysis runs and their outputs.")
        lines.append("")

    return "\n".join(lines)


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


def _pretty_json_like(obj: Any, indent: int = 0) -> str:
    """
    Lightweight JSON-like formatting without importing json everywhere.
    """
    import json
    try:
        return json.dumps(obj, indent=2, ensure_ascii=False)
    except Exception:
        return str(obj)


def get_active_version(
    data_versions: List[Dict[str, Any]],
    active_data_version_id: Optional[str],
) -> Optional[Dict[str, Any]]:
    for v in data_versions or []:
        if v.get("version_id") == active_data_version_id:
            return v
    return None


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
    Build a clean, reader-facing Markdown report.

    The report intentionally avoids dumping internal/debug fields.
    """
    active_version = get_active_version(data_versions, active_data_version_id)

    def fmt_na(x: Any) -> str:
        if x is None or str(x).strip() == "" or str(x) == "None":
            return "N/A"
        return str(x)

    def fmt_p_value(x: Any) -> str:
        try:
            v = float(x)
            if v < 0.0001:
                return "<0.0001"
            return f"{v:.4f}".rstrip("0").rstrip(".")
        except Exception:
            return fmt_na(x)

    def fmt_model_metric_value(key: str, value: Any) -> str:
        if "p_value" in key or key.endswith("_p") or key == "p_value":
            return fmt_p_value(value)
        return _fmt_value(value)

    def pretty_metric_name_local(name: str) -> str:
        mapping = {
            "nobs": "Observations used",
            "r_squared": "R-squared",
            "adj_r_squared": "Adjusted R-squared",
            "f_statistic": "F statistic",
            "f_p_value": "Model p-value",
            "max_vif": "Maximum VIF",
            "breusch_pagan_lm_p_value": "Breusch-Pagan p-value",
            "breusch_pagan_f_p_value": "Breusch-Pagan F-test p-value",
            "heteroscedasticity_flag_0_05": "Heteroscedasticity flag",
            "n_residuals": "Residual count",
            "residual_mean": "Residual mean",
            "residual_std": "Residual SD",
            "residual_skewness": "Residual skewness",
            "residual_kurtosis_fisher": "Residual kurtosis",
            "outliers_abs_2sd": "Residuals beyond 2 SD",
            "outliers_abs_3sd": "Residuals beyond 3 SD",
            "diagnostic_flags": "Diagnostic flags",
            "GPA mean": "GPA mean",
        }
        return mapping.get(name, name.replace("_", " ").title())

    def format_selected_metrics_local(metrics: Dict[str, Any], keys: List[str]) -> str:
        rows = []
        for key in keys:
            if key in metrics:
                rows.append(
                    f"| {pretty_metric_name_local(key)} | {fmt_model_metric_value(key, metrics[key])} |"
                )

        if not rows:
            return ""

        return "\n".join(["| Metric | Value |", "|---|---|"] + rows)

    def format_coef_table_local(rows: List[Dict[str, Any]]) -> str:
        if not rows:
            return ""

        lines = [
            "| Term | Estimate | Std. Error | t | p-value | 95% CI |",
            "|---|---:|---:|---:|---:|---|",
        ]

        for row in rows:
            term = row.get("term", "")
            if term == "const":
                term = "Intercept"

            ci = f"[{_fmt_value(row.get('ci_lower'))}, {_fmt_value(row.get('ci_upper'))}]"

            lines.append(
                f"| {term} "
                f"| {_fmt_value(row.get('coef'))} "
                f"| {_fmt_value(row.get('std_err'))} "
                f"| {_fmt_value(row.get('t'))} "
                f"| {fmt_p_value(row.get('p_value'))} "
                f"| {ci} |"
            )

        return "\n".join(lines)

    def build_executive_summary_local(
        analysis_runs: List[Dict[str, Any]],
        active_version: Optional[Dict[str, Any]],
    ) -> str:
        reg = _find_run(analysis_runs, "run_multiple_regression")
        diag = _find_run(analysis_runs, "regression_diagnostics")
        hist = _find_run(analysis_runs, "generate_residual_histogram")
        stats = _find_run(analysis_runs, "get_summary_stats")

        lines = ["## Executive Summary", ""]

        if active_version:
            lines.append(
                f"This report uses data version `{active_version.get('version_id')}`, "
                f"with {active_version.get('n_rows')} rows and {active_version.get('n_cols')} columns."
            )
            lines.append("")

        findings = []

        if stats:
            mean_gpa = (stats.get("metrics") or {}).get("GPA mean")
            if mean_gpa is not None:
                findings.append(f"The mean GPA is **{_fmt_value(mean_gpa, 6)}**.")

        if reg:
            metrics = reg.get("metrics", {})
            r2 = metrics.get("r_squared")
            p = metrics.get("f_p_value")

            if r2 is not None:
                findings.append(
                    f"The OLS model explains about **{_fmt_value(100 * float(r2), 2)}%** "
                    "of GPA variation."
                )

            if p is not None:
                findings.append(
                    f"The overall regression model is statistically significant "
                    f"with p-value **{fmt_p_value(p)}**."
                )

        if diag:
            metrics = diag.get("metrics", {})
            max_vif = metrics.get("max_vif")
            bp_p = metrics.get("breusch_pagan_lm_p_value")

            if max_vif is not None:
                findings.append(
                    f"The maximum VIF is **{_fmt_value(max_vif)}**, indicating no apparent "
                    "multicollinearity issue in this model."
                )

            if bp_p is not None:
                findings.append(
                    f"The Breusch-Pagan p-value is **{fmt_p_value(bp_p)}**, so there is "
                    "no strong evidence of heteroscedasticity."
                )

        if hist:
            flags = (hist.get("metrics") or {}).get("diagnostic_flags")
            if flags:
                findings.append(f"Residual diagnostics flagged: **{_fmt_value(flags)}**.")

        if findings:
            lines.append("Key findings:")
            lines.append("")
            for item in findings:
                lines.append(f"- {item}")
            lines.append("")
        else:
            lines.append("The report summarizes the completed analysis runs and their outputs.")
            lines.append("")

        return "\n".join(lines)

    lines: List[str] = []

    lines.append(f"# {title}")
    lines.append("")

    if user_request:
        lines.append("## Analysis Request")
        lines.append("")
        lines.append(user_request)
        lines.append("")

    lines.append(build_executive_summary_local(analysis_runs, active_version))

    lines.append("## Data Provenance")
    lines.append("")

    if active_version:
        parent_version = fmt_na(active_version.get("parent_version_id"))

        lines.append("| Field | Value |")
        lines.append("|---|---|")
        lines.append(f"| Active data version | `{active_version.get('version_id')}` |")
        lines.append(f"| Parent version | {parent_version} |")
        lines.append(f"| Rows | {active_version.get('n_rows')} |")
        lines.append(f"| Columns | {active_version.get('n_cols')} |")
        lines.append(f"| Operation | `{active_version.get('operation')}` |")
        lines.append("")
    else:
        lines.append(f"Active data version: `{fmt_na(active_data_version_id)}`")
        lines.append("")

    if data_audit_log:
        lines.append("### Data Audit Trail")
        lines.append("")
        lines.append("| Event | Version | Parent | Description |")
        lines.append("|---|---|---|---|")

        for event in data_audit_log:
            lines.append(
                f"| {fmt_na(event.get('event_type'))} "
                f"| `{fmt_na(event.get('version_id'))}` "
                f"| {fmt_na(event.get('parent_version_id'))} "
                f"| {fmt_na(event.get('description'))} |"
            )
        lines.append("")

    # ------------------------------------------------------------
    # Summary Statistics
    # ------------------------------------------------------------
    stats = _find_run(analysis_runs, "get_summary_stats")
    if stats:
        lines.append("## Summary Statistics")
        lines.append("")

        metrics = stats.get("metrics", {})
        if metrics:
            lines.append(format_selected_metrics_local(metrics, list(metrics.keys())))
            lines.append("")

    # ------------------------------------------------------------
    # Regression Results
    # ------------------------------------------------------------
    reg = _find_run(analysis_runs, "run_multiple_regression")
    if reg:
        lines.append("## Regression Results")
        lines.append("")
        lines.append(reg.get("summary", ""))
        lines.append("")

        metrics = reg.get("metrics", {})
        metric_keys = [
            "nobs",
            "r_squared",
            "adj_r_squared",
            "f_statistic",
            "f_p_value",
        ]

        table = format_selected_metrics_local(metrics, metric_keys)
        if table:
            lines.append(table)
            lines.append("")

        coef_table = (reg.get("tables") or {}).get("coef_table", [])
        if coef_table:
            lines.append("### Coefficients")
            lines.append("")
            lines.append(format_coef_table_local(coef_table))
            lines.append("")

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    diag = _find_run(analysis_runs, "regression_diagnostics")
    hist = _find_run(analysis_runs, "generate_residual_histogram")

    if diag or hist:
        lines.append("## Diagnostics")
        lines.append("")

    if diag:
        lines.append("### Multicollinearity and Heteroscedasticity")
        lines.append("")
        lines.append(diag.get("summary", ""))
        lines.append("")

        metrics = diag.get("metrics", {})
        metric_keys = [
            "max_vif",
            "breusch_pagan_lm_p_value",
            "breusch_pagan_f_p_value",
            "heteroscedasticity_flag_0_05",
        ]

        table = format_selected_metrics_local(metrics, metric_keys)
        if table:
            lines.append(table)
            lines.append("")

        vif_table = (diag.get("tables") or {}).get("vif", [])
        if vif_table:
            lines.append("**VIF table**")
            lines.append("")
            lines.append(_format_list_of_dicts_as_markdown_table(vif_table))
            lines.append("")

    if hist:
        lines.append("### Residual Diagnostics")
        lines.append("")
        lines.append(hist.get("summary", ""))
        lines.append("")

        metrics = hist.get("metrics", {})
        metric_keys = [
            "n_residuals",
            "residual_mean",
            "residual_std",
            "residual_skewness",
            "residual_kurtosis_fisher",
            "outliers_abs_2sd",
            "outliers_abs_3sd",
            "diagnostic_flags",
        ]

        table = format_selected_metrics_local(metrics, metric_keys)
        if table:
            lines.append(table)
            lines.append("")

    # ------------------------------------------------------------
    # Other analysis runs
    # ------------------------------------------------------------
    handled_tools = {
        "get_summary_stats",
        "run_multiple_regression",
        "regression_diagnostics",
        "generate_residual_histogram",
    }

    other_runs = [
        run for run in analysis_runs
        if run.get("tool_name") not in handled_tools
    ]

    if other_runs:
        lines.append("## Additional Analysis Runs")
        lines.append("")

        for run in other_runs:
            lines.append(f"### {run.get('title', run.get('tool_name'))}")
            lines.append("")

            if run.get("summary"):
                lines.append(run.get("summary"))
                lines.append("")

            metrics = run.get("metrics", {})
            if metrics:
                lines.append(format_selected_metrics_local(metrics, list(metrics.keys())))
                lines.append("")

    lines.append("## Notes")
    lines.append("")
    lines.append("- Results are tied to the data version shown above.")
    lines.append("- If the active data version changes, statistics and model results should be recomputed.")
    lines.append(
        "- Diagnostic flags should be interpreted as screening signals, "
        "not definitive proof of assumption violations."
    )
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

    # Append embedded images before Notes section area if image artifacts exist.
    # Build embedded image blocks from artifacts.
    residual_figure_blocks = []
    other_figure_blocks = []

    for run in analysis_runs or []:
        tool_name = run.get("tool_name")
        artifacts = run.get("artifacts", []) or []

        for artifact in artifacts:
            if artifact.get("type") != "png":
                continue

            path = artifact.get("path")
            name = artifact.get("name") or path or "image artifact"
            data_uri = _image_file_to_base64_data_uri(path)

            if not data_uri:
                continue

            figure_html = f"""
    <div class="figure-block">
      <h4>{html.escape(name)}</h4>
      <img class="report-image" src="{data_uri}" alt="{html.escape(name)}">
    </div>
    """

            if tool_name == "generate_residual_histogram":
                residual_figure_blocks.append(figure_html)
            else:
                other_figure_blocks.append(f"""
    <h2>Embedded Figure</h2>
    {figure_html}
    """)

    # Insert residual histogram near Residual Diagnostics.
    if residual_figure_blocks:
        residual_html = "\n".join(residual_figure_blocks)

        marker = "<h2>Notes</h2>"
        if marker in html_report:
            html_report = html_report.replace(
                marker,
                f"<h3>Residual Histogram</h3>\n{residual_html}\n{marker}",
                1,
            )
        else:
            html_report = html_report.replace(
                "  </article>",
                f"<h2>Residual Histogram</h2>\n{residual_html}\n  </article>",
            )

    # Append other figures near the end.
    if other_figure_blocks:
        other_html = "\n".join(other_figure_blocks)
        html_report = html_report.replace(
            "  </article>",
            f"{other_html}\n  </article>",
        )

    return html_report