from core.report_builder import build_markdown_report


def _section_between(markdown: str, start_heading: str, next_heading: str) -> str:
    start = markdown.index(start_heading)
    end = markdown.index(next_heading, start)
    return markdown[start:end]


def test_executive_summary_uses_analysis_run_findings():
    markdown = build_markdown_report(
        user_request="Run a model.",
        active_data_version_id="data_v_test",
        data_versions=[
            {
                "version_id": "data_v_test",
                "parent_version_id": None,
                "n_rows": 98,
                "n_cols": 4,
                "operation": "test",
            }
        ],
        data_audit_log=[],
        analysis_runs=[
            {
                "tool_name": "materialize_data",
                "title": "Materialize Data",
                "status": "ok",
                "summary": "Materialized an analysis-ready dataset.",
                "metrics": {},
                "guardrails": [],
                "report_blocks": [],
            },
            {
                "tool_name": "model_tool",
                "title": "Model Result",
                "status": "ok",
                "summary": (
                    "Fitted a model. R-squared was 0.61. "
                    "The overall model was statistically significant. "
                    "One predictor was statistically significant."
                ),
                "metrics": {
                    "r_squared": 0.61,
                    "model_significant": True,
                },
                "guardrails": [],
                "report_blocks": [],
            },
        ],
    )

    executive = _section_between(
        markdown,
        "## Executive Summary",
        "## Data Provenance",
    )

    assert "Key findings:" in executive
    assert "Model Result" in executive
    assert "R-squared was 0.61" in executive
    assert "Execution summary:" in executive
    assert "Completed analysis runs" in executive

    # Generic data-preparation runs without metrics should not dominate
    # the executive findings.
    key_findings_part = executive.split("Execution summary:")[0]
    assert "Materialized an analysis-ready dataset" not in key_findings_part


def test_executive_summary_surfaces_blocked_runs():
    markdown = build_markdown_report(
        user_request="Run an analysis.",
        active_data_version_id="data_v_test",
        data_versions=[
            {
                "version_id": "data_v_test",
                "parent_version_id": None,
                "n_rows": 4,
                "n_cols": 2,
                "operation": "test",
            }
        ],
        data_audit_log=[],
        analysis_runs=[
            {
                "tool_name": "analysis_tool",
                "title": "Analysis Attempt",
                "status": "blocked",
                "summary": "The requested analysis was blocked because the data were insufficient.",
                "metrics": {},
                "guardrails": [],
                "report_blocks": [],
            },
        ],
    )

    executive = _section_between(
        markdown,
        "## Executive Summary",
        "## Data Provenance",
    )

    assert "Key findings:" in executive
    assert "Needs attention:" in executive
    assert "Analysis Attempt" in executive
    assert "blocked because the data were insufficient" in executive