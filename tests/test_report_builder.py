from core.report_builder import build_html_report


def test_html_report_handles_pipes_inside_table_cells():
    markdown_text = "\n".join(
        [
            "# Test Report",
            "",
            "| Metric | Value |",
            "|---|---|",
            r"| High \|DFFITS\| observations | 42 |",
        ]
    )

    html = build_html_report(markdown_text)

    assert '<div class="table-scroll">' in html
    assert "High |DFFITS| observations" in html
    assert "<th>Metric</th>" in html
    assert "<td>42</td>" in html