from pathlib import Path


def test_app_v3_report_export_contract_doc_exists():
    path = Path("docs/22_app_v3_report_export_contract.md")

    assert path.exists()


def test_app_v3_report_export_contract_mentions_boundaries():
    text = Path("docs/22_app_v3_report_export_contract.md").read_text(
        encoding="utf-8"
    )

    required = [
        "build_report_package_from_state",
        "build_markdown_report",
        "build_html_report_from_state",
        "App V3 should not call these directly",
        "Download Markdown",
        "Download HTML",
        "construct markdown manually",
        "construct HTML manually",
    ]

    for item in required:
        assert item in text


def test_app_v3_report_export_contract_mentions_diagnostics():
    text = Path("docs/22_app_v3_report_export_contract.md").read_text(
        encoding="utf-8"
    )

    required = [
        "regression_diagnostics",
        "residual_histogram",
        "diagnostic guardrails",
        "residual guardrails",
    ]

    for item in required:
        assert item in text