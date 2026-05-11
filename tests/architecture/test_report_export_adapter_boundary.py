from pathlib import Path


def test_report_export_adapter_does_not_import_streamlit_or_graph():
    text = Path("core/ui_adapter/report_export.py").read_text(encoding="utf-8")

    forbidden = [
        "streamlit",
        "from core.graph",
        "import core.graph",
        "execute_node",
        "verify_node",
        "summarize_node",
        "plan_only_node",
        "execute_pending_plan_node",
    ]

    offenders = [
        item
        for item in forbidden
        if item in text
    ]

    assert offenders == []


def test_report_export_adapter_uses_report_builder():
    text = Path("core/ui_adapter/report_export.py").read_text(encoding="utf-8")

    assert "build_markdown_report" in text
    assert "build_html_report_from_state" in text
    assert "build_report_package_from_state" in text