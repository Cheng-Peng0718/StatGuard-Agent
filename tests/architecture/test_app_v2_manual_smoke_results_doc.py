from pathlib import Path


def test_app_v2_manual_smoke_results_doc_exists():
    path = Path("docs/19_app_v2_manual_smoke_results.md")

    assert path.exists()


def test_app_v2_manual_smoke_results_doc_mentions_verified_flows():
    text = Path("docs/19_app_v2_manual_smoke_results.md").read_text(
        encoding="utf-8"
    )

    required = [
        "get_summary_stats",
        "missingness_report",
        "get_correlation_matrix",
        "run_multiple_regression",
        "run_anova",
        "clean_data",
        "Human Review panel appears",
        "active_data_version_id changes from raw_v1",
        "execution_audit.status is ok",
        "state_serialization_audit.status is ok",
    ]

    for item in required:
        assert item in text


def test_app_v2_manual_smoke_results_doc_preserves_thin_ui_rule():
    text = Path("docs/19_app_v2_manual_smoke_results.md").read_text(
        encoding="utf-8"
    )

    required = [
        "UIEvent -> apply_ui_event_to_state -> run_backend_turn -> UISnapshot",
        "The UI must not directly implement:",
        "tool execution",
        "verification",
        "planning rules",
        "data-version logic",
        "human-review policy",
    ]

    for item in required:
        assert item in text