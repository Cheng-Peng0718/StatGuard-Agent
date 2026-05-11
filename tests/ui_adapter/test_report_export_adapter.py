import json

from core.ui_adapter.report_export import build_report_package_from_state


def test_build_report_package_from_empty_state_is_safe():
    state = {
        "analysis_runs": [],
        "data_versions": [],
        "data_audit_log": [],
        "active_data_version_id": None,
        "user_request": "Please analyze this dataset.",
    }

    package = build_report_package_from_state(state)

    assert package["title"] == "Data Analysis Report"
    assert isinstance(package["markdown"], str)
    assert isinstance(package["html"], str)
    assert isinstance(package["plain_text_summary"], str)

    assert "Data Analysis Report" in package["markdown"]
    assert "<html" in package["html"].lower()

    assert package["metadata"]["n_analysis_runs"] == 0
    assert package["metadata"]["has_analysis_runs"] is False

    json.dumps(package)


def test_build_report_package_includes_analysis_run_summary():
    state = {
        "user_request": "Summarize my analysis.",
        "active_data_version_id": "raw_v1",
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "workspaces/test/data_versions/raw_v1.parquet",
            }
        ],
        "data_audit_log": [],
        "analysis_runs": [
            {
                "analysis_run_id": "run_1",
                "observation_id": "obs_1",
                "tool_name": "get_summary_stats",
                "status": "ok",
                "success": True,
                "summary": "Computed summary statistics.",
                "message": "Computed summary statistics.",
                "data_version_id": "raw_v1",
                "arguments": {},
                "metrics": {
                    "n_rows": 5,
                    "n_cols": 3,
                },
                "tables": {},
                "artifacts": [],
                "guardrails": [],
                "report_blocks": [],
            }
        ],
    }

    package = build_report_package_from_state(state)

    assert package["metadata"]["n_analysis_runs"] == 1
    assert package["metadata"]["has_analysis_runs"] is True

    assert "get_summary_stats" in package["markdown"]
    assert "Computed summary statistics" in package["markdown"]
    assert "raw_v1" in package["markdown"]

    assert "get_summary_stats" in package["html"]
    assert "Computed summary statistics" in package["html"]
    assert package["metadata"]["n_data_audit_events"] == 0