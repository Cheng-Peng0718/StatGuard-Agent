from core.app_backend.snapshot import build_ui_snapshot


def test_build_ui_snapshot_exposes_stable_top_level_sections():
    state = {
        "assistant_response": {
            "response_type": "dataset_overview",
            "content": "Dataset loaded.",
        },
        "dataset_name": "student_data",
        "active_data_version_id": "raw_v1",
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "workspaces/test/data_versions/raw_v1.parquet",
            }
        ],
        "dataset_profile_v2": {
            "dataset_name": "student_data",
            "data_version_id": "raw_v1",
            "columns": {
                "GPA": {
                    "semantic_type": "continuous_numeric",
                }
            },
        },
        "dataset_summary": {
            "n_rows": 3,
            "n_cols": 1,
        },
        "pending_plan": {
            "plan_id": "plan_1",
            "status": "verified",
            "steps": [],
        },
        "plan_status": "verified",
        "observations": [
            {
                "observation_id": "obs_1",
                "tool_name": "get_summary_stats",
            }
        ],
        "analysis_runs": [
            {
                "run_id": "run_1",
                "tool_name": "get_summary_stats",
            }
        ],
    }

    snapshot = build_ui_snapshot(state)

    assert snapshot["schema_version"] == "ui_snapshot_v2"
    assert set(snapshot.keys()) == {
        "schema_version",
        "assistant_response",
        "dataset",
        "plan",
        "analysis",
        "review",
        "metadata",
    }

    assert snapshot["dataset"]["dataset_name"] == "student_data"
    assert snapshot["dataset"]["active_data_version_id"] == "raw_v1"
    assert snapshot["plan"]["plan_id"] == "plan_1"
    assert snapshot["plan"]["plan_status"] == "verified"
    assert snapshot["analysis"]["observations"][0]["observation_id"] == "obs_1"


def test_build_ui_snapshot_tolerates_missing_sections():
    snapshot = build_ui_snapshot({})

    assert snapshot["schema_version"] == "ui_snapshot_v2"
    assert snapshot["assistant_response"] == {}
    assert snapshot["dataset"]["data_versions"] == []
    assert snapshot["plan"]["pending_plan"] is None
    assert snapshot["analysis"]["observations"] == []
    assert snapshot["review"]["human_review_required"] is False