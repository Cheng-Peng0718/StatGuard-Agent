import pandas as pd

from core.workflow.nodes.context import build_context_node


def test_build_context_node_creates_profiles_summary_and_capability_map(tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    data_path = workspace / "working_data.parquet"

    df = pd.DataFrame({
        "GPA": [3.0, 3.5, None, 4.0],
        "SATM": [600, 650, 700, 720],
        "Sex": ["F", "M", "F", "M"],
    })
    df.to_parquet(data_path)

    state = {
        "workspace_dir": str(workspace),
        "current_step": 0,
        "max_steps": 5,
        "user_request": "What does the data look like?",
        "observations": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": str(data_path),
            }
        ],
        "active_data_version_id": "raw_v1",
        "data_audit_log": [],
        "dataset_name": "student_data",
    }

    updates = build_context_node(state)

    assert updates["current_step"] == 1
    assert updates["current_context_text"]

    assert "dataset_profile" in updates
    assert "dataset_profile_v2" in updates
    assert "dataset_summary" in updates
    assert "capability_map" in updates

    assert updates["dataset_profile_v2"]["dataset_name"] == "student_data"
    assert updates["dataset_profile_v2"]["data_version_id"] == "raw_v1"

    summary = updates["dataset_summary"]
    assert summary["n_rows"] == 4
    assert summary["n_cols"] == 3
    assert "GPA" in summary["numeric_columns"]

    capability_map = updates["capability_map"]
    assert "capabilities" in capability_map