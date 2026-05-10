import pandas as pd

from core.app_backend.dataset_upload import (
    initialize_dataset_session_from_file,
    read_uploaded_dataframe,
)


def test_read_uploaded_dataframe_supports_csv(tmp_path):
    path = tmp_path / "student_data.csv"

    pd.DataFrame({
        "GPA": [3.0, 3.5, 4.0],
        "SATM": [600, 650, 700],
    }).to_csv(path, index=False)

    df = read_uploaded_dataframe(str(path))

    assert list(df.columns) == ["GPA", "SATM"]
    assert df.shape == (3, 2)


def test_initialize_dataset_session_from_csv_creates_state_and_snapshot(tmp_path):
    source_path = tmp_path / "student_data.csv"
    workspace_dir = tmp_path / "workspace"

    pd.DataFrame({
        "GPA": [3.0, 3.5, 4.0],
        "SATM": [600, 650, 700],
    }).to_csv(source_path, index=False)

    result = initialize_dataset_session_from_file(
        str(source_path),
        workspace_dir=str(workspace_dir),
        dataset_name="student_data",
    )

    state = result["state"]
    snapshot = result["snapshot"]

    assert state["dataset_name"] == "student_data"
    assert state["active_data_version_id"] == "raw_v1"
    assert len(state["data_versions"]) == 1
    assert state["data_versions"][0]["version_id"] == "raw_v1"
    assert state["data_versions"][0]["n_rows"] == 3
    assert state["data_versions"][0]["n_cols"] == 2

    raw_path = workspace_dir / "data_versions" / "raw_v1.parquet"
    assert raw_path.exists()

    assert "dataset_profile" in state
    assert "dataset_profile_v2" in state
    assert "dataset_summary" in state
    assert "capability_map" in state
    assert "dataset_context" in state

    assert state["dataset_profile_v2"]["dataset_name"] == "student_data"
    assert state["dataset_profile_v2"]["data_version_id"] == "raw_v1"
    assert "GPA" in state["dataset_profile_v2"]["columns"]

    assert snapshot["schema_version"] == "ui_snapshot_v2"
    assert snapshot["assistant_response"]["response_type"] == "dataset_loaded"
    assert snapshot["dataset"]["dataset_name"] == "student_data"
    assert snapshot["dataset"]["active_data_version_id"] == "raw_v1"
    assert snapshot["dataset"]["profile"]["data_version_id"] == "raw_v1"


def test_initialize_dataset_session_uses_filename_as_default_dataset_name(tmp_path):
    source_path = tmp_path / "my_uploaded_file.csv"
    workspace_dir = tmp_path / "workspace"

    pd.DataFrame({
        "x": [1, 2],
        "y": [3, 4],
    }).to_csv(source_path, index=False)

    result = initialize_dataset_session_from_file(
        str(source_path),
        workspace_dir=str(workspace_dir),
    )

    assert result["state"]["dataset_name"] == "my_uploaded_file"
    assert result["snapshot"]["dataset"]["dataset_name"] == "my_uploaded_file"


def test_initialize_dataset_session_rejects_unsupported_file_type(tmp_path):
    source_path = tmp_path / "notes.txt"
    source_path.write_text("not a table", encoding="utf-8")

    try:
        initialize_dataset_session_from_file(
            str(source_path),
            workspace_dir=str(tmp_path / "workspace"),
        )
    except ValueError as exc:
        assert "Unsupported tabular file type" in str(exc)
    else:
        raise AssertionError("Expected unsupported file type to raise ValueError.")

def test_initialize_dataset_session_from_xlsx_creates_state_and_snapshot(tmp_path):
    source_path = tmp_path / "student_data.xlsx"
    workspace_dir = tmp_path / "workspace"

    pd.DataFrame({
        "GPA": [3.0, 3.5, 4.0],
        "SATM": [600, 650, 700],
    }).to_excel(source_path, index=False)

    result = initialize_dataset_session_from_file(
        str(source_path),
        workspace_dir=str(workspace_dir),
        dataset_name="student_data",
    )

    state = result["state"]
    snapshot = result["snapshot"]

    assert state["active_data_version_id"] == "raw_v1"
    assert state["data_versions"][0]["n_rows"] == 3
    assert state["data_versions"][0]["n_cols"] == 2
    assert "GPA" in state["dataset_profile_v2"]["columns"]
    assert snapshot["dataset"]["dataset_name"] == "student_data"
    assert snapshot["dataset"]["active_data_version_id"] == "raw_v1"