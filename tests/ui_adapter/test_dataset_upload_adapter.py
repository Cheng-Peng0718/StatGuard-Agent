import json
from pathlib import Path

import pandas as pd
import pytest

from core.ui_adapter.dataset_upload import (
    build_legacy_dataset_profile_from_df,
    prepare_uploaded_dataset_state,
)
from core.ui_adapter.snapshot import build_ui_snapshot


def test_build_legacy_dataset_profile_from_df_detects_basic_columns():
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, None],
        "Sex": ["F", "M", "F"],
        "Section": ["A", "B", "C"],
    })

    profile = build_legacy_dataset_profile_from_df(df)

    assert profile["n_rows"] == 3
    assert profile["n_cols"] == 3

    columns = {
        col["name"]: col
        for col in profile["columns"]
    }

    assert columns["GPA"]["semantic_type"] == "continuous_numeric"
    assert columns["GPA"]["missing_count"] == 1
    assert columns["GPA"]["missing_rate"] == pytest.approx(1 / 3)

    assert columns["Sex"]["semantic_type"] == "binary_categorical"
    assert columns["Section"]["semantic_type"] == "nominal_categorical"

    json.dumps(profile)


def test_prepare_uploaded_dataset_state_creates_initial_data_version(tmp_path):
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, 3.5],
        "SATM": [600, 620, 650],
    })

    updates = prepare_uploaded_dataset_state(
        df=df,
        workspace_dir=str(tmp_path / "workspace"),
        filename="student_data.csv",
    )

    assert updates["workspace_dir"]
    assert updates["active_data_version_id"] == "raw_v1"

    assert len(updates["data_versions"]) == 1
    version = updates["data_versions"][0]

    assert version["version_id"] == "raw_v1"
    assert Path(version["path"]).exists()

    loaded = pd.read_parquet(version["path"])

    assert list(loaded.columns) == ["GPA", "SATM"]
    assert len(loaded) == 3

    assert updates["dataset_profile"]["n_rows"] == 3
    assert updates["dataset_profile"]["n_cols"] == 2

    assert len(updates["data_audit_log"]) == 1
    assert updates["data_audit_log"][0]["event_type"] == "data_version_created"

    assert updates["uploaded_dataset_info"]["filename"] == "student_data.csv"
    assert updates["uploaded_dataset_info"]["active_data_version_id"] == "raw_v1"

    assert updates["observations"] == []
    assert updates["analysis_runs"] == []
    assert updates["pending_plan"] is None
    assert updates["current_action"] is None
    assert updates["current_execution"] is None
    assert updates["current_verification"] is None

    json.dumps(updates)


def test_prepare_uploaded_dataset_state_snapshot_is_json_safe(tmp_path):
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, 3.5],
        "SATM": [600, 620, 650],
    })

    state = prepare_uploaded_dataset_state(
        df=df,
        workspace_dir=str(tmp_path / "workspace"),
        filename="student_data.csv",
    )

    snapshot = build_ui_snapshot(state)

    assert snapshot["assistant_response"]["response_type"] == "dataset_loaded"
    assert "loaded successfully" in snapshot["assistant_response"]["content"]

    assert snapshot["data"]["active_data_version_id"] == "raw_v1"
    assert len(snapshot["data"]["data_versions"]) == 1

    assert snapshot["analysis"]["analysis_runs"] == []
    assert snapshot["runtime"]["has_current_action"] is False

    json.dumps(snapshot)


def test_prepare_uploaded_dataset_state_rejects_empty_dataset(tmp_path):
    df = pd.DataFrame()

    with pytest.raises(ValueError):
        prepare_uploaded_dataset_state(
            df=df,
            workspace_dir=str(tmp_path / "workspace"),
            filename="empty.csv",
        )