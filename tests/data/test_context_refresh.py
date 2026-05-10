import json

import pandas as pd

from core.data.context_refresh import (
    refresh_dataset_context_from_df,
    refresh_dataset_context_from_path,
)


def test_refresh_dataset_context_from_df_builds_profile_summary_and_capability_map():
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, None, 4.0],
        "SATM": [600, 620, 650, 700],
        "Sex": ["F", "M", "F", "M"],
    })

    refreshed = refresh_dataset_context_from_df(
        df,
        dataset_name="student_data",
        data_version_id="raw_v1",
    )

    updates = refreshed.to_state_updates()

    assert updates["dataset_profile_v2"]["dataset_name"] == "student_data"
    assert updates["dataset_profile_v2"]["data_version_id"] == "raw_v1"

    summary = updates["dataset_summary"]
    assert summary["n_rows"] == 4
    assert summary["n_cols"] == 3
    assert "GPA" in summary["numeric_columns"]
    assert "Sex" in summary["binary_columns"]
    assert summary["missingness_summary"]["n_columns_with_missing"] == 1
    assert "GPA" in summary["missingness_summary"]["missing_by_column"]

    capability_map = updates["capability_map"]
    assert capability_map["data_version_id"] == "raw_v1"
    assert "capabilities" in capability_map
    assert any(
        capability["tool_name"] == "get_summary_stats"
        for capability in capability_map["capabilities"]
    )

    json.dumps(updates)


def test_state_updates_default_shape_remains_legacy_only():
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, None, 4.0],
        "SATM": [600, 620, 650, 700],
    })

    refreshed = refresh_dataset_context_from_df(
        df,
        dataset_name="student_data",
        data_version_id="raw_v1",
    )

    updates = refreshed.to_state_updates()

    assert set(updates.keys()) == {
        "dataset_profile_v2",
        "dataset_summary",
        "capability_map",
    }
    assert "dataset_context" not in updates


def test_state_updates_can_opt_in_to_dataset_context_dict():
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, None, 4.0],
        "SATM": [600, 620, 650, 700],
    })

    refreshed = refresh_dataset_context_from_df(
        df,
        dataset_name="student_data",
        data_version_id="raw_v1",
        data_path="workspace/data_versions/raw_v1.parquet",
    )

    updates = refreshed.to_state_updates(
        include_dataset_context=True,
        dataset_name="student_data",
        state_dataset_profile={
            "n_rows": 4,
            "n_cols": 2,
        },
        source="build_context",
    )

    assert {
        "dataset_profile_v2",
        "dataset_summary",
        "capability_map",
        "dataset_context",
    }.issubset(updates)

    dataset_context = updates["dataset_context"]

    assert isinstance(dataset_context, dict)
    assert dataset_context["data_version_id"] == "raw_v1"
    assert dataset_context["dataset_name"] == "student_data"
    assert dataset_context["data_path"] == "workspace/data_versions/raw_v1.parquet"
    assert dataset_context["source"] == "build_context"
    assert dataset_context["state_dataset_profile"]["n_cols"] == 2

    json.dumps(updates)


def test_refresh_dataset_context_from_path_matches_dataframe_refresh(tmp_path):
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, None, 4.0],
        "SATM": [600, 620, 650, 700],
    })
    data_path = tmp_path / "data.parquet"
    df.to_parquet(data_path)

    from_df = refresh_dataset_context_from_df(
        df,
        dataset_name="student_data",
        data_version_id="raw_v1",
    ).to_state_updates()
    from_path = refresh_dataset_context_from_path(
        str(data_path),
        dataset_name="student_data",
        data_version_id="raw_v1",
    ).to_state_updates()

    assert from_path["dataset_profile_v2"] == from_df["dataset_profile_v2"]
    assert from_path["dataset_summary"] == from_df["dataset_summary"]
    assert from_path["capability_map"] == from_df["capability_map"]
