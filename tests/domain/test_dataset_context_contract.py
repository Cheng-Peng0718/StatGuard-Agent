import json

import pandas as pd

from core.data.context_refresh import refresh_dataset_context_from_df
from core.domain import DatasetContext


def _refresh_context():
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, None, 4.0],
        "SATM": [600, 620, 650, 700],
        "Sex": ["F", "M", "F", "M"],
    })

    return refresh_dataset_context_from_df(
        df,
        dataset_name="student_data",
        data_version_id="raw_v1",
        data_path="workspace/data_versions/raw_v1.parquet",
    )


def test_dataset_context_accepts_refresh_outputs():
    refreshed = _refresh_context()

    context = DatasetContext(
        data_version_id=refreshed.data_version_id,
        dataset_name="student_data",
        data_path=refreshed.data_path,
        dataset_profile_v2=refreshed.dataset_profile_v2,
        dataset_summary=refreshed.dataset_summary,
        capability_map=refreshed.capability_map,
        state_dataset_profile={
            "n_rows": 4,
            "n_cols": 3,
        },
        source="upload",
    )

    dumped = context.model_dump()

    assert dumped["data_version_id"] == "raw_v1"
    assert dumped["dataset_name"] == "student_data"
    assert dumped["data_path"] == "workspace/data_versions/raw_v1.parquet"
    assert dumped["dataset_profile_v2"]["data_version_id"] == "raw_v1"
    assert dumped["dataset_summary"]["n_rows"] == 4
    assert dumped["capability_map"]["data_version_id"] == "raw_v1"
    assert dumped["state_dataset_profile"]["n_cols"] == 3
    assert dumped["source"] == "upload"

    json.dumps(dumped)


def test_refresh_can_build_domain_dataset_context():
    refreshed = _refresh_context()

    context = refreshed.to_domain_context(
        dataset_name="student_data",
        state_dataset_profile={
            "n_rows": 4,
            "n_cols": 3,
        },
        source="build_context",
    )

    assert isinstance(context, DatasetContext)
    assert context.data_version_id == "raw_v1"
    assert context.data_path == "workspace/data_versions/raw_v1.parquet"
    assert context.dataset_profile_v2.data_version_id == "raw_v1"
    assert context.dataset_summary.n_cols == 3
    assert context.capability_map.data_version_id == "raw_v1"
    assert context.state_dataset_profile == {
        "n_rows": 4,
        "n_cols": 3,
    }
    assert context.source == "build_context"


def test_refresh_state_updates_shape_remains_legacy_compatible():
    refreshed = _refresh_context()

    updates = refreshed.to_state_updates()

    assert set(updates.keys()) == {
        "dataset_profile_v2",
        "dataset_summary",
        "capability_map",
    }
    assert "dataset_context" not in updates

    json.dumps(updates)


def test_refresh_state_updates_opt_in_includes_domain_dataset_context_dict():
    refreshed = _refresh_context()

    updates = refreshed.to_state_updates(
        include_dataset_context=True,
        dataset_name="student_data",
        state_dataset_profile={
            "n_rows": 4,
            "n_cols": 3,
        },
        source="upload",
    )

    assert {
        "dataset_profile_v2",
        "dataset_summary",
        "capability_map",
        "dataset_context",
    }.issubset(updates.keys())

    dataset_context = updates["dataset_context"]

    assert isinstance(dataset_context, dict)
    assert dataset_context["data_version_id"] == "raw_v1"
    assert dataset_context["data_path"] == "workspace/data_versions/raw_v1.parquet"
    assert dataset_context["dataset_profile_v2"]["data_version_id"] == "raw_v1"
    assert dataset_context["dataset_summary"]["n_rows"] == 4
    assert dataset_context["capability_map"]["data_version_id"] == "raw_v1"
    assert dataset_context["state_dataset_profile"]["n_cols"] == 3
    assert dataset_context["source"] == "upload"

    json.dumps(updates)
