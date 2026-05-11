from pathlib import Path

import pandas as pd

from core.data_versions import (
    create_initial_data_version,
    create_child_data_version,
    make_audit_event,
    get_active_data_path,
)


def test_create_initial_data_version_has_required_fields(tmp_path):
    df = pd.DataFrame({
        "x": [1, 2, 3],
        "y": ["a", "b", "c"],
    })

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    version = create_initial_data_version(
        df=df,
        workspace_dir=str(workspace_dir),
    )

    assert version["version_id"]
    assert version["path"]
    assert version["version_id"] == "raw_v1"
    assert version.get("parent_version_id") is None
    assert Path(version["path"]).exists()


def test_create_child_data_version_has_parent_and_path(tmp_path):
    df = pd.DataFrame({
        "x": [1, 2, 3],
    })

    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    child_path = workspace_dir / "child.parquet"
    df.to_parquet(child_path)

    version = create_child_data_version(
        workspace_dir=str(workspace_dir),
        parent_version_id="raw_v1",
        df=df,
        operation="test_operation",
    )

    assert version["version_id"].startswith("data_v_")
    assert version["parent_version_id"] == "raw_v1"
    assert version["path"]
    assert Path(version["path"]).exists()


def test_make_audit_event_records_version_transition():
    event = make_audit_event(
        event_type="data_version_created",
        description="Created cleaned data version from raw_v1.",
        version_id="data_v_1",
        parent_version_id="raw_v1",
        tool_name="clean_data",
        action_id="act_drop_rows",
        details={
            "action": "drop_rows",
            "columns": ["GPA"],
        },
    )

    assert event["event_type"] == "data_version_created"
    assert event["description"] == "Created cleaned data version from raw_v1."
    assert event["version_id"] == "data_v_1"
    assert event["parent_version_id"] == "raw_v1"
    assert event["tool_name"] == "clean_data"
    assert event["action_id"] == "act_drop_rows"
    assert event["details"]["action"] == "drop_rows"
    assert event["details"]["columns"] == ["GPA"]


def test_get_active_data_path_uses_active_version_id(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    raw_path = workspace_dir / "raw.parquet"
    child_path = workspace_dir / "child.parquet"

    pd.DataFrame({"x": [1]}).to_parquet(raw_path)
    pd.DataFrame({"x": [2]}).to_parquet(child_path)

    data_versions = [
        {
            "version_id": "raw_v1",
            "path": str(raw_path),
        },
        {
            "version_id": "data_v_1",
            "path": str(child_path),
            "parent_version_id": "raw_v1",
        },
    ]

    resolved = get_active_data_path(
        workspace_dir=str(workspace_dir),
        data_versions=data_versions,
        active_data_version_id="data_v_1",
        fallback_file="working_data.parquet",
    )

    assert resolved == str(child_path)


def test_get_active_data_path_returns_none_for_unknown_active_version(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    raw_path = workspace_dir / "raw.parquet"
    pd.DataFrame({"x": [1]}).to_parquet(raw_path)

    data_versions = [
        {
            "version_id": "raw_v1",
            "path": str(raw_path),
        },
    ]

    resolved = get_active_data_path(
        workspace_dir=str(workspace_dir),
        data_versions=data_versions,
        active_data_version_id="missing_version",
        fallback_file="working_data.parquet",
    )

    assert resolved is None