from core.graph import _validate_data_version_update


def test_validate_data_version_update_rejects_none_active_version():
    result = _validate_data_version_update({
        "old_version_id": "raw_v1",
        "new_version_id": None,
        "active_data_version_id": None,
        "new_version": None,
    })

    assert result is None


def test_validate_data_version_update_requires_active_id_to_match_new_version_id():
    result = _validate_data_version_update({
        "old_version_id": "raw_v1",
        "new_version_id": "data_v_1",
        "active_data_version_id": "data_v_other",
        "new_version": {
            "version_id": "data_v_1",
        },
    })

    assert result is None


def test_validate_data_version_update_accepts_valid_update():
    result = _validate_data_version_update({
        "old_version_id": "raw_v1",
        "new_version_id": "data_v_1",
        "active_data_version_id": "data_v_1",
        "new_version": {
            "version_id": "data_v_1",
            "parent_version_id": "raw_v1",
            "path": "workspaces/test/data_versions/data_v_1.parquet",
        },
    })

    assert result is not None
    assert result["active_data_version_id"] == "data_v_1"
    assert result["new_version"]["version_id"] == "data_v_1"