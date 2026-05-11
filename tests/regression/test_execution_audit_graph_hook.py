from core.workflow.audit_runtime import attach_execution_audit


def test_attach_execution_audit_records_ok_status_for_consistent_update():
    state = {
        "observations": [
            {
                "observation_id": "obs_1",
                "tool_name": "get_summary_stats",
                "status": "ok",
            }
        ],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": "raw_v1",
    }

    updates = {
        "analysis_runs": [
            {
                "tool_name": "get_summary_stats",
                "status": "ok",
                "success": True,
                "observation_id": "obs_1",
                "data_version_id": "raw_v1",
            }
        ],
    }

    result = attach_execution_audit(state, updates)

    assert "execution_audit" in result
    assert result["execution_audit"]["status"] == "ok"


def test_attach_execution_audit_records_error_for_bad_data_version():
    state = {
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": "raw_v1",
    }

    updates = {
        "active_data_version_id": "missing_version",
    }

    result = attach_execution_audit(state, updates)

    assert result["execution_audit"]["status"] == "error"
    assert any(
        issue["code"] == "ACTIVE_DATA_VERSION_NOT_REGISTERED"
        for issue in result["execution_audit"]["issues"]
    )