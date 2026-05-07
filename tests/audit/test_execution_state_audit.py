from core.audit.execution_state import audit_execution_state


def test_execution_audit_ok_for_consistent_state():
    result = audit_execution_state({
        "observations": [
            {
                "observation_id": "obs_1",
                "tool_name": "get_summary_stats",
                "status": "ok",
            }
        ],
        "analysis_runs": [
            {
                "tool_name": "get_summary_stats",
                "status": "ok",
                "success": True,
                "observation_id": "obs_1",
                "data_version_id": "raw_v1",
            }
        ],
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": "raw_v1",
    })

    assert result.status == "ok"
    assert result.issues == []


def test_execution_audit_flags_analysis_run_unknown_observation():
    result = audit_execution_state({
        "observations": [
            {
                "observation_id": "obs_1",
            }
        ],
        "analysis_runs": [
            {
                "tool_name": "get_summary_stats",
                "status": "ok",
                "success": True,
                "observation_id": "obs_missing",
                "data_version_id": "raw_v1",
            }
        ],
    })

    assert result.status == "error"
    assert any(
        issue.code == "ANALYSIS_RUN_REFERENCES_UNKNOWN_OBSERVATION"
        for issue in result.issues
    )


def test_execution_audit_flags_active_data_version_not_registered():
    result = audit_execution_state({
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": "data_v_missing",
    })

    assert result.status == "error"
    assert any(
        issue.code == "ACTIVE_DATA_VERSION_NOT_REGISTERED"
        for issue in result.issues
    )


def test_execution_audit_warns_when_data_versions_exist_but_active_missing():
    result = audit_execution_state({
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": None,
    })

    assert result.status == "warning"
    assert any(
        issue.code == "ACTIVE_DATA_VERSION_MISSING"
        for issue in result.issues
    )


def test_execution_audit_flags_current_plan_step_not_in_pending_plan():
    result = audit_execution_state({
        "pending_plan": {
            "plan_id": "plan_1",
            "steps": [
                {
                    "step_id": "s1",
                    "execution_status": "running",
                }
            ],
        },
        "current_plan_step_id": "s_missing",
        "action_origin": "pending_plan",
        "current_action": {
            "tool_name": "get_summary_stats",
        },
    })

    assert result.status == "error"
    assert any(
        issue.code == "CURRENT_PLAN_STEP_NOT_IN_PENDING_PLAN"
        for issue in result.issues
    )


def test_execution_audit_warns_pending_plan_step_without_current_action():
    result = audit_execution_state({
        "pending_plan": {
            "plan_id": "plan_1",
            "steps": [
                {
                    "step_id": "s1",
                    "execution_status": "running",
                }
            ],
        },
        "current_plan_step_id": "s1",
        "action_origin": "pending_plan",
        "current_action": None,
    })

    assert result.status == "warning"
    assert any(
        issue.code == "PENDING_PLAN_STEP_WITHOUT_CURRENT_ACTION"
        for issue in result.issues
    )