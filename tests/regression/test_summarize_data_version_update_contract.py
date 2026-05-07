from types import SimpleNamespace

from core.graph import summarize_node


def make_action(
    *,
    action_id="act_clean",
    tool_name="clean_data",
    arguments=None,
):
    return SimpleNamespace(
        action_id=action_id,
        tool_name=tool_name,
        arguments=arguments or {},
    )


def test_summarize_node_applies_data_version_update_to_state_observation_and_analysis_run():
    new_version = {
        "version_id": "data_v_cleaned",
        "parent_version_id": "raw_v1",
        "path": "workspaces/test/data_versions/data_v_cleaned.parquet",
        "n_rows": 216,
        "n_cols": 25,
    }

    audit_event = {
        "event_type": "data_version_created",
        "old_version_id": "raw_v1",
        "new_version_id": "data_v_cleaned",
        "tool_name": "clean_data",
    }

    state = {
        "current_action": make_action(
            action_id="act_clean",
            tool_name="clean_data",
            arguments={
                "action_type": "drop",
                "strategy": "rows",
                "columns": ["GPA", "SATM"],
            },
        ),
        "current_execution": {
            "execution_id": "exec_clean",
            "status": "ok",
            "success": True,
            "error_code": None,
            "message": "Dropped rows with missing GPA/SATM.",
            "artifacts": [],
            "payload": {
                "original_n_rows": 226,
                "final_n_rows": 216,
                "rows_removed": 10,
                "data_version_update": {
                    "old_version_id": "raw_v1",
                    "new_version_id": "data_v_cleaned",
                    "active_data_version_id": "data_v_cleaned",
                    "new_version": new_version,
                    "audit_event": audit_event,
                },
            },
        },
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "workspaces/test/data_versions/raw_v1.parquet",
                "n_rows": 226,
                "n_cols": 25,
            }
        ],
        "data_audit_log": [],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
    }

    result = summarize_node(state)

    assert result["active_data_version_id"] == "data_v_cleaned"

    assert "data_versions" in result
    assert len(result["data_versions"]) == 2
    assert result["data_versions"][-1]["version_id"] == "data_v_cleaned"
    assert result["data_versions"][-1]["parent_version_id"] == "raw_v1"

    assert "data_audit_log" in result
    assert result["data_audit_log"][-1]["new_version_id"] == "data_v_cleaned"

    assert len(result["observations"]) == 1
    observation = result["observations"][0]

    assert observation["tool_name"] == "clean_data"
    assert observation["data_version_id"] == "data_v_cleaned"
    assert observation["structured_data"]["data_version_id"] == "data_v_cleaned"
    assert observation["structured_data"]["payload"]["data_version_id"] == "data_v_cleaned"
    assert observation["structured_data"]["payload"]["active_data_version_id"] == "data_v_cleaned"

    assert len(result["analysis_runs"]) == 1
    analysis_run = result["analysis_runs"][0]

    assert analysis_run["tool_name"] == "clean_data"
    assert analysis_run["observation_id"] == observation["observation_id"]
    assert analysis_run["data_version_id"] == "data_v_cleaned"
    assert analysis_run["status"] == "ok"
    assert analysis_run["success"] is True

    assert "execution_audit" in result
    assert result["execution_audit"]["status"] == "ok"


def test_summarize_node_does_not_set_active_data_version_to_none_when_update_is_invalid():
    state = {
        "current_action": make_action(
            action_id="act_bad_clean",
            tool_name="clean_data",
            arguments={
                "action_type": "drop",
                "strategy": "rows",
                "columns": ["GPA"],
            },
        ),
        "current_execution": {
            "execution_id": "exec_bad_clean",
            "status": "ok",
            "success": True,
            "error_code": None,
            "message": "Tool returned malformed data_version_update.",
            "artifacts": [],
            "payload": {
                "data_version_update": {
                    "old_version_id": "raw_v1",
                    "new_version_id": None,
                    "active_data_version_id": None,
                    "new_version": None,
                },
            },
        },
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "workspaces/test/data_versions/raw_v1.parquet",
            }
        ],
        "data_audit_log": [],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
    }

    result = summarize_node(state)

    assert "active_data_version_id" not in result

    # Invalid data_version_update should not append a broken version.
    assert "data_versions" not in result or all(
        version.get("version_id") is not None
        for version in result["data_versions"]
    )

    observation = result["observations"][0]
    assert observation["data_version_id"] == "raw_v1"

    analysis_run = result["analysis_runs"][0]
    assert analysis_run["data_version_id"] == "raw_v1"