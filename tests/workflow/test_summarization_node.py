from types import SimpleNamespace

from core.workflow.nodes.summarization import summarize_node


def test_summarize_node_returns_observation_and_analysis_run_delta():
    action = SimpleNamespace(
        action_id="act_summary",
        action_type="tool_call",
        tool_name="get_summary_stats",
        arguments={
            "columns": ["GPA"],
        },
    )

    state = {
        "current_action": action,
        "current_execution": {
            "execution_id": "exec_summary",
            "action_id": "act_summary",
            "tool_name": "get_summary_stats",
            "status": "ok",
            "success": True,
            "error_code": None,
            "message": "Summary completed.",
            "artifacts": [],
            "payload": {
                "rows": 10,
            },
        },
        "active_data_version_id": "raw_v1",
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "dummy.parquet",
            }
        ],
        "data_audit_log": [],
        "current_step": 3,
        "repair_attempts": [],
    }

    updates = summarize_node(state)

    assert len(updates["observations"]) == 1
    assert len(updates["analysis_runs"]) == 1

    obs = updates["observations"][0]
    run = updates["analysis_runs"][0]

    assert obs["tool_name"] == "get_summary_stats"
    assert obs["data_version_id"] == "raw_v1"
    assert run["tool_name"] == "get_summary_stats"

    assert updates["current_action"] is None
    assert updates["current_execution"] is None
    assert updates["current_verification"] is None
    assert updates["current_step"] == 4