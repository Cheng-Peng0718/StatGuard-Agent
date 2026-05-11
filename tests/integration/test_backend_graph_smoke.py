from types import SimpleNamespace
from core.workflow.nodes.finalization import (
    deliverable_gate_node,
    final_response_node,
)

from core.workflow.nodes.summarization import summarize_node


def make_action(
    *,
    action_id="act_1",
    tool_name="get_summary_stats",
    arguments=None,
):
    return SimpleNamespace(
        action_id=action_id,
        action_type="tool_call",
        tool_name=tool_name,
        arguments=arguments or {},
    )


def apply_updates(state, updates):
    merged = dict(state)
    merged.update(updates)
    return merged


def test_backend_smoke_successful_tool_to_deliverable_gate_to_final_response():
    state = {
        "current_action": make_action(
            action_id="act_summary",
            tool_name="get_summary_stats",
            arguments={},
        ),
        "current_execution": {
            "execution_id": "exec_summary",
            "status": "ok",
            "success": True,
            "error_code": None,
            "message": "Summary statistics completed.",
            "artifacts": [],
            "payload": {
                "n_rows": 226,
                "n_cols": 25,
            },
        },
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
        "task_contract": {
            "required_tools": ["get_summary_stats"],
        },
    }

    summarize_updates = summarize_node(state)
    state = apply_updates(state, summarize_updates)

    assert len(state["observations"]) == 1
    assert len(state["analysis_runs"]) == 1
    assert state["analysis_runs"][0]["tool_name"] == "get_summary_stats"
    assert state["analysis_runs"][0]["status"] == "ok"
    assert state["execution_audit"]["status"] == "ok"

    gate_updates = deliverable_gate_node(state)
    state = apply_updates(state, gate_updates)

    assert state["deliverable_check"]["status"] == "ok"
    assert "tool:get_summary_stats" in state["deliverable_check"]["satisfied"]

    state["final_answer"] = "Summary statistics completed successfully."

    final_updates = final_response_node(state)

    assert "assistant_response" in final_updates
    assert final_updates["assistant_response"]["response_type"] == "final_answer"
    assert (
        final_updates["assistant_response"]["content"]
        == "Summary statistics completed successfully."
    )


def test_backend_smoke_failed_required_tool_blocks_final_answer():
    state = {
        "current_action": make_action(
            action_id="act_reg",
            tool_name="run_multiple_regression",
            arguments={
                "target_col": "GPA",
                "feature_cols": ["SATM"],
            },
        ),
        "current_execution": {
            "execution_id": "exec_reg_failed",
            "status": "failed",
            "success": False,
            "error_code": "INTERNAL_PLUGIN_ERROR",
            "message": "Regression plugin crashed.",
            "artifacts": [],
            "payload": {},
        },
        "observations": [],
        "analysis_runs": [],
        "repair_attempts": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
            }
        ],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
        "task_contract": {
            "required_tools": ["run_multiple_regression"],
        },
    }

    summarize_updates = summarize_node(state)
    state = apply_updates(state, summarize_updates)

    assert len(state["observations"]) == 1
    assert len(state["analysis_runs"]) == 1

    run = state["analysis_runs"][0]

    assert run["tool_name"] == "run_multiple_regression"
    assert run["status"] == "failed"
    assert run["success"] is False
    assert run["error_code"] == "INTERNAL_PLUGIN_ERROR"

    assert "repair_decision" in state
    assert state["repair_decision"]["status"] == "terminal"

    gate_updates = deliverable_gate_node(state)
    state = apply_updates(state, gate_updates)

    assert state["deliverable_check"]["status"] == "needs_more_work"
    assert "tool:run_multiple_regression" in state["deliverable_check"]["missing"]
    assert (
        "tool_failed:run_multiple_regression"
        in state["deliverable_check"]["blocked"]
    )


def test_backend_smoke_clean_data_version_update_reaches_observation_and_analysis_run():
    new_version = {
        "version_id": "data_v_cleaned",
        "parent_version_id": "raw_v1",
        "path": "workspaces/test/data_versions/data_v_cleaned.parquet",
        "n_rows": 216,
        "n_cols": 25,
    }

    audit_event = {
        "event_type": "data_version_created",
        "description": "Created cleaned version.",
        "version_id": "data_v_cleaned",
        "parent_version_id": "raw_v1",
        "tool_name": "clean_data",
        "action_id": "act_clean",
        "details": {
            "columns": ["GPA", "SATM"],
        },
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
            }
        ],
        "data_audit_log": [],
        "repair_attempts": [],
        "active_data_version_id": "raw_v1",
        "current_step": 0,
        "task_contract": {
            "required_tools": ["clean_data"],
        },
    }

    summarize_updates = summarize_node(state)
    state = apply_updates(state, summarize_updates)

    assert state["active_data_version_id"] == "data_v_cleaned"

    assert len(state["data_versions"]) == 2
    assert state["data_versions"][-1]["version_id"] == "data_v_cleaned"

    assert len(state["data_audit_log"]) == 1
    assert state["data_audit_log"][0]["version_id"] == "data_v_cleaned"

    observation = state["observations"][0]
    analysis_run = state["analysis_runs"][0]

    assert observation["tool_name"] == "clean_data"
    assert observation["data_version_id"] == "data_v_cleaned"

    assert analysis_run["tool_name"] == "clean_data"
    assert analysis_run["observation_id"] == observation["observation_id"]
    assert analysis_run["data_version_id"] == "data_v_cleaned"

    assert state["execution_audit"]["status"] == "ok"

    gate_updates = deliverable_gate_node(state)
    state = apply_updates(state, gate_updates)

    assert state["deliverable_check"]["status"] == "ok"
    assert "tool:clean_data" in state["deliverable_check"]["satisfied"]