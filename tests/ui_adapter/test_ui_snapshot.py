import json
from types import SimpleNamespace

from core.ui_adapter.snapshot import build_ui_snapshot


def test_build_ui_snapshot_for_final_answer_state_is_json_safe():
    state = {
        "assistant_response": {
            "response_type": "final_answer",
            "content": "Done.",
            "source_node": "final_response",
        },
        "pending_plan": None,
        "plan_status": None,
        "observations": [],
        "analysis_runs": [
            {
                "tool_name": "get_summary_stats",
                "status": "ok",
                "success": True,
            }
        ],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": "data.parquet",
            }
        ],
        "active_data_version_id": "raw_v1",
        "execution_audit": {
            "status": "ok",
            "issues": [],
        },
        "state_serialization_audit": {
            "status": "ok",
            "n_issues": 0,
            "issues": [],
        },
        "deliverable_check": {
            "status": "ok",
            "satisfied": ["tool:get_summary_stats"],
            "missing": [],
            "blocked": [],
        },
        "repair_attempts": [],
    }

    snapshot = build_ui_snapshot(state)

    assert snapshot["schema_version"] == "ui_snapshot_v1"
    assert snapshot["assistant_response"]["content"] == "Done."
    assert snapshot["analysis"]["analysis_runs"][0]["tool_name"] == "get_summary_stats"
    assert snapshot["data"]["active_data_version_id"] == "raw_v1"
    assert snapshot["audits"]["execution_audit"]["status"] == "ok"

    json.dumps(snapshot)


def test_build_ui_snapshot_for_pending_plan_state():
    state = {
        "pending_plan": {
            "plan_id": "plan_1",
            "status": "partially_ready",
            "steps": [
                {
                    "step_id": "s1",
                    "tool_name": "get_summary_stats",
                    "execution_status": "not_started",
                }
            ],
        },
        "plan_status": "partially_ready",
        "observations": [],
        "analysis_runs": [],
        "data_versions": [],
        "repair_attempts": [],
    }

    snapshot = build_ui_snapshot(state)

    assert snapshot["plan"]["pending_plan"]["plan_id"] == "plan_1"
    assert snapshot["plan"]["plan_status"] == "partially_ready"
    assert snapshot["analysis"]["analysis_runs"] == []

    json.dumps(snapshot)


def test_build_ui_snapshot_exposes_human_review_state_for_simple_object_action():
    action = SimpleNamespace(
        action_id="act_clean",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "drop",
            "strategy": "rows",
            "columns": ["GPA"],
        },
        reasoning_summary="Drop missing rows.",
    )

    state = {
        "current_action": action,
        "current_verification": {
            "status": "needs_review",
            "feedback": "Action requires confirmation.",
            "details": {
                "action_hash": "abc123",
                "requires_confirmation": True,
            },
        },
        "observations": [],
        "analysis_runs": [],
        "data_versions": [],
        "repair_attempts": [],
    }

    snapshot = build_ui_snapshot(state)

    assert snapshot["human_review"]["required"] is True
    assert snapshot["human_review"]["action_hash"] == "abc123"
    assert snapshot["human_review"]["requires_confirmation"] is True

    action_snapshot = snapshot["human_review"]["action"]

    assert action_snapshot["action_id"] == "act_clean"
    assert action_snapshot["tool_name"] == "clean_data"
    assert action_snapshot["arguments"]["strategy"] == "rows"

    json.dumps(snapshot)


def test_build_ui_snapshot_runtime_exposes_current_action_without_leaking_object():
    action = SimpleNamespace(
        action_id="act_summary",
        action_type="tool_call",
        tool_name="get_summary_stats",
        arguments={},
    )

    state = {
        "current_action": action,
        "current_execution": None,
        "current_verification": None,
        "observations": [],
        "analysis_runs": [],
        "data_versions": [],
        "repair_attempts": [],
    }

    snapshot = build_ui_snapshot(state)

    assert snapshot["runtime"]["has_current_action"] is True
    assert snapshot["runtime"]["current_action"]["action_id"] == "act_summary"
    assert snapshot["runtime"]["current_action"]["tool_name"] == "get_summary_stats"

    # The UI snapshot should not expose repr(SimpleNamespace(...)).
    assert "SimpleNamespace" not in json.dumps(snapshot)

    json.dumps(snapshot)


def test_build_ui_snapshot_includes_repair_state():
    state = {
        "repair_decision": {
            "status": "terminal",
            "tool_name": "run_multiple_regression",
            "error_code": "INTERNAL_PLUGIN_ERROR",
        },
        "repair_proposal": {
            "proposal_type": "no_op",
            "source_tool_name": "run_multiple_regression",
        },
        "repair_attempts": [],
        "observations": [],
        "analysis_runs": [],
        "data_versions": [],
    }

    snapshot = build_ui_snapshot(state)

    assert snapshot["repair"]["decision"]["status"] == "terminal"
    assert snapshot["repair"]["proposal"]["proposal_type"] == "no_op"
    assert snapshot["repair"]["attempts"] == []

    json.dumps(snapshot)