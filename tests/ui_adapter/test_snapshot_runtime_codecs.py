import json

from core.schema import ActionProposal, VerificationResult, ToolExecutionResult
from core.ui_adapter.snapshot import build_ui_snapshot


def test_ui_snapshot_uses_codecs_for_runtime_objects():
    state = {
        "current_action": ActionProposal(
            action_id="act_1",
            action_type="tool_call",
            tool_name="get_summary_stats",
            arguments={"columns": ["GPA"]},
            reasoning_summary="Compute GPA summary.",
        ),
        "current_verification": VerificationResult(
            action_id="act_1",
            status="allowed",
            feedback="Allowed.",
            error_code=None,
            details={"action_hash": "hash_1"},
        ),
        "current_execution": ToolExecutionResult(
            execution_id="exec_1",
            action_id="act_1",
            tool_name="get_summary_stats",
            success=True,
            status="ok",
            message="Done.",
            payload={"rows": 3},
            artifacts=[],
        ),
        "observations": [],
        "analysis_runs": [],
    }

    snapshot = build_ui_snapshot(state)

    assert snapshot["runtime"]["current_action"]["action_id"] == "act_1"
    assert snapshot["runtime"]["current_verification"]["status"] == "allowed"
    assert snapshot["runtime"]["current_execution"]["execution_id"] == "exec_1"

    json.dumps(snapshot)


def test_ui_snapshot_handles_dict_runtime_state():
    state = {
        "current_action": {
            "action_id": "act_2",
            "action_type": "tool_call",
            "tool_name": "clean_data",
            "arguments": {"strategy": "rows"},
            "summary": "Legacy summary field.",
        },
        "current_verification": {
            "action_id": "act_2",
            "status": "needs_review",
            "feedback": "Cleaning requires approval.",
            "error_code": None,
            "details": {"action_hash": "hash_2"},
        },
        "current_execution": {
            "execution_id": "exec_2",
            "action_id": "act_2",
            "tool_name": "clean_data",
            "success": False,
            "status": "blocked",
            "message": "Waiting for approval.",
            "payload": {},
            "artifacts": [],
        },
        "observations": [],
        "analysis_runs": [],
    }

    snapshot = build_ui_snapshot(state)

    assert snapshot["runtime"]["current_action"]["reasoning_summary"] == "Legacy summary field."
    assert snapshot["human_review"]["required"] is True
    assert snapshot["human_review"]["action_hash"] == "hash_2"
    assert snapshot["runtime"]["current_execution"]["status"] == "blocked"

    json.dumps(snapshot)