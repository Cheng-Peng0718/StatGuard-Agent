import pandas as pd

from core.data.context_refresh import refresh_dataset_context_from_df

from core.schema import ActionProposal
from core.workflow.nodes.verification import verify_node

def make_dataset_profile_v2():
    refreshed = refresh_dataset_context_from_df(
        pd.DataFrame({
            "GPA": [3.0, 3.5, 4.0],
            "SATM": [600, 650, 700],
        }),
        dataset_name="test_data",
        data_version_id="raw_v1",
    )

    return refreshed.dataset_profile_v2.model_dump()

def test_verify_node_attaches_action_hash_to_allowed_verification(monkeypatch):
    action = ActionProposal(
        action_id="act_summary",
        action_type="tool_call",
        tool_name="get_summary_stats",
        arguments={
            "columns": ["GPA"],
        },
        reasoning_summary="Summarize GPA.",
    )

    def fake_verify(action, profile):
        return (
            "allowed",
            "ok",
            {
                "status": "allowed",
                "feedback": "ok",
                "error_code": None,
                "details": {},
            },
        )

    monkeypatch.setattr(
        "core.workflow.nodes.verification.verify",
        fake_verify,
    )

    updates = verify_node({
        "current_action": action,
        "dataset_profile": {
            "columns": ["GPA"],
        },
        "repair_attempts": [],
        "dataset_profile_v2": make_dataset_profile_v2(),
    })

    verification = updates["current_verification"]

    assert verification["status"] == "allowed"
    assert "action_hash" in verification["details"]
    assert updates["human_review_required"] is False


def test_verify_node_rejected_failure_has_user_visible_response(monkeypatch):
    action = ActionProposal(
        action_id="act_bad",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "strategy": "drop",
        },
        reasoning_summary="Bad clean_data request.",
    )

    def fake_verify(action, profile):
        return (
            "rejected_recoverable",
            "Missing required arguments.",
            {
                "status": "rejected_recoverable",
                "feedback": "Missing required arguments.",
                "error_code": "INVALID_TOOL_ARGUMENTS",
                "details": {},
            },
        )

    monkeypatch.setattr(
        "core.workflow.nodes.verification.verify",
        fake_verify,
    )

    updates = verify_node({
        "current_action": action,
        "dataset_profile": {
            "columns": ["GPA"],
        },
        "dataset_profile_v2": make_dataset_profile_v2(),
        "observations": [],
        "repair_attempts": [],
        "active_data_version_id": "raw_v1",
    })

    assert updates["current_verification"]["status"] == "rejected_recoverable"
    assert updates["observations"][0]["status"] == "rejected"

    assert updates["assistant_response"]["response_type"] == "error"
    assert (
        updates["assistant_response"]["metadata"]["semantic_type"]
        == "verification_blocked"
    )

    assert "repair_decision" in updates
    assert "repair_proposal" in updates