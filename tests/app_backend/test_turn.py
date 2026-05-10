import pytest

from core.app_backend.turn import prepare_turn_state, run_user_turn


def test_prepare_turn_state_sets_user_request_and_clears_transient_outputs():
    state = {
        "user_request": "old request",
        "assistant_response": {
            "content": "old response",
        },
        "current_execution": {
            "status": "ok",
        },
        "current_verification": {
            "status": "allowed",
        },
        "execution_audit": {
            "status": "old",
        },
        "dataset_name": "student_data",
    }

    prepared = prepare_turn_state(
        state,
        "What does the data look like?",
    )

    assert prepared["user_request"] == "What does the data look like?"
    assert prepared["assistant_response"] == {}
    assert prepared["current_execution"] is None
    assert prepared["current_verification"] is None
    assert prepared["execution_audit"] == {}

    # Original state should not be mutated.
    assert state["user_request"] == "old request"
    assert state["assistant_response"]["content"] == "old response"


def test_prepare_turn_state_rejects_empty_user_message():
    with pytest.raises(ValueError):
        prepare_turn_state({}, "   ")


def test_run_user_turn_invokes_graph_runner_and_returns_snapshot(monkeypatch):
    seen = {}

    def fake_run_graph_once(state, *, config=None):
        seen["state"] = state
        seen["config"] = config

        updated = dict(state)
        updated["assistant_response"] = {
            "response_type": "advisory",
            "content": "Here is what the data looks like.",
        }
        updated["dataset_profile_v2"] = {
            "dataset_name": "student_data",
            "data_version_id": "raw_v1",
            "columns": {
                "GPA": {
                    "semantic_type": "continuous_numeric",
                }
            },
        }
        updated["active_data_version_id"] = "raw_v1"
        updated["data_versions"] = [
            {
                "version_id": "raw_v1",
                "path": "workspace/data_versions/raw_v1.parquet",
            }
        ]

        return updated

    monkeypatch.setattr(
        "core.app_backend.turn.run_graph_once",
        fake_run_graph_once,
    )

    result = run_user_turn(
        {
            "dataset_name": "student_data",
            "workspace_dir": "workspace",
            "user_request": "old request",
        },
        "What does the data look like?",
        config={
            "configurable": {
                "thread_id": "session_1",
            }
        },
    )

    assert seen["state"]["user_request"] == "What does the data look like?"
    assert seen["config"] == {
        "configurable": {
            "thread_id": "session_1",
        }
    }

    assert result["state"]["assistant_response"]["response_type"] == "advisory"
    assert result["snapshot"]["schema_version"] == "ui_snapshot_v2"
    assert result["snapshot"]["assistant_response"]["content"] == (
        "Here is what the data looks like."
    )
    assert result["snapshot"]["dataset"]["dataset_name"] == "student_data"