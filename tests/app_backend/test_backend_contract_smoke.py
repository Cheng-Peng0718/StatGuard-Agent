import pandas as pd

from core.app_backend import (
    create_app_session,
    initialize_dataset_session_from_file,
    run_pending_plan_until_pause,
    run_user_turn,
)


def test_backend_contract_smoke_upload_then_user_turn(monkeypatch, tmp_path):
    session = create_app_session(
        workspace_root=str(tmp_path / "sessions"),
        session_id="ui smoke session",
    )

    source_path = tmp_path / "student_data.csv"

    pd.DataFrame({
        "GPA": [3.0, 3.5, 4.0],
        "SATM": [600, 650, 700],
        "Major": ["Stats", "Stats", "Math"],
    }).to_csv(source_path, index=False)

    upload_result = initialize_dataset_session_from_file(
        str(source_path),
        workspace_dir=session.workspace_dir,
        dataset_name="student_data",
    )

    state = upload_result["state"]
    snapshot = upload_result["snapshot"]

    assert snapshot["schema_version"] == "ui_snapshot_v2"
    assert snapshot["assistant_response"]["response_type"] == "dataset_loaded"
    assert snapshot["dataset"]["dataset_name"] == "student_data"
    assert snapshot["dataset"]["active_data_version_id"] == "raw_v1"
    assert "GPA" in snapshot["dataset"]["profile"]["columns"]

    seen = {}

    def fake_run_graph_once(input_state, *, config=None):
        seen["input_state"] = input_state
        seen["config"] = config

        updated = dict(input_state)
        updated["assistant_response"] = {
            "response_type": "advisory",
            "content": "The dataset has 3 rows and 3 columns.",
            "source_node": "advisory_answer",
            "data_version_id": updated.get("active_data_version_id"),
            "metadata": {},
        }
        updated["interaction_intent"] = "advisory"
        return updated

    monkeypatch.setattr(
        "core.app_backend.turn.run_graph_once",
        fake_run_graph_once,
    )

    turn_result = run_user_turn(
        state,
        "What does the data look like?",
        config=session.graph_config,
    )

    state = turn_result["state"]
    snapshot = turn_result["snapshot"]

    assert seen["input_state"]["user_request"] == "What does the data look like?"
    assert seen["config"] == session.graph_config

    assert snapshot["schema_version"] == "ui_snapshot_v2"
    assert snapshot["assistant_response"]["response_type"] == "advisory"
    assert snapshot["assistant_response"]["content"] == (
        "The dataset has 3 rows and 3 columns."
    )
    assert snapshot["dataset"]["dataset_name"] == "student_data"
    assert snapshot["dataset"]["active_data_version_id"] == "raw_v1"


def test_backend_contract_smoke_plan_runner_uses_public_flow(monkeypatch, tmp_path):
    session = create_app_session(
        workspace_root=str(tmp_path / "sessions"),
        session_id="plan smoke session",
    )

    source_path = tmp_path / "student_data.csv"

    pd.DataFrame({
        "GPA": [3.0, 3.5, 4.0],
        "SATM": [600, 650, 700],
    }).to_csv(source_path, index=False)

    upload_result = initialize_dataset_session_from_file(
        str(source_path),
        workspace_dir=session.workspace_dir,
        dataset_name="student_data",
    )

    state = upload_result["state"]

    state["pending_plan"] = {
        "plan_id": "plan_1",
        "status": "verified",
        "steps": [
            {
                "step_id": "s1",
                "title": "Summarize GPA",
                "tool_name": "get_summary_stats",
                "status": "ready",
                "execution_ready": True,
                "arguments": {
                    "columns": ["GPA"],
                },
            }
        ],
    }
    state["plan_status"] = "verified"
    state["plan_execution_status"] = None

    calls = []

    def fake_run_user_turn(input_state, user_message, *, config=None):
        calls.append({
            "user_message": user_message,
            "config": config,
        })

        updated = dict(input_state)
        updated["pending_plan"] = {
            "plan_id": "plan_1",
            "status": "completed",
            "steps": [],
        }
        updated["plan_status"] = "completed"
        updated["plan_execution_status"] = "completed"
        updated["assistant_response"] = {
            "response_type": "plan_execution_status",
            "content": "Plan completed.",
            "source_node": "execute_pending_plan",
            "data_version_id": updated.get("active_data_version_id"),
            "metadata": {},
        }
        updated["analysis_runs"] = [
            {
                "run_id": "run_1",
                "tool_name": "get_summary_stats",
                "status": "ok",
                "success": True,
            }
        ]
        return {
            "state": updated,
            "snapshot": {},
        }

    monkeypatch.setattr(
        "core.app_backend.plan_runner.run_user_turn",
        fake_run_user_turn,
    )

    result = run_pending_plan_until_pause(
        state,
        config=session.graph_config,
    )

    assert calls == [
        {
            "user_message": "Run the pending plan.",
            "config": session.graph_config,
        }
    ]

    assert result["plan_run"]["status"] == "completed"
    assert result["plan_run"]["reason"] == "terminal_plan_status:completed"
    assert result["snapshot"]["schema_version"] == "ui_snapshot_v2"
    assert result["snapshot"]["plan"]["plan_status"] == "completed"
    assert result["snapshot"]["analysis"]["analysis_runs"][0]["tool_name"] == (
        "get_summary_stats"
    )