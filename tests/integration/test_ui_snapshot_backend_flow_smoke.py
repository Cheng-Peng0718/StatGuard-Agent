import json

import pandas as pd

from core.ui_adapter.snapshot import build_ui_snapshot

from core.workflow.nodes.execution import execute_node

from core.workflow.nodes.finalization import final_response_node

from core.workflow.nodes.summarization import summarize_node

from core.workflow.nodes.verification import verify_node

from core.workflow.nodes.plan_execution import execute_pending_plan_node

from core.workflow.nodes.finalization import deliverable_gate_node


def apply_updates(state, updates):
    merged = dict(state)
    merged.update(updates)
    return merged


def as_dict(value):
    if value is None:
        return {}

    if isinstance(value, dict):
        return value

    if hasattr(value, "model_dump"):
        return value.model_dump()

    if hasattr(value, "dict"):
        return value.dict()

    return {}


def make_legacy_dataset_profile():
    return {
        "n_rows": 5,
        "n_cols": 2,
        "columns": [
            {
                "name": "GPA",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 5,
            },
            {
                "name": "SATM",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "missing_count": 0,
                "missing_rate": 0.0,
                "n_unique": 5,
            },
        ],
    }


def make_workspace(tmp_path):
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    df = pd.DataFrame({
        "GPA": [3.0, 3.2, 3.5, 3.8, 4.0],
        "SATM": [600, 620, 650, 680, 700],
    })

    data_path = workspace_dir / "working_data.parquet"
    df.to_parquet(data_path)

    return workspace_dir, data_path


def test_ui_snapshot_from_successful_backend_flow_is_json_safe(tmp_path):
    workspace_dir, data_path = make_workspace(tmp_path)

    state = {
        "user_request": "run the plan",
        "workspace_dir": str(workspace_dir),
        "current_step": 0,
        "max_steps": 5,
        "dataset_profile": make_legacy_dataset_profile(),
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": str(data_path),
            }
        ],
        "data_audit_log": [],
        "active_data_version_id": "raw_v1",
        "pending_plan": {
            "plan_id": "plan_ui_snapshot_smoke",
            "status": "partially_ready",
            "steps": [
                {
                    "step_id": "s1",
                    "title": "Compute summary statistics",
                    "tool_name": "get_summary_stats",
                    "status": "ready",
                    "execution_ready": True,
                    "execution_status": "not_started",
                    "arguments": {},
                    "reason": "Summarize active dataset.",
                }
            ],
            "blocked_or_not_recommended": [],
        },
        "plan_status": "partially_ready",
        "current_plan_step_id": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "repair_attempts": [],
        "task_contract": {
            "required_tools": ["get_summary_stats"],
        },
    }

    updates = execute_pending_plan_node(state)
    state = apply_updates(state, updates)

    updates = verify_node(state)
    state = apply_updates(state, updates)

    verification = as_dict(state["current_verification"])
    assert verification["status"] == "allowed"

    updates = execute_node(state)
    state = apply_updates(state, updates)

    updates = summarize_node(state)
    state = apply_updates(state, updates)

    updates = deliverable_gate_node(state)
    state = apply_updates(state, updates)

    state["final_answer"] = "Summary statistics completed successfully."

    updates = final_response_node(state)
    state = apply_updates(state, updates)

    snapshot = build_ui_snapshot(state)

    assert snapshot["assistant_response"]["response_type"] == "final_answer"
    assert snapshot["assistant_response"]["content"] == "Summary statistics completed successfully."

    assert len(snapshot["analysis"]["analysis_runs"]) == 1
    assert snapshot["analysis"]["analysis_runs"][0]["tool_name"] == "get_summary_stats"

    assert snapshot["audits"]["deliverable_check"]["status"] == "ok"
    assert snapshot["runtime"]["has_current_action"] is False

    json.dumps(snapshot)