from pathlib import Path

import pandas as pd

from core.workflow.nodes.execution import execute_node
from core.schema import ActionProposal


def make_action():
    return ActionProposal(
        action_id="act_clean",
        action_type="tool_call",
        tool_name="clean_data",
        arguments={
            "action_type": "drop",
            "strategy": "rows",
        },
        reasoning_summary="Drop rows with missing values.",
    )


def make_state(tmp_path):
    data_path = tmp_path / "data.csv"
    pd.DataFrame({
        "GPA": [3.0, None, 3.5],
        "SATM": [600, 620, None],
    }).to_csv(data_path, index=False)

    return {
        "workspace_dir": str(tmp_path),
        "current_step": 1,
        "max_steps": 5,
        "user_request": "clean the data",
        "dataset_profile": {
            "path": str(data_path),
            "columns": ["GPA", "SATM"],
        },
        "observations": [],
        "analysis_runs": [],
        "data_versions": [
            {
                "version_id": "raw_v1",
                "path": str(data_path),
                "label": "Raw data",
            }
        ],
        "active_data_version_id": "raw_v1",
        "data_audit_log": [],
        "deliverable_check": None,
        "current_action": make_action(),
    }


def test_execute_node_does_not_treat_human_review_observation_as_duplicate(tmp_path):
    state = make_state(tmp_path)

    state["observations"] = [
        {
            "tool_name": "clean_data",
            "arguments": {
                "action_type": "drop",
                "strategy": "rows",
            },
            "status": "rejected",
            "success": False,
            "error_code": "HUMAN_CONFIRMATION_REQUIRED",
            "raw_data": {
                "pending_action": {},
            },
        }
    ]

    result = execute_node(state)

    current_execution = result["current_execution"]

    assert isinstance(current_execution, dict)
    assert current_execution["tool_name"] == "clean_data"
    assert current_execution["status"] != "blocked"


def test_execute_node_blocks_duplicate_real_analysis_run(tmp_path):
    state = make_state(tmp_path)

    state["analysis_runs"] = [
        {
            "tool_name": "clean_data",
            "arguments": {
                "action_type": "drop",
                "strategy": "rows",
            },
            "status": "ok",
        }
    ]

    result = execute_node(state)

    current_execution = result["current_execution"]

    assert isinstance(current_execution, dict)
    assert current_execution["success"] is False
    assert "identical to a previous executed attempt" in current_execution["message"]