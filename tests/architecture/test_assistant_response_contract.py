import json

import pandas as pd

from core.responses import AssistantResponse
from core.ui_adapter.dataset_upload import prepare_uploaded_dataset_state
from core.ui_adapter.events import apply_ui_event_to_state


def test_dataset_upload_assistant_response_matches_contract(tmp_path):
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, 3.5],
        "SATM": [600, 620, 650],
    })

    updates = prepare_uploaded_dataset_state(
        df=df,
        workspace_dir=str(tmp_path / "workspace"),
        filename="student_data.csv",
    )

    response = AssistantResponse.model_validate(updates["assistant_response"])

    assert response.response_type == "dataset_loaded"
    assert response.source_node == "dataset_upload"

    json.dumps(response.model_dump())


def test_plan_step_choices_assistant_response_matches_contract():
    state = {
        "pending_plan": {
            "plan_id": "plan_1",
            "steps": [
                {
                    "step_id": "s1",
                    "tool_name": "run_multiple_regression",
                    "status": "needs_user_choice",
                    "execution_ready": False,
                    "execution_status": "not_started",
                    "variables": {},
                    "arguments": {},
                    "required_user_choices": ["target_col", "feature_cols"],
                }
            ],
        }
    }

    event = {
        "event_type": "update_plan_step_choices",
        "payload": {
            "step_id": "s1",
            "choices": {
                "target_col": "GPA",
                "feature_cols": ["SATM"],
            },
        },
    }

    updates = apply_ui_event_to_state(state, event)

    response = AssistantResponse.model_validate(updates["assistant_response"])

    assert response.response_type == "plan_step_choices_updated"
    assert response.source_node == "ui_event_adapter"

    json.dumps(response.model_dump())