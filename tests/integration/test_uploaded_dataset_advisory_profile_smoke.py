import pandas as pd

from core.controller.backend_turn import run_backend_turn
from core.ui_adapter.dataset_upload import prepare_uploaded_dataset_state
from core.ui_adapter.events import apply_ui_event_to_state, make_user_message_event


def apply_updates(state, updates):
    merged = dict(state)
    merged.update(updates)
    return merged


def test_uploaded_dataset_advisory_uses_dataset_summary_counts(tmp_path):
    df = pd.DataFrame({
        "GPA": [3.0, 3.2, None, 3.8, 4.0],
        "SATM": [600, None, 650, 680, 700],
        "Sex": ["F", "M", "F", "M", "F"],
    })

    state = prepare_uploaded_dataset_state(
        df=df,
        workspace_dir=str(tmp_path / "workspace"),
        filename="test_data.csv",
    )

    updates = apply_ui_event_to_state(
        state,
        make_user_message_event(
            "I want to do analysis to this dataset, what can I do?"
        ),
    )
    state = apply_updates(state, updates)

    result = run_backend_turn(state)

    assert result["status"] == "ok"

    response = result["ui_snapshot"]["assistant_response"]
    content = response["content"]

    assert response["response_type"] == "advisory"

    assert "Rows: unknown" not in content
    assert "Columns: unknown" not in content
    assert "Rows: 5" in content
    assert "Columns: 3" in content
    assert "Numeric columns: 2" in content
    assert "Categorical columns: 1" in content
    assert "Binary columns: 1" in content
    assert "Columns with missing values: 2" in content