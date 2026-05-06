import pandas as pd

from core.dataset_intelligence.profiler import profile_dataframe
from core.dataset_intelligence.capability_map import build_capability_map
from core.graph import plan_only_node, advisory_answer_node


def make_state():
    df = pd.DataFrame({
        "y": [1.2, 2.4, 3.1, 4.7],
        "x": [10.5, 20.2, 30.8, 40.1],
        "group": ["A", "B", "A", "B"],
    })

    profile = profile_dataframe(df, data_version_id="raw_v1")
    capability_map = build_capability_map(profile)

    return {
        "user_request": "could you make up a plan and tell me?",
        "active_data_version_id": "raw_v1",
        "dataset_profile_v2": profile.model_dump(),
        "capability_map": capability_map.model_dump(),
        "dataset_summary": {
            "n_rows": 4,
            "n_cols": 3,
            "numeric_columns": ["y", "x"],
            "categorical_columns": ["group"],
            "binary_columns": ["group"],
            "id_like_columns": [],
            "missingness_summary": {"n_columns_with_missing": 0},
        },
    }


def test_plan_only_node_does_not_create_action():
    state = make_state()

    result = plan_only_node(state)

    assert result["pending_plan"] is not None
    assert result["plan_status"] in {"draft", "verified", "partially_ready"}
    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None

    assert "assistant_response" in result
    assert result["assistant_response"]["response_type"] == "plan"
    assert "No tools have been executed" in result["assistant_response"]["content"]

def test_advisory_answer_node_does_not_create_action():
    state = make_state()
    state["user_request"] = "what can I do?"

    result = advisory_answer_node(state)

    assert result["current_action"] is None
    assert result["current_execution"] is None
    assert result["current_verification"] is None

    assert "assistant_response" in result
    assert result["assistant_response"]["response_type"] == "advisory"
    assert "I have not run any analysis tools yet" in result["assistant_response"]["content"]