from core.workflow.nodes.interaction import (
    advisory_answer_node,
    intent_router_node,
)
from core.domain.intent import IntentDecision


def test_intent_router_node_classifies_advisory_request(monkeypatch):
    def fake_decide_llm_interaction_intent(user_request, *, state=None):
        assert user_request == "I want to do analysis to this dataset, what can I do?"
        return IntentDecision(
            intent="advisory",
            confidence=0.9,
            reason="The user is asking for general analysis guidance.",
        )

    monkeypatch.setattr(
        "core.workflow.nodes.interaction.decide_llm_interaction_intent",
        fake_decide_llm_interaction_intent,
    )

    updates = intent_router_node({
        "user_request": "I want to do analysis to this dataset, what can I do?"
    })

    assert updates["interaction_intent"] == "advisory"
    assert updates["intent_decision"]["intent"] == "advisory"


def test_advisory_answer_node_returns_advisory_response_without_action():
    state = {
        "interaction_intent": "advisory",
        "active_data_version_id": "raw_v1",
        "dataset_summary": {
            "n_rows": 5,
            "n_cols": 3,
            "numeric_columns": ["GPA", "SATM"],
            "categorical_columns": ["Sex"],
            "binary_columns": [],
            "id_like_columns": [],
            "missingness_summary": {
                "n_columns_with_missing": 1,
            },
        },
        "capability_map": {
            "capabilities": [
                {
                    "tool_name": "get_summary_stats",
                    "display_name": "Summary Statistics",
                    "status": "ready",
                    "reason": "Numeric variables are available.",
                }
            ]
        },
    }

    updates = advisory_answer_node(state)

    assert updates["assistant_response"]["response_type"] == "advisory"
    assert "Rows: 5" in updates["assistant_response"]["content"]
    assert "Summary Statistics" in updates["assistant_response"]["content"]

    assert updates["current_action"] is None
    assert updates["current_execution"] is None
    assert updates["current_verification"] is None
