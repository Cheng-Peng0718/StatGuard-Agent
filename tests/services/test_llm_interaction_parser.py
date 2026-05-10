from core.domain.intent import IntentDecision
from core.services.llm_interaction_parser import (
    build_llm_interaction_input,
    decide_llm_interaction_intent,
    generate_llm_interaction_draft,
    legacy_interaction_intent_from_decision,
)
from core.services.llm_interaction_contracts import LLMInteractionDraft


def _state():
    return {
        "active_data_version_id": "raw_v1",
        "dataset_profile_v2": {
            "dataset_name": "student_data",
            "data_version_id": "raw_v1",
            "columns": {
                "GPA": {
                    "semantic_type": "continuous_numeric",
                    "dtype": "float64",
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "n_unique": 4,
                    "examples": [3.0, 3.2],
                },
                "SATM": {
                    "semantic_type": "continuous_numeric",
                    "dtype": "int64",
                    "n_missing": 0,
                    "missing_rate": 0.0,
                    "n_unique": 4,
                    "examples": [600, 620],
                },
            },
        },
        "dataset_summary": {
            "n_rows": 4,
            "n_cols": 2,
        },
    }


def test_build_llm_interaction_input_includes_dataset_view():
    payload = build_llm_interaction_input(
        "run regression of GPA on SATM",
        state=_state(),
    )

    assert payload["user_message"] == "run regression of GPA on SATM"
    assert payload["dataset"]["dataset_available"] is True
    assert payload["dataset"]["dataset_name"] == "student_data"
    assert "GPA" in payload["dataset"]["columns"]
    assert "plan_analysis" in payload["allowed_intents"]


def test_decide_llm_interaction_intent_uses_structured_draft(monkeypatch):
    def fake_generate_llm_interaction_draft(payload):
        assert payload["user_message"] == "run regression of GPA on SATM"

        return LLMInteractionDraft(
            intent="direct_analysis",
            user_goal="Run a regression of GPA on SATM.",
            confidence=0.92,
            goal_type="regression_modeling",
            target_variables=["GPA"],
            predictor_variables=["SATM"],
            requested_methods=["linear_regression"],
            rationale="The user directly requested a regression.",
        )

    monkeypatch.setattr(
        "core.services.llm_interaction_parser.generate_llm_interaction_draft",
        fake_generate_llm_interaction_draft,
    )

    decision = decide_llm_interaction_intent(
        "run regression of GPA on SATM",
        state=_state(),
    )

    assert isinstance(decision, IntentDecision)
    assert decision.intent == "direct_analysis"
    assert decision.should_execute is True
    assert decision.task_spec is not None
    assert decision.task_spec.goal_type == "regression_modeling"
    assert decision.task_spec.target_variables == ["GPA"]
    assert decision.task_spec.predictor_variables == ["SATM"]


def test_legacy_route_mapping_from_llm_intent_decision():
    assert (
        legacy_interaction_intent_from_decision(
            IntentDecision(intent="plan_analysis", confidence=0.9)
        )
        == "plan_only"
    )
    assert (
        legacy_interaction_intent_from_decision(
            IntentDecision(intent="direct_analysis", confidence=0.9)
        )
        == "direct_tool"
    )
    assert (
        legacy_interaction_intent_from_decision(
            IntentDecision(intent="modify_data", confidence=0.9)
        )
        == "direct_tool"
    )


def test_generate_llm_interaction_draft_invokes_structured_llm(monkeypatch):
    seen = {}

    class FakeStructuredLLM:
        def invoke(self, messages):
            seen["messages"] = messages
            return LLMInteractionDraft(
                intent="plan_analysis",
                user_goal="Recommend an analysis plan.",
                confidence=0.9,
                goal_type="analysis_recommendation",
                rationale="The user asked what analysis can be done.",
            )

    class FakeChatOpenAI:
        def __init__(self, **kwargs):
            seen["llm_kwargs"] = kwargs

        def with_structured_output(self, schema, **kwargs):
            seen["schema"] = schema
            seen["structured_output_kwargs"] = kwargs
            return FakeStructuredLLM()

    monkeypatch.setattr(
        "langchain_openai.ChatOpenAI",
        FakeChatOpenAI,
    )

    draft = generate_llm_interaction_draft({
        "user_message": "what can I do with this data?",
        "dataset": {},
    })

    assert draft.intent == "plan_analysis"
    assert seen["schema"] is LLMInteractionDraft
    assert seen["structured_output_kwargs"] == {
        "method": "function_calling",
    }
    assert seen["messages"][0][0] == "system"
    assert seen["messages"][1][0] == "user"