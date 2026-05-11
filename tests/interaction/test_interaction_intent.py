from core.interaction_intent import (
    InteractionIntent,
    classify_interaction_intent,
)


def test_what_can_i_do_is_advisory():
    intent = classify_interaction_intent(
        "I want to do analysis to this dataset, what can I do?"
    )

    assert intent == InteractionIntent.ADVISORY


def test_make_plan_and_tell_me_is_plan_only():
    intent = classify_interaction_intent(
        "could you make up a plan and tell me?"
    )

    assert intent == InteractionIntent.PLAN_ONLY


def test_run_the_plan_is_execute_plan():
    intent = classify_interaction_intent("run the plan")

    assert intent == InteractionIntent.EXECUTE_PLAN


def test_direct_regression_request_is_direct_tool():
    intent = classify_interaction_intent("run linear regression of GPA on SATM")

    assert intent == InteractionIntent.DIRECT_TOOL