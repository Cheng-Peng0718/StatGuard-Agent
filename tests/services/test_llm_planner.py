import pandas as pd

from core.data.context_refresh import refresh_dataset_context_from_df
from core.services.llm_planner import build_llm_planner_input, create_llm_plan_from_state


def _state():
    refreshed = refresh_dataset_context_from_df(
        pd.DataFrame({
            "GPA": [3.0, 3.2, 3.8, 4.0],
            "SATM": [600, 620, 650, 700],
            "Sex": ["F", "M", "F", "M"],
        }),
        dataset_name="student_data",
        data_version_id="raw_v1",
    )

    updates = refreshed.to_state_updates()

    updates.update({
        "user_request": "What analysis can I do with this data?",
        "interaction_intent": "plan_only",
        "active_data_version_id": "raw_v1",
    })

    return updates


def test_build_llm_planner_input_includes_dataset_profile_and_tools():
    planner_input = build_llm_planner_input(_state())

    assert planner_input.user_request == "What analysis can I do with this data?"
    assert planner_input.dataset.dataset_name == "student_data"
    assert planner_input.dataset.data_version_id == "raw_v1"
    assert "GPA" in planner_input.dataset.columns
    assert planner_input.dataset.columns["GPA"]["semantic_type"] == "continuous_numeric"

    tool_names = {
        tool.tool_name
        for tool in planner_input.tools
    }

    assert "inspect_dataset" in tool_names
    assert "run_multiple_regression" in tool_names
    assert "clean_data" in tool_names


def test_build_llm_planner_input_exposes_tool_contracts():
    planner_input = build_llm_planner_input(_state())

    regression_tool = next(
        tool
        for tool in planner_input.tools
        if tool.tool_name == "run_multiple_regression"
    )

    assert "regression_modeling" in regression_tool.supported_goal_types
    assert regression_tool.argument_schema["required"]["target_col"] == "str"
    assert regression_tool.task_argument_bindings
    assert regression_tool.requires_confirmation is False
    assert regression_tool.mutates_data is False

    clean_data_tool = next(
        tool
        for tool in planner_input.tools
        if tool.tool_name == "clean_data"
    )

    assert clean_data_tool.requires_confirmation is True
    assert clean_data_tool.mutates_data is True
    assert clean_data_tool.required_planning_choices == [
        "action_type",
        "strategy",
    ]


def test_create_llm_plan_from_state_returns_plan_proposal():
    plan = create_llm_plan_from_state(_state())

    assert plan.plan_id.startswith("plan_")
    assert plan.steps
    assert plan.status in {
        "draft",
        "ready",
        "verified",
        "partially_ready",
        "needs_clarification",
        "blocked",
    }