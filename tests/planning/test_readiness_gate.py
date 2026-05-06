from core.planning.readiness import assess_plan_step_readiness


def test_readiness_rejects_non_ready_step():
    step = {
        "step_id": "s1",
        "tool_name": "get_summary_stats",
        "status": "needs_user_choice",
        "execution_ready": False,
        "arguments": {},
        "required_user_choices": ["columns"],
    }

    result = assess_plan_step_readiness(step)

    assert result.executable is False
    assert result.status == "not_ready"
    assert "columns" in result.missing_user_choices


def test_readiness_rejects_completed_step():
    step = {
        "step_id": "s1",
        "tool_name": "get_summary_stats",
        "status": "ready",
        "execution_ready": True,
        "execution_status": "completed",
        "arguments": {},
    }

    result = assess_plan_step_readiness(step)

    assert result.executable is False
    assert result.status == "not_executable"


def test_readiness_rejects_unknown_tool():
    step = {
        "step_id": "s1",
        "tool_name": "not_a_real_tool",
        "status": "ready",
        "execution_ready": True,
        "arguments": {},
    }

    result = assess_plan_step_readiness(step)

    assert result.executable is False
    assert result.status == "blocked"
    assert "Unknown tool" in result.reason


def test_readiness_rejects_clean_data_missing_required_arguments():
    step = {
        "step_id": "s_clean",
        "tool_name": "clean_data",
        "status": "ready",
        "execution_ready": True,
        "arguments": {},
    }

    result = assess_plan_step_readiness(step)

    assert result.executable is False
    assert result.status == "needs_user_choice"
    assert "action_type" in result.missing_required_arguments
    assert "strategy" in result.missing_required_arguments
    assert "action_type" in result.missing_user_choices
    assert "strategy" in result.missing_user_choices


def test_readiness_allows_summary_stats_without_arguments():
    step = {
        "step_id": "s_summary",
        "tool_name": "get_summary_stats",
        "status": "ready",
        "execution_ready": True,
        "arguments": {},
    }

    result = assess_plan_step_readiness(step)

    assert result.executable is True
    assert result.action is not None
    assert result.action.tool_name == "get_summary_stats"
    assert result.action.arguments == {}


def test_readiness_allows_correlation_matrix_without_arguments():
    step = {
        "step_id": "s_corr",
        "tool_name": "get_correlation_matrix",
        "status": "ready",
        "execution_ready": True,
        "arguments": {},
    }

    result = assess_plan_step_readiness(step)

    assert result.executable is True
    assert result.action is not None
    assert result.action.tool_name == "get_correlation_matrix"
    assert result.action.arguments == {}


def test_readiness_rejects_regression_missing_arguments():
    step = {
        "step_id": "s_reg",
        "tool_name": "run_multiple_regression",
        "status": "ready",
        "execution_ready": True,
        "arguments": {},
    }

    result = assess_plan_step_readiness(step)

    assert result.executable is False
    assert result.status == "needs_user_choice"
    assert "target_col" in result.missing_required_arguments
    assert "feature_cols" in result.missing_required_arguments