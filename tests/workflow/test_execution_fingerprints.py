from core.workflow.execution_fingerprints import (
    has_duplicate_executed_action,
    iter_executed_action_hashes,
)


def test_human_confirmation_observation_does_not_count_as_execution():
    state = {
        "observations": [
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
        ],
        "analysis_runs": [],
    }

    assert list(iter_executed_action_hashes(state)) == []

    assert not has_duplicate_executed_action(
        state=state,
        tool_name="clean_data",
        arguments={
            "action_type": "drop",
            "strategy": "rows",
        },
    )


def test_verification_rejection_observation_does_not_count_as_execution():
    state = {
        "observations": [
            {
                "tool_name": "run_multiple_regression",
                "arguments": {
                    "target_col": "GPA",
                    "feature_cols": ["SATM"],
                },
                "status": "rejected",
                "success": False,
                "error_code": "VERIFICATION_FAILED",
                "raw_data": {
                    "verification": {},
                    "recoverable": True,
                },
            }
        ],
        "analysis_runs": [],
    }

    assert list(iter_executed_action_hashes(state)) == []

    assert not has_duplicate_executed_action(
        state=state,
        tool_name="run_multiple_regression",
        arguments={
            "target_col": "GPA",
            "feature_cols": ["SATM"],
        },
    )


def test_analysis_run_counts_as_executed_action():
    state = {
        "analysis_runs": [
            {
                "tool_name": "get_summary_stats",
                "arguments": {
                    "columns": ["GPA"],
                },
                "status": "ok",
            }
        ],
        "observations": [],
    }

    assert has_duplicate_executed_action(
        state=state,
        tool_name="get_summary_stats",
        arguments={
            "columns": ["GPA"],
        },
    )


def test_real_execution_observation_counts_as_execution_fallback():
    state = {
        "analysis_runs": [],
        "observations": [
            {
                "tool_name": "get_summary_stats",
                "arguments": {
                    "columns": ["GPA"],
                },
                "status": "ok",
                "success": True,
                "raw_data": {
                    "execution_id": "exec_1",
                },
            }
        ],
    }

    assert has_duplicate_executed_action(
        state=state,
        tool_name="get_summary_stats",
        arguments={
            "columns": ["GPA"],
        },
    )

def test_human_review_rejection_observation_does_not_count_as_execution():
    state = {
        "observations": [
            {
                "tool_name": "clean_data",
                "arguments": {
                    "action_type": "drop",
                    "strategy": "rows",
                },
                "status": "rejected",
                "success": False,
                "error_code": "HUMAN_REVIEW_REJECTED",
                "raw_data": {
                    "human_review_decision": "rejected",
                },
            }
        ],
        "analysis_runs": [],
    }

    assert list(iter_executed_action_hashes(state)) == []

    assert not has_duplicate_executed_action(
        state=state,
        tool_name="clean_data",
        arguments={
            "action_type": "drop",
            "strategy": "rows",
        },
    )