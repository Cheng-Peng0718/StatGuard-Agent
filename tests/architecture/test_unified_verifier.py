from verifiers.validators import verify


class DummyAction:
    action_id = "act_test"
    tool_name = "run_multiple_regression"
    arguments = {
        "target_col": "GPA",
        "feature_cols": ["SATM"],
    }


class DummyCleanAction:
    action_id = "act_clean"
    tool_name = "clean_data"
    arguments = {
        "action_type": "drop",
        "strategy": "rows",
        "columns": ["GPA", "SATM"],
    }


def test_verifier_allows_non_mutating_unified_tool_without_profile():
    status, feedback = verify(DummyAction(), dataset_profile=None)

    assert status == "allowed"
    assert isinstance(feedback, str)


def test_verifier_requires_review_for_clean_data():
    status, feedback = verify(DummyCleanAction(), dataset_profile=None)

    assert status == "needs_review"
    assert "requires user confirmation" in feedback


def test_verifier_rejects_unknown_tool_recoverably():
    class UnknownAction:
        action_id = "act_unknown"
        tool_name = "unknown_tool"
        arguments = {}

    status, feedback = verify(UnknownAction(), dataset_profile=None)

    assert status == "rejected_recoverable"
    assert "Unknown tool" in feedback