import uuid
from core.schema import ActionProposal
from verifiers.validators import verify
from core.schema import DatasetProfile, ColumnProfile


def _profile():
    return DatasetProfile(
        dataset_name="test",
        n_rows=10,
        n_cols=3,
        columns={
            "GPA": ColumnProfile(name="GPA", dtype="float64", n_missing=0),
            "SATM": ColumnProfile(name="SATM", dtype="float64", n_missing=0),
            "Sex": ColumnProfile(name="Sex", dtype="object", n_missing=0),
        },
    )


def test_regression_missing_target_col_is_rejected():
    import tools.methods  # ensure registry populated

    action = ActionProposal(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        action_type="tool_call",
        tool_name="run_multiple_regression",
        arguments={"feature_cols": ["SATM"]},
        reasoning_summary="test",
    )

    status, feedback = verify(action, _profile())

    assert status == "rejected_recoverable"
    assert "INVALID_TOOL_ARGUMENTS" in feedback


def test_regression_feature_cols_must_be_list():
    import tools.methods

    action = ActionProposal(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        action_type="tool_call",
        tool_name="run_multiple_regression",
        arguments={"target_col": "GPA", "feature_cols": "SATM"},
        reasoning_summary="test",
    )

    status, feedback = verify(action, _profile())

    assert status == "rejected_recoverable"
    assert "INVALID_TOOL_ARGUMENTS" in feedback


def test_nonexistent_column_is_rejected():
    import tools.methods

    action = ActionProposal(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        action_type="tool_call",
        tool_name="run_multiple_regression",
        arguments={"target_col": "FakeGPA", "feature_cols": ["SATM"]},
        reasoning_summary="test",
    )

    status, feedback = verify(action, _profile())

    assert status == "rejected_recoverable"
    assert "COLUMN_NOT_FOUND" in feedback


def test_valid_regression_tool_call_passes_validation():
    import tools.methods

    action = ActionProposal(
        action_id=f"act_{uuid.uuid4().hex[:8]}",
        action_type="tool_call",
        tool_name="run_multiple_regression",
        arguments={"target_col": "GPA", "feature_cols": ["SATM"]},
        reasoning_summary="test",
    )

    status, feedback = verify(action, _profile())

    assert status == "allowed"