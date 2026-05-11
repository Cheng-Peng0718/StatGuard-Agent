import pandas as pd

from core.dataset_intelligence.profiler import profile_dataframe, summarize_profile


def test_profile_dataframe_builds_column_profiles():
    df = pd.DataFrame({
        "GPA": [3.0, 3.5, 4.0, None],
        "Sex": ["M", "F", "F", "M"],
        "student_id": [1, 2, 3, 4],
    })

    profile = profile_dataframe(
        df,
        dataset_name="class_survey",
        data_version_id="raw_v1",
    )

    assert profile.dataset_name == "class_survey"
    assert profile.data_version_id == "raw_v1"
    assert profile.n_rows == 4
    assert profile.n_cols == 3

    assert profile.columns["GPA"].n_missing == 1
    assert profile.columns["Sex"].semantic_type == "binary_categorical"
    assert profile.columns["student_id"].semantic_type == "id_like"


def test_summarize_profile_groups_columns():
    df = pd.DataFrame({
        "GPA": [3.0, 3.5, 4.0, None],
        "Sex": ["M", "F", "F", "M"],
        "student_id": [1, 2, 3, 4],
    })

    profile = profile_dataframe(df, data_version_id="raw_v1")
    summary = summarize_profile(profile)

    assert summary.data_version_id == "raw_v1"
    assert "GPA" in summary.numeric_columns
    assert "Sex" in summary.binary_columns
    assert "student_id" in summary.id_like_columns
    assert summary.missingness_summary["n_columns_with_missing"] == 1