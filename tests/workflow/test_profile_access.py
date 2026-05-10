import pytest

from core.data.context_refresh import refresh_dataset_context_from_df
from core.workflow.profile_access import (
    get_context_profile,
    get_dataset_profile_v2,
    require_context_profile,
    require_dataset_profile_v2,
)

import pandas as pd


def make_dataset_profile_v2():
    return {
        "dataset_name": "test_data",
        "data_version_id": "raw_v1",
        "n_rows": 3,
        "columns": {
            "GPA": {
                "name": "GPA",
                "dtype": "float64",
                "semantic_type": "continuous_numeric",
                "n_missing": 0,
                "missing_rate": 0.0,
                "n_unique": 3,
                "examples": [3.0, 3.5],
            },
            "SATM": {
                "name": "SATM",
                "dtype": "int64",
                "semantic_type": "continuous_numeric",
                "n_missing": 0,
                "missing_rate": 0.0,
                "n_unique": 3,
                "examples": [600, 700],
            },
        },
    }

def test_get_context_profile_returns_dataset_profile_state_value():
    state = {
        "dataset_profile": {
            "columns": {
                "GPA": {
                    "type": "numeric",
                }
            }
        }
    }

    assert get_context_profile(state) == state["dataset_profile"]


def test_require_context_profile_raises_when_missing():
    with pytest.raises(KeyError):
        require_context_profile({})


def test_get_dataset_profile_v2_validates_dict_profile():
    refreshed = refresh_dataset_context_from_df(
        pd.DataFrame({
            "GPA": [3.0, 3.5],
            "SATM": [600, 700],
        }),
        dataset_name="student_data",
        data_version_id="raw_v1",
    )

    state = {
        "dataset_profile_v2": refreshed.dataset_profile_v2.model_dump(),
    }

    profile = get_dataset_profile_v2(state)

    assert profile is not None
    assert profile.dataset_name == "student_data"
    assert profile.data_version_id == "raw_v1"
    assert "GPA" in profile.columns


def test_require_dataset_profile_v2_raises_when_missing():
    with pytest.raises(KeyError):
        require_dataset_profile_v2({})