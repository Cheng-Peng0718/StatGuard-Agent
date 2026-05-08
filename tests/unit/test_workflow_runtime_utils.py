import numpy as np

from core.workflow.runtime_utils import get_action_hash, sanitize_results


def test_sanitize_results_converts_numpy_types():
    raw = {
        "float": np.float64(1.5),
        "int": np.int64(3),
        "bool": np.bool_(True),
        "array": np.array([1, 2, 3]),
        "nested": {
            "value": np.float32(2.5),
        },
    }

    result = sanitize_results(raw)

    assert result == {
        "float": 1.5,
        "int": 3,
        "bool": True,
        "array": [1, 2, 3],
        "nested": {
            "value": 2.5,
        },
    }


def test_get_action_hash_is_stable_for_argument_key_order():
    hash_1 = get_action_hash(
        "get_summary_stats",
        {
            "columns": ["GPA", "SATM"],
            "metrics": ["mean"],
        },
    )

    hash_2 = get_action_hash(
        "get_summary_stats",
        {
            "metrics": ["mean"],
            "columns": ["GPA", "SATM"],
        },
    )

    assert hash_1 == hash_2


def test_get_action_hash_distinguishes_tool_name():
    hash_1 = get_action_hash("get_summary_stats", {"columns": ["GPA"]})
    hash_2 = get_action_hash("run_multiple_regression", {"columns": ["GPA"]})

    assert hash_1 != hash_2