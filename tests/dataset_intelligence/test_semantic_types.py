import pandas as pd

from core.dataset_intelligence.semantic_types import infer_semantic_type


def test_continuous_numeric_inferred():
    s = pd.Series([1.2, 2.5, 3.1, 4.8, 5.0, 6.2, 7.7, 8.1, 9.3, 10.4, 11.2])
    result = infer_semantic_type(s, "GPA")

    assert result["semantic_type"] == "continuous_numeric"


def test_binary_categorical_inferred_from_two_values():
    s = pd.Series(["M", "F", "F", "M", None])
    result = infer_semantic_type(s, "Sex")

    assert result["semantic_type"] == "binary_categorical"


def test_id_like_inferred_from_name():
    s = pd.Series([1, 2, 3, 4])
    result = infer_semantic_type(s, "student_id")

    assert result["semantic_type"] == "id_like"


def test_nominal_categorical_inferred_from_strings():
    s = pd.Series(["A", "B", "A", "C", "B"])
    result = infer_semantic_type(s, "Section")

    assert result["semantic_type"] == "nominal_categorical"

def test_low_unique_non_integer_float_is_continuous_numeric():
    s = pd.Series([1.2, 2.4, 3.1, 4.7])
    result = infer_semantic_type(s, "score")

    assert result["semantic_type"] == "continuous_numeric"


def test_low_unique_integer_like_numeric_is_discrete_numeric():
    s = pd.Series([1, 2, 3, 4])
    result = infer_semantic_type(s, "count_score")

    assert result["semantic_type"] == "discrete_numeric"