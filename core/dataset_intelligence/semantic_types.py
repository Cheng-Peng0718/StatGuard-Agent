from __future__ import annotations

from typing import Any, Dict

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


ID_NAME_HINTS = {
    "id",
    "studentid",
    "student_id",
    "user_id",
    "userid",
    "record_id",
    "index",
}


def _clean_name(name: str) -> str:
    return str(name).strip().lower().replace(" ", "_")


def infer_semantic_type(series: pd.Series, column_name: str) -> Dict[str, Any]:
    """
    Infer a generic semantic type for a column.

    This is intentionally method-agnostic.
    It does not know about regression, chi-square, etc.
    """
    n = len(series)
    non_missing = series.dropna()
    n_non_missing = len(non_missing)

    n_unique = int(non_missing.nunique(dropna=True)) if n_non_missing else 0
    unique_rate = float(n_unique / n_non_missing) if n_non_missing else 0.0

    name_clean = _clean_name(column_name)

    warnings = []

    if n == 0:
        return {
            "semantic_type": "unknown",
            "measurement_scale": "unknown",
            "warnings": ["Column has no rows."],
        }

    if name_clean in ID_NAME_HINTS or name_clean.endswith("_id"):
        return {
            "semantic_type": "id_like",
            "measurement_scale": "identifier",
            "warnings": ["Column name suggests an identifier."],
        }

    if n_non_missing == 0:
        return {
            "semantic_type": "unknown",
            "measurement_scale": "unknown",
            "warnings": ["Column is entirely missing."],
        }

    if is_datetime64_any_dtype(series):
        return {
            "semantic_type": "datetime",
            "measurement_scale": "datetime",
            "warnings": warnings,
        }

    if is_bool_dtype(series):
        return {
            "semantic_type": "binary_categorical",
            "measurement_scale": "binary",
            "warnings": warnings,
        }

    if n_unique == 1:
        return {
            "semantic_type": "constant",
            "measurement_scale": "unknown",
            "warnings": ["Column has only one observed value."],
        }

    if n_unique == 2:
        return {
            "semantic_type": "binary_categorical",
            "measurement_scale": "binary",
            "warnings": warnings,
        }

    if is_numeric_dtype(series):
        # Numeric columns with very high uniqueness may be identifiers.
        if unique_rate > 0.95 and n_unique > 20:
            return {
                "semantic_type": "id_like",
                "measurement_scale": "identifier",
                "warnings": ["Numeric column has very high uniqueness; may be an identifier."],
            }

        # Important:
        # Low unique count alone is not enough to call a numeric column discrete.
        # In small datasets, continuous numeric variables may naturally have only
        # a few observed values.
        #
        # Use integer-like values as stronger evidence for discrete/count/ordinal.
        try:
            numeric_values = pd.to_numeric(non_missing, errors="coerce").dropna()
            is_integer_like = bool((numeric_values % 1 == 0).all())
        except Exception:
            is_integer_like = False

        if n_unique <= 10 and is_integer_like:
            return {
                "semantic_type": "discrete_numeric",
                "measurement_scale": "count_or_ordinal",
                "warnings": warnings,
            }

        if n_unique <= 10 and not is_integer_like:
            return {
                "semantic_type": "continuous_numeric",
                "measurement_scale": "continuous",
                "warnings": warnings + [
                    "Numeric column has few observed unique values; verify whether it is truly continuous or ordinal."
                ],
            }

        return {
            "semantic_type": "continuous_numeric",
            "measurement_scale": "continuous",
            "warnings": warnings,
        }

    # Object/string columns.
    sample_values = non_missing.astype(str)
    avg_len = float(sample_values.str.len().mean()) if len(sample_values) else 0.0

    if unique_rate > 0.8 and avg_len > 20:
        return {
            "semantic_type": "text",
            "measurement_scale": "text",
            "warnings": ["High-cardinality text-like column."],
        }

    return {
        "semantic_type": "nominal_categorical",
        "measurement_scale": "categorical",
        "warnings": warnings,
    }