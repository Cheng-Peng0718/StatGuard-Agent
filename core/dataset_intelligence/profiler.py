from __future__ import annotations

from typing import Any, Dict, List

import pandas as pd
from pandas.api.types import is_numeric_dtype

from core.dataset_intelligence.schemas import (
    ColumnProfileV2,
    DatasetProfileV2,
    DatasetSummary,
)
from core.dataset_intelligence.semantic_types import infer_semantic_type


def _safe_examples(series: pd.Series, max_examples: int = 5) -> List[Any]:
    values = series.dropna().head(max_examples).tolist()
    cleaned = []

    for value in values:
        try:
            if hasattr(value, "item"):
                value = value.item()
        except Exception:
            pass

        cleaned.append(value)

    return cleaned


def _numeric_summary(series: pd.Series) -> Dict[str, Any] | None:
    if not is_numeric_dtype(series):
        return None

    non_missing = series.dropna()

    if len(non_missing) == 0:
        return None

    return {
        "mean": float(non_missing.mean()),
        "std": float(non_missing.std()) if len(non_missing) > 1 else None,
        "min": float(non_missing.min()),
        "median": float(non_missing.median()),
        "max": float(non_missing.max()),
    }


def _categorical_summary(series: pd.Series, max_levels: int = 10) -> Dict[str, Any] | None:
    non_missing = series.dropna()

    if len(non_missing) == 0:
        return None

    counts = non_missing.value_counts(dropna=True).head(max_levels)

    return {
        "top_values": {
            str(k): int(v)
            for k, v in counts.items()
        },
        "n_displayed_levels": int(len(counts)),
    }


def profile_dataframe(
    df: pd.DataFrame,
    *,
    dataset_name: str = "uploaded_dataset",
    data_version_id: str = "unknown",
) -> DatasetProfileV2:
    columns: Dict[str, ColumnProfileV2] = {}
    warnings: List[str] = []

    n_rows, n_cols = df.shape

    if n_rows == 0:
        warnings.append("Dataset has zero rows.")

    if n_cols == 0:
        warnings.append("Dataset has zero columns.")

    for col in df.columns:
        series = df[col]

        n_missing = int(series.isna().sum())
        missing_rate = float(n_missing / n_rows) if n_rows else 0.0

        non_missing = series.dropna()
        n_unique = int(non_missing.nunique(dropna=True)) if len(non_missing) else 0
        unique_rate = float(n_unique / len(non_missing)) if len(non_missing) else 0.0

        semantic = infer_semantic_type(series, str(col))

        numeric_summary = _numeric_summary(series)
        categorical_summary = None

        if semantic["semantic_type"] in {
            "binary_categorical",
            "nominal_categorical",
            "ordinal_categorical",
            "discrete_numeric",
            "constant",
        }:
            categorical_summary = _categorical_summary(series)

        col_profile = ColumnProfileV2(
            name=str(col),
            raw_dtype=str(series.dtype),
            semantic_type=semantic["semantic_type"],
            measurement_scale=semantic["measurement_scale"],
            n_missing=n_missing,
            missing_rate=missing_rate,
            n_unique=n_unique,
            unique_rate=unique_rate,
            examples=_safe_examples(series),
            warnings=semantic.get("warnings", []),
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
        )

        columns[str(col)] = col_profile

    return DatasetProfileV2(
        dataset_name=dataset_name,
        data_version_id=data_version_id,
        n_rows=int(n_rows),
        n_cols=int(n_cols),
        columns=columns,
        warnings=warnings,
    )


def summarize_profile(profile: DatasetProfileV2) -> DatasetSummary:
    numeric_columns = []
    categorical_columns = []
    binary_columns = []
    datetime_columns = []
    text_columns = []
    id_like_columns = []

    missing_columns = {}

    for name, col in profile.columns.items():
        if col.semantic_type in {"continuous_numeric", "discrete_numeric"}:
            numeric_columns.append(name)

        if col.semantic_type in {
            "binary_categorical",
            "nominal_categorical",
            "ordinal_categorical",
        }:
            categorical_columns.append(name)

        if col.semantic_type == "binary_categorical":
            binary_columns.append(name)

        if col.semantic_type == "datetime":
            datetime_columns.append(name)

        if col.semantic_type == "text":
            text_columns.append(name)

        if col.semantic_type == "id_like":
            id_like_columns.append(name)

        if col.n_missing > 0:
            missing_columns[name] = {
                "n_missing": col.n_missing,
                "missing_rate": col.missing_rate,
            }

    return DatasetSummary(
        dataset_name=profile.dataset_name,
        data_version_id=profile.data_version_id,
        n_rows=profile.n_rows,
        n_cols=profile.n_cols,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        binary_columns=binary_columns,
        datetime_columns=datetime_columns,
        text_columns=text_columns,
        id_like_columns=id_like_columns,
        missingness_summary={
            "n_columns_with_missing": len(missing_columns),
            "columns": missing_columns,
        },
        warnings=profile.warnings,
    )