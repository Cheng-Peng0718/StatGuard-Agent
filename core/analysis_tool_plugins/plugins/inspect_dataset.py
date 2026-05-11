from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin


MISSING_TOKENS = {
    "", " ", "na", "n/a", "nan", "null", "none", "missing", "unknown", "unk",
    "?", "-", "--", ".", "inf", "+inf", "-inf", "infinity", "+infinity", "-infinity",
    "NA", "N/A", "NaN", "NULL", "None", "Missing", "Unknown",
}


def _ok(message: str, details: Dict[str, Any], artifacts=None):
    return {
        "status": "ok",
        "message": message,
        "recoverable": False,
        "details": details or {},
        "artifacts": artifacts or [],
    }


def _failed(error_code: str, message: str, exc: Exception):
    return {
        "status": "failed",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
        },
        "artifacts": [],
    }


def _normalize_missing_tokens(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    lower_missing = {str(x).strip().lower() for x in MISSING_TOKENS}

    for col in df.columns:
        if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
            def norm(x):
                if isinstance(x, str):
                    lx = x.strip().lower()
                    if lx in lower_missing:
                        return np.nan
                    return x.strip()
                return x

            df[col] = df[col].map(norm)

    return df


def _replace_inf(df: pd.DataFrame) -> pd.DataFrame:
    return df.copy().replace([np.inf, -np.inf], np.nan)


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _replace_inf(_normalize_missing_tokens(df))


def _count_inf(df: pd.DataFrame) -> int:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    if not numeric_cols:
        return 0

    try:
        return int(np.isinf(df[numeric_cols].to_numpy(dtype=float)).sum())
    except Exception:
        return 0


def _infer_column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "categorical"

    if pd.api.types.is_numeric_dtype(s):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    return "categorical"


def _is_id_like(col: str, s: pd.Series) -> bool:
    name = str(col).lower().strip()

    if name in {"id", "index", "row", "record", "case", "subject_id", "student_id"}:
        return True

    if name.endswith("_id") or name.endswith("id"):
        return True

    non_missing = s.dropna()

    if len(non_missing) == 0:
        return False

    unique_ratio = non_missing.nunique() / max(len(non_missing), 1)

    return bool(unique_ratio > 0.95 and len(non_missing) > 20)


def execute_inspect_dataset(context) -> Dict[str, Any]:
    """
    Inspect dataset shape, column types, missingness, infinity counts, and simple column profiles.
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "status": "blocked",
                "error_code": "INVALID_DATAFRAME",
                "message": "context.load_df() did not return a valid pandas DataFrame.",
                "recoverable": True,
                "details": {},
                "artifacts": [],
            }

        std = _standardize_dataframe(df)

        columns = []

        for col in std.columns:
            s = std[col]

            columns.append({
                "name": str(col),
                "dtype": str(s.dtype),
                "kind": _infer_column_kind(s),
                "missing_count": int(s.isna().sum()),
                "missing_rate": round(float(s.isna().mean()), 6),
                "unique_count": int(s.nunique(dropna=True)),
                "id_like": bool(_is_id_like(col, s)),
            })

        return _ok(
            "Dataset inspection completed.",
            {
                "shape": {
                    "rows": int(std.shape[0]),
                    "columns": int(std.shape[1]),
                },
                "total_missing": int(std.isna().sum().sum()),
                "total_inf": _count_inf(std),
                "columns": columns,
            },
        )

    except Exception as e:
        return _failed(
            "INSPECT_DATASET_EXCEPTION",
            "Dataset inspection failed.",
            e,
        )


def extract_inspect_dataset(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Dataset Inspection"

    shape = payload.get("shape", {})
    rows = shape.get("rows") if isinstance(shape, dict) else None
    cols = shape.get("columns") if isinstance(shape, dict) else None

    metrics = compact_dict({
        "rows": rows,
        "columns": cols,
        "total_missing": payload.get("total_missing"),
        "total_inf": payload.get("total_inf"),
    })

    tables: Dict[str, Any] = {}

    column_profiles = payload.get("columns", [])
    if column_profiles:
        tables["columns"] = column_profiles

    metadata: Dict[str, Any] = {}

    summary = "Inspected dataset shape, column types, missingness, non-finite values, and simple column profiles."

    return title, summary, metrics, tables, metadata


INSPECT_DATASET_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "rows": "Rows",
            "columns": "Columns",
            "total_missing": "Total missing values",
            "total_inf": "Total infinite values",
        },
        formatters={
            "total_missing": lambda x: format_number(x, digits=0),
            "total_inf": lambda x: format_number(x, digits=0),
        },
        order=[
            "rows",
            "columns",
            "total_missing",
            "total_inf",
        ],
    )
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="inspect_dataset",
    display_name="Dataset Inspection",
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={},
        optional={},
        column_args=[],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_inspect_dataset,
    extractor=extract_inspect_dataset,
    guardrail_evaluators=[],
    display_config=INSPECT_DATASET_DISPLAY,
))