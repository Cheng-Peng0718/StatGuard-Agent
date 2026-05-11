from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
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


def _blocked(error_code: str, message: str, details=None, suggested_next_actions=None):
    result = {
        "status": "blocked",
        "error_code": error_code,
        "message": message,
        "recoverable": True,
        "details": details or {},
        "artifacts": [],
    }

    if suggested_next_actions:
        result["suggested_next_actions"] = suggested_next_actions

    return result


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


def _get_arg(context, name: str, default: Any = None) -> Any:
    try:
        return context.get_arg(name, default)
    except TypeError:
        try:
            value = context.get_arg(name)
            return default if value is None else value
        except Exception:
            return default
    except Exception:
        return default


def _select_columns(df: pd.DataFrame, cols: Any) -> List[str]:
    if cols is None or cols == "all":
        return df.columns.tolist()

    if isinstance(cols, str):
        cols = [cols]

    if not isinstance(cols, list):
        raise ValueError("columns must be 'all', a column name, or a list of column names.")

    missing_cols = [c for c in cols if c not in df.columns]

    if missing_cols:
        raise ValueError(f"Columns not found in dataset: {missing_cols}")

    return cols


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


def _standardize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    return _normalize_missing_tokens(df).replace([np.inf, -np.inf], np.nan)


def _matrix_to_table_rows(corr: pd.DataFrame) -> List[Dict[str, Any]]:
    rows = []

    for row_name in corr.index:
        row = {"variable": str(row_name)}

        for col_name in corr.columns:
            value = corr.loc[row_name, col_name]
            row[str(col_name)] = None if pd.isna(value) else round(float(value), 6)

        rows.append(row)

    return rows


def execute_correlation_matrix(context) -> Dict[str, Any]:
    """
    Compute Pearson correlation matrix for selected numeric columns.

    Args:
        columns: 'all', a column name, or a list of column names.
    """
    try:
        df = context.load_df()

        if df is None or not isinstance(df, pd.DataFrame):
            return _blocked(
                "INVALID_DATAFRAME",
                "context.load_df() did not return a valid pandas DataFrame.",
            )

        df = _standardize_dataframe(df)

        cols_arg = _get_arg(context, "columns", "all")

        try:
            cols = _select_columns(df, cols_arg)
        except ValueError as e:
            return _blocked(
                "COLUMNS_NOT_FOUND",
                str(e),
                details={
                    "requested_columns": cols_arg,
                    "available_columns": list(df.columns),
                },
                suggested_next_actions=[
                    "Inspect dataset columns and retry with valid column names."
                ],
            )

        numeric_df = (
            df[cols]
            .select_dtypes(include=[np.number])
            .replace([np.inf, -np.inf], np.nan)
            .dropna(axis=1, how="all")
        )

        numeric_cols = numeric_df.columns.tolist()

        if len(numeric_cols) < 2:
            return _blocked(
                "INSUFFICIENT_NUMERIC_COLUMNS",
                "At least two numeric columns are required for a correlation matrix.",
                details={
                    "requested_columns": cols_arg,
                    "resolved_columns": cols,
                    "numeric_columns": numeric_cols,
                },
                suggested_next_actions=[
                    "Select at least two numeric columns."
                ],
            )

        corr = numeric_df.corr(method="pearson").round(6)

        return _ok(
            "Correlation matrix completed.",
            {
                "method": "pearson",
                "requested_columns": cols_arg,
                "resolved_columns": cols,
                "numeric_columns": numeric_cols,
                "n_numeric_columns": int(len(numeric_cols)),
                "correlation_matrix": corr.to_dict(),
                "correlation_rows": _matrix_to_table_rows(corr),
            },
        )

    except Exception as e:
        return _failed(
            "CORRELATION_MATRIX_EXCEPTION",
            "Correlation matrix failed.",
            e,
        )


def extract_correlation_matrix(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Correlation Matrix"

    metrics = compact_dict({
        "method": payload.get("method"),
        "n_numeric_columns": payload.get("n_numeric_columns"),
    })

    tables: Dict[str, Any] = {}

    correlation_rows = payload.get("correlation_rows", [])
    if correlation_rows:
        tables["correlation_matrix"] = correlation_rows

    metadata = compact_dict({
        "requested_columns": payload.get("requested_columns"),
        "resolved_columns": payload.get("resolved_columns"),
        "numeric_columns": payload.get("numeric_columns"),
        "correlation_matrix_raw": payload.get("correlation_matrix"),
    })

    numeric_cols = payload.get("numeric_columns", [])
    summary = "Computed Pearson correlation matrix for selected numeric variables."

    if numeric_cols:
        summary += f" Numeric variables used: `{', '.join(str(c) for c in numeric_cols)}`."

    return title, summary, metrics, tables, metadata


CORRELATION_MATRIX_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "method": "Method",
            "n_numeric_columns": "Numeric columns used",
        },
        order=[
            "method",
            "n_numeric_columns",
        ],
    ),
    tables={
        "correlation_matrix": TableDisplayConfig(
            column_labels={
                "variable": "Variable",
            },
            column_formatters={
                # Only known static column is "variable".
                # Dynamic numeric columns will use generic formatting.
            },
            column_order=[
                "variable",
            ],
        )
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="get_correlation_matrix",
    display_name="Correlation Matrix",
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={},
        optional={
            "columns": object,
        },
        column_args=[],
        column_list_args=[
            "columns",
        ],
        allow_all_columns=True,
    ),
    execute=execute_correlation_matrix,
    extractor=extract_correlation_matrix,
    guardrail_evaluators=[],
    display_config=CORRELATION_MATRIX_DISPLAY,
))