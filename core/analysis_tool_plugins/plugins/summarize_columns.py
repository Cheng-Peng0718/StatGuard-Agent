from typing import Any, Dict, List, Tuple
import math

import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    VariableRoleSpec,
    DisplayConfig,
    MetricDisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin

from core.analysis_tool_plugins.policies import (
    EDA_READY_PLANNING,
    NON_MUTATING_VERSIONING,
    DEFAULT_LOW_RISK_REPAIR,
)

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


def _json_safe_value(x: Any) -> Any:
    if x is None:
        return None

    try:
        if pd.isna(x):
            return None
    except Exception:
        pass

    if isinstance(x, (np.integer,)):
        return int(x)

    if isinstance(x, (np.floating,)):
        v = float(x)
        return v if math.isfinite(v) else None

    if isinstance(x, (np.bool_,)):
        return bool(x)

    if isinstance(x, pd.Timestamp):
        return x.isoformat()

    if isinstance(x, float):
        return x if math.isfinite(x) else None

    return x


def _round_or_none(x: Any, digits: int = 6):
    try:
        v = float(x)
        if not math.isfinite(v):
            return None
        return round(v, digits)
    except Exception:
        return None


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


def _infer_column_kind(s: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(s):
        return "categorical"

    if pd.api.types.is_numeric_dtype(s):
        return "numeric"

    if pd.api.types.is_datetime64_any_dtype(s):
        return "datetime"

    return "categorical"


def _summarize_column(name: str, s: pd.Series) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "column": str(name),
        "dtype": str(s.dtype),
        "kind": _infer_column_kind(s),
        "missing_count": int(s.isna().sum()),
        "missing_rate": round(float(s.isna().mean()), 6),
        "unique_count": int(s.nunique(dropna=True)),
    }

    if pd.api.types.is_numeric_dtype(s):
        clean = pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()

        item.update({
            "n": int(len(clean)),
            "mean": _round_or_none(clean.mean()),
            "std": _round_or_none(clean.std()),
            "min": _round_or_none(clean.min()),
            "median": _round_or_none(clean.median()),
            "max": _round_or_none(clean.max()),
        })
    else:
        top_values = s.value_counts(dropna=True).head(10)
        item["top_values"] = {
            str(_json_safe_value(k)): int(v)
            for k, v in top_values.items()
        }

    return item


def execute_summarize_columns(context) -> Dict[str, Any]:
    """
    Summarize selected columns.

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

        rows = [
            _summarize_column(col, df[col])
            for col in cols
        ]

        return _ok(
            "Column summary completed.",
            {
                "requested_columns": cols_arg,
                "resolved_columns": cols,
                "n_columns_summarized": len(rows),
                "summary_rows": rows,
            },
        )

    except Exception as e:
        return _failed(
            "SUMMARIZE_COLUMNS_EXCEPTION",
            "Column summary failed.",
            e,
        )


def extract_summarize_columns(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    resolved_columns = payload.get("resolved_columns", [])

    if resolved_columns and len(resolved_columns) <= 3:
        title = "Column Summary: " + ", ".join(str(c) for c in resolved_columns)
    else:
        title = "Column Summary"

    metrics = compact_dict({
        "n_columns_summarized": payload.get("n_columns_summarized"),
    })

    tables: Dict[str, Any] = {}

    summary_rows = payload.get("summary_rows", [])
    if summary_rows:
        tables["summary_rows"] = summary_rows

    metadata = compact_dict({
        "requested_columns": payload.get("requested_columns"),
        "resolved_columns": resolved_columns,
    })

    summary = "Summarized selected columns."

    if resolved_columns:
        summary += f" Columns: `{', '.join(str(c) for c in resolved_columns)}`."

    return title, summary, metrics, tables, metadata


SUMMARIZE_COLUMNS_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "n_columns_summarized": "Columns summarized",
        },
        order=[
            "n_columns_summarized",
        ],
    ),
    tables={
        "summary_rows": TableDisplayConfig(
            column_labels={
                "column": "Column",
                "dtype": "Data type",
                "kind": "Kind",
                "missing_count": "Missing count",
                "missing_rate": "Missing rate",
                "unique_count": "Unique values",
                "n": "Valid n",
                "mean": "Mean",
                "std": "SD",
                "min": "Min",
                "median": "Median",
                "max": "Max",
                "top_values": "Top values",
            },
            column_formatters={
                "missing_rate": lambda x: format_number(x, digits=4),
                "mean": lambda x: format_number(x, digits=4),
                "std": lambda x: format_number(x, digits=4),
                "min": lambda x: format_number(x, digits=4),
                "median": lambda x: format_number(x, digits=4),
                "max": lambda x: format_number(x, digits=4),
            },
            column_order=[
                "column",
                "dtype",
                "kind",
                "missing_count",
                "missing_rate",
                "unique_count",
                "n",
                "mean",
                "std",
                "min",
                "median",
                "max",
                "top_values",
            ],
        )
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="summarize_columns",
    display_name="Column Summary",
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

    execute=execute_summarize_columns,
    extractor=extract_summarize_columns,
    guardrail_evaluators=[],
    display_config=SUMMARIZE_COLUMNS_DISPLAY,

    # Generic method/planning contract.
    method_family="eda",

    # Column summary can run with default all-column selection.
    # If columns are specified, any semantic type is acceptable.
    variable_roles=[
        VariableRoleSpec(
            role_name="columns",
            required=False,
            user_must_select=False,
            allowed_semantic_types=[
                "continuous_numeric",
                "discrete_numeric",
                "binary_categorical",
                "nominal_categorical",
                "ordinal_categorical",
                "datetime",
                "text",
                "id_like",
                "unknown",
                "constant",
            ],
            min_variables=1,
            max_variables=None,
            allow_auto_select=True,
            description=(
                "Columns to summarize. If omitted, all eligible columns may be "
                "summarized by default."
            ),
        ),
    ],

    planning_policy=EDA_READY_PLANNING,

    # Column summary does not mutate data.
    mutates_data=False,
    versioning_policy=NON_MUTATING_VERSIONING,

    repair_policy=DEFAULT_LOW_RISK_REPAIR,
))