from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import AnalysisToolPlugin
from core.analysis_tool_plugins.arguments import ArgumentSchema
from core.analysis_tool_plugins.roles import VariableRoleSpec
from core.analysis_tool_plugins.display import (
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
from core.analysis_tool_plugins.planning_contracts import PlanningMetadata

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


def _count_inf_for_series(s: pd.Series) -> int:
    if not pd.api.types.is_numeric_dtype(s):
        return 0

    try:
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return int(np.isinf(arr).sum())
    except Exception:
        return 0


def execute_missingness_report(context) -> Dict[str, Any]:
    """
    Report missingness and non-finite values by selected columns.

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

        rows = []

        for col in cols:
            s = df[col]

            rows.append({
                "column": str(col),
                "dtype": str(s.dtype),
                "missing_count": int(s.isna().sum()),
                "missing_rate": round(float(s.isna().mean()), 6),
                "inf_count": _count_inf_for_series(s),
                "unique_count": int(s.nunique(dropna=True)),
            })

        rows.sort(
            key=lambda r: (r["missing_rate"], r["inf_count"]),
            reverse=True,
        )

        return _ok(
            "Missingness report completed.",
            {
                "requested_columns": cols_arg,
                "resolved_columns": cols,
                "n_columns_reported": int(len(rows)),
                "shape": {
                    "rows": int(df.shape[0]),
                    "columns": int(df.shape[1]),
                },
                "columns": rows,
            },
        )

    except Exception as e:
        return _failed(
            "MISSINGNESS_REPORT_EXCEPTION",
            "Missingness report failed.",
            e,
        )


def extract_missingness_report(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Missingness Report"

    shape = payload.get("shape", {})
    n_rows = shape.get("rows") if isinstance(shape, dict) else None
    n_columns = shape.get("columns") if isinstance(shape, dict) else None

    metrics = compact_dict({
        "n_rows": n_rows,
        "n_columns": n_columns,
        "n_columns_reported": payload.get("n_columns_reported"),
    })

    tables: Dict[str, Any] = {}

    column_rows = payload.get("columns", [])
    if column_rows:
        tables["missingness_by_column"] = column_rows

    metadata = compact_dict({
        "requested_columns": payload.get("requested_columns"),
        "resolved_columns": payload.get("resolved_columns"),
    })

    summary = "Computed missingness and non-finite values by column."

    return title, summary, metrics, tables, metadata


MISSINGNESS_REPORT_DISPLAY = DisplayConfig(
    metrics=MetricDisplayConfig(
        labels={
            "n_rows": "Rows",
            "n_columns": "Columns",
            "n_columns_reported": "Columns reported",
        },
        order=[
            "n_rows",
            "n_columns",
            "n_columns_reported",
        ],
    ),
    tables={
        "missingness_by_column": TableDisplayConfig(
            column_labels={
                "column": "Column",
                "dtype": "Data type",
                "missing_count": "Missing count",
                "missing_rate": "Missing rate",
                "inf_count": "Infinite values",
                "unique_count": "Unique values",
            },
            column_formatters={
                "missing_rate": lambda x: format_number(x, digits=4),
            },
            column_order=[
                "column",
                "dtype",
                "missing_count",
                "missing_rate",
                "inf_count",
                "unique_count",
            ],
        )
    },
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="missingness_report",
    display_name="Missingness Report",
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

    execute=execute_missingness_report,
    extractor=extract_missingness_report,
    guardrail_evaluators=[],
    display_config=MISSINGNESS_REPORT_DISPLAY,

    # Generic method/planning contract.
    method_family="eda",

    # Missingness report can run with default all-column selection.
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
                "Columns to include in the missingness report. "
                "If omitted, all columns may be inspected by default."
            ),
        ),
    ],

    planning_policy=EDA_READY_PLANNING,

    planning_metadata=PlanningMetadata(
        supported_goal_types=[
            "dataset_overview",
            "analysis_recommendation",
            "analysis_planning",
            "eda",
        ],
        planning_tags=[
            "overview",
            "missingness",
            "data_quality",
            "eda",
        ],
        default_plan_purpose="Assess missing values before recommending analyses.",
        expected_deliverables=[
            "missingness_assessment",
        ],
        plan_order=20,
    ),

    # Missingness report does not mutate data.
    mutates_data=False,
    versioning_policy=NON_MUTATING_VERSIONING,

    repair_policy=DEFAULT_LOW_RISK_REPAIR,
))