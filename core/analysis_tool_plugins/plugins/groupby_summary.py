from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from core.analysis_tool_plugins.base import AnalysisToolPlugin, ArgumentSchema
from core.analysis_tool_plugins.registry import register_plugin


DEFAULT_AGG_FUNCS = ["count", "sum", "mean", "median", "min", "max"]


def _get_arguments(context) -> dict[str, Any]:
    return (
        getattr(context, "arguments", None)
        or getattr(context, "args", None)
        or {}
    )


def _find_active_data_version(context) -> dict[str, Any] | None:
    active_id = getattr(context, "active_data_version_id", None)
    data_versions = getattr(context, "data_versions", []) or []

    if not active_id:
        return None

    for version in data_versions:
        if isinstance(version, dict) and version.get("version_id") == active_id:
            return version

    return None


def _load_active_dataframe(context) -> pd.DataFrame:
    version = _find_active_data_version(context)

    if version is None:
        raise FileNotFoundError(
            "No active DataFrame data version is available. "
            "Upload a dataset or materialize a SQL query result first."
        )

    path = version.get("path")

    if not path:
        raise FileNotFoundError(
            f"Active data version `{version.get('version_id')}` has no path."
        )

    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"Active data file does not exist: {path}")

    suffix = path_obj.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path_obj)

    if suffix == ".csv":
        return pd.read_csv(path_obj)

    raise ValueError(f"Unsupported active data file type: {suffix}")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, str):
        return [value]
    return [value]


def _validate_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col not in df.columns]


def _normalize_agg_funcs(value: Any) -> list[str]:
    funcs = _as_list(value) or DEFAULT_AGG_FUNCS

    normalized = []
    allowed = {"count", "sum", "mean", "median", "min", "max", "std", "nunique"}

    for func in funcs:
        func = str(func).strip().lower()

        if func in allowed and func not in normalized:
            normalized.append(func)

    return normalized or DEFAULT_AGG_FUNCS


def _flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    flat = df.copy()

    flat.columns = [
        "_".join(str(part) for part in col if str(part))
        if isinstance(col, tuple)
        else str(col)
        for col in flat.columns
    ]

    return flat


def _execute(context) -> dict[str, Any]:
    arguments = _get_arguments(context)

    group_cols = _as_list(arguments.get("group_cols"))
    value_col = arguments.get("value_col")
    agg_funcs = _normalize_agg_funcs(arguments.get("agg_funcs"))
    sort_by = arguments.get("sort_by")
    ascending = bool(arguments.get("ascending", False))
    top_n = arguments.get("top_n")

    try:
        top_n = int(top_n) if top_n is not None else None
    except Exception:
        top_n = None

    if not group_cols:
        return {
            "status": "blocked",
            "error_code": "MISSING_GROUP_COLUMNS",
            "message": "group_cols must contain at least one grouping column.",
            "recoverable": True,
            "details": {
                "received_arguments": arguments,
            },
            "artifacts": [],
        }

    if not value_col or not isinstance(value_col, str):
        return {
            "status": "blocked",
            "error_code": "MISSING_VALUE_COLUMN",
            "message": "value_col must be provided as a column name.",
            "recoverable": True,
            "details": {
                "received_arguments": arguments,
            },
            "artifacts": [],
        }

    try:
        df = _load_active_dataframe(context)

        missing = _validate_columns(df, group_cols + [value_col])

        if missing:
            return {
                "status": "blocked",
                "error_code": "COLUMN_NOT_FOUND",
                "message": "One or more requested columns do not exist in the active dataset.",
                "recoverable": True,
                "details": {
                    "missing_columns": missing,
                    "available_columns": list(df.columns),
                    "received_arguments": arguments,
                },
                "artifacts": [],
            }

        if not pd.api.types.is_numeric_dtype(df[value_col]):
            return {
                "status": "blocked",
                "error_code": "VALUE_COLUMN_NOT_NUMERIC",
                "message": f"value_col `{value_col}` must be numeric for groupby_summary.",
                "recoverable": True,
                "details": {
                    "value_col": value_col,
                    "dtype": str(df[value_col].dtype),
                    "available_columns": list(df.columns),
                },
                "artifacts": [],
            }

        working = df[group_cols + [value_col]].copy()
        before_rows = int(len(working))
        working = working.dropna(subset=group_cols + [value_col])
        rows_used = int(len(working))
        rows_dropped = before_rows - rows_used

        if working.empty:
            return {
                "status": "blocked",
                "error_code": "NO_COMPLETE_ROWS",
                "message": "No complete rows are available for the requested groupby summary.",
                "recoverable": True,
                "details": {
                    "group_cols": group_cols,
                    "value_col": value_col,
                    "rows_before_dropna": before_rows,
                    "rows_used": rows_used,
                },
                "artifacts": [],
            }

        grouped = (
            working
            .groupby(group_cols, dropna=False)[value_col]
            .agg(agg_funcs)
            .reset_index()
        )

        grouped = _flatten_columns(grouped)

        # Rename aggregation columns to make report clearer.
        rename_map = {}
        for func in agg_funcs:
            if func in grouped.columns:
                rename_map[func] = f"{func}_{value_col}"

        grouped = grouped.rename(columns=rename_map)

        if sort_by is None:
            preferred_sort = f"sum_{value_col}"
            fallback_sort = f"mean_{value_col}"

            if preferred_sort in grouped.columns:
                sort_by = preferred_sort
            elif fallback_sort in grouped.columns:
                sort_by = fallback_sort

        if sort_by in grouped.columns:
            grouped = grouped.sort_values(by=sort_by, ascending=ascending)

        if top_n is not None and top_n > 0:
            grouped = grouped.head(top_n)

        result_rows = grouped.to_dict(orient="records")

        summary = (
            f"Computed groupby summary for `{value_col}` grouped by "
            f"{', '.join(group_cols)}. "
            f"Returned {len(result_rows)} group(s) using {rows_used} complete row(s)."
        )

        return {
            "status": "ok",
            "message": summary,
            "recoverable": False,
            "details": {
                "group_cols": group_cols,
                "value_col": value_col,
                "agg_funcs": agg_funcs,
                "sort_by": sort_by,
                "ascending": ascending,
                "top_n": top_n,
                "rows_before_dropna": before_rows,
                "rows_used": rows_used,
                "rows_dropped_due_to_missing": rows_dropped,
                "n_groups": int(len(result_rows)),
                "columns": list(grouped.columns),
                "rows": result_rows,
            },
            "artifacts": [],
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_code": "GROUPBY_SUMMARY_FAILED",
            "message": f"groupby_summary failed: {exc}",
            "recoverable": True,
            "details": {
                "exception_type": type(exc).__name__,
                "error_message": str(exc),
                "received_arguments": arguments,
            },
            "artifacts": [],
        }


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="groupby_summary",
    display_name="Groupby Summary",
    evidence_categories=["group_summary", "descriptive_comparison"],
    description=(
        "Compute descriptive grouped aggregations such as count, sum, mean, "
        "median, min, and max by one or more grouping columns."
    ),
    usage_guidance=(
        "Use this tool only for descriptive aggregation questions, such as "
        "'summarize average revenue by segment and region' or "
        "'show total sales by category'. "
        "Do not use this tool when the user asks whether a numeric outcome "
        "statistically differs across groups. "
        "This tool does not produce group_comparison evidence. "
        "For inferential comparison questions, use statistical_group_comparison."
    ),
    use_when=[
        "The user asks to compare a numeric column across categories, groups, segments, regions, or cohorts.",
        "The user asks for total_revenue by region, total_revenue by segment, or customer value by group.",
        "An active DataFrame dataset exists and contains both the grouping columns and numeric value column.",
    ],
    do_not_use_when=[
        "No active DataFrame dataset exists.",
        "The user is asking to inspect or query a SQL database directly.",
        "The value column is non-numeric.",
        "The requested columns are not in the active dataset.",
    ],
    requires_data_source="dataframe",
    produces_active_dataset=False,
    examples=[
        {
            "user_request": "Compare total_revenue by region in the active dataset.",
            "arguments": {
                "group_cols": ["region"],
                "value_col": "total_revenue",
                "agg_funcs": ["count", "sum", "mean", "median"],
                "sort_by": "sum_total_revenue",
                "ascending": False,
            },
        },
        {
            "user_request": "Compare total_revenue by region and segment.",
            "arguments": {
                "group_cols": ["region", "segment"],
                "value_col": "total_revenue",
                "agg_funcs": ["count", "sum", "mean"],
            },
        },
    ],
    execute=_execute,
    argument_schema=ArgumentSchema(
        required={
            "group_cols": list,
            "value_col": str,
        },
        optional={
            "agg_funcs": list,
            "sort_by": str,
            "ascending": bool,
            "top_n": int,
        },
        column_args=["value_col"],
        column_list_args=["group_cols"],
    ),
    requires_confirmation=False,
))