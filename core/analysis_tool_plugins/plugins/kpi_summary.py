from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    TableDisplayConfig,
    compact_dict,
    format_number,
)
from core.analysis_tool_plugins.registry import register_plugin


def _get_arguments(context) -> dict[str, Any]:
    return (
        getattr(context, "arguments", None)
        or getattr(context, "args", None)
        or {}
    )


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
    if hasattr(context, "load_df"):
        df = context.load_df()
        if isinstance(df, pd.DataFrame):
            return df

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


def _clean_float(value: Any) -> float | None:
    if pd.isna(value):
        return None

    try:
        numeric = float(value)
    except Exception:
        return None

    if not np.isfinite(numeric):
        return None

    return numeric


def _is_identifier_like_column(col: str) -> bool:
    lower = str(col).lower()

    identifier_patterns = [
        "id",
        "_id",
        "id_",
        "customer_id",
        "order_id",
        "product_id",
        "user_id",
        "account_id",
        "transaction_id",
        "record_id",
    ]

    if lower in identifier_patterns:
        return True

    if lower.endswith("_id"):
        return True

    if lower.startswith("id_"):
        return True

    if lower == "id":
        return True

    return False


def _infer_metric_columns(df: pd.DataFrame, limit: int = 8) -> list[str]:
    numeric_cols = [
        str(col)
        for col in df.select_dtypes(include=[np.number]).columns
    ]

    if not numeric_cols:
        return []

    metric_candidates = [
        col for col in numeric_cols
        if not _is_identifier_like_column(col)
    ]

    if not metric_candidates:
        return []

    priority_terms = [
        "revenue",
        "sales",
        "profit",
        "amount",
        "price",
        "cost",
        "margin",
        "order",
        "quantity",
        "qty",
        "spend",
        "value",
        "score",
        "rate",
    ]

    def score(col: str) -> tuple[int, str]:
        lower = col.lower()
        matched = any(term in lower for term in priority_terms)
        return (0 if matched else 1, lower)

    return sorted(metric_candidates, key=score)[:limit]

def _infer_id_columns(df: pd.DataFrame, limit: int = 5) -> list[str]:
    columns = [str(col) for col in df.columns]

    id_candidates = [
        col for col in columns
        if _is_identifier_like_column(col)
    ]

    return id_candidates[:limit]

def _validate_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [col for col in columns if col not in df.columns]


def execute_kpi_summary(context) -> Dict[str, Any]:
    arguments = _get_arguments(context)

    requested_metric_columns = _as_list(arguments.get("metric_columns"))
    id_columns = _as_list(arguments.get("id_columns"))

    try:
        max_metrics = int(arguments.get("max_metrics", 8))
    except Exception:
        max_metrics = 8

    max_metrics = max(1, min(max_metrics, 20))

    try:
        df = _load_active_dataframe(context)

        if df is None or not isinstance(df, pd.DataFrame):
            return {
                "status": "blocked",
                "error_code": "INVALID_DATAFRAME",
                "message": "The active data source did not return a valid pandas DataFrame.",
                "recoverable": True,
                "details": {},
                "artifacts": [],
            }

        df = df.copy().replace([np.inf, -np.inf], np.nan)

        if df.empty:
            return {
                "status": "blocked",
                "error_code": "EMPTY_DATAFRAME",
                "message": "The active dataset is empty, so KPI summary cannot be computed.",
                "recoverable": True,
                "details": {"n_rows": 0, "n_columns": int(df.shape[1])},
                "artifacts": [],
            }

        if requested_metric_columns:
            metric_columns = [str(col) for col in requested_metric_columns]
        else:
            metric_columns = _infer_metric_columns(df, limit=max_metrics)

        if id_columns:
            id_columns = [str(col) for col in id_columns]
        else:
            id_columns = _infer_id_columns(df)

        missing = _validate_columns(df, metric_columns + id_columns)
        if missing:
            return {
                "status": "blocked",
                "error_code": "COLUMN_NOT_FOUND",
                "message": "One or more requested KPI columns do not exist in the active dataset.",
                "recoverable": True,
                "details": {
                    "missing_columns": missing,
                    "available_columns": list(df.columns),
                    "received_arguments": arguments,
                },
                "artifacts": [],
            }

        non_numeric = [
            col for col in metric_columns
            if not pd.api.types.is_numeric_dtype(df[col])
        ]
        if non_numeric:
            return {
                "status": "blocked",
                "error_code": "METRIC_COLUMN_NOT_NUMERIC",
                "message": "All metric_columns must be numeric for kpi_summary.",
                "recoverable": True,
                "details": {
                    "non_numeric_columns": non_numeric,
                    "available_columns": list(df.columns),
                    "received_arguments": arguments,
                },
                "artifacts": [],
            }

        if not metric_columns:
            return {
                "status": "blocked",
                "error_code": "NO_NUMERIC_METRICS",
                "message": "No numeric columns are available for KPI summary.",
                "recoverable": True,
                "details": {
                    "available_columns": list(df.columns),
                    "received_arguments": arguments,
                },
                "artifacts": [],
            }

        kpi_rows = []

        for col in metric_columns:
            series = pd.to_numeric(df[col], errors="coerce")
            non_missing = series.dropna()

            kpi_rows.append({
                "metric": col,
                "non_missing_count": int(non_missing.shape[0]),
                "missing_count": int(series.isna().sum()),
                "total": _clean_float(non_missing.sum()) if not non_missing.empty else None,
                "mean": _clean_float(non_missing.mean()) if not non_missing.empty else None,
                "median": _clean_float(non_missing.median()) if not non_missing.empty else None,
                "min": _clean_float(non_missing.min()) if not non_missing.empty else None,
                "max": _clean_float(non_missing.max()) if not non_missing.empty else None,
            })

        distinct_count_rows = []

        for col in id_columns:
            distinct_count_rows.append({
                "column": col,
                "distinct_count": int(df[col].nunique(dropna=True)),
                "missing_count": int(df[col].isna().sum()),
            })

        summary = (
            f"Computed KPI summary for {len(metric_columns)} numeric metric(s) "
            f"using {int(df.shape[0])} row(s)."
        )

        return {
            "status": "ok",
            "message": summary,
            "recoverable": False,
            "details": {
                "n_rows": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
                "metric_columns": metric_columns,
                "id_columns": id_columns,
                "kpi_rows": kpi_rows,
                "distinct_count_rows": distinct_count_rows,
            },
            "artifacts": [],
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_code": "KPI_SUMMARY_FAILED",
            "message": f"kpi_summary failed: {exc}",
            "recoverable": True,
            "details": {
                "exception_type": type(exc).__name__,
                "error_message": str(exc),
                "received_arguments": arguments,
            },
            "artifacts": [],
        }


def extract_kpi_summary(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    metric_columns = payload.get("metric_columns", []) or []
    id_columns = payload.get("id_columns", []) or []

    title = "KPI Summary"

    summary = (
        f"Computed business KPI summary for {len(metric_columns)} numeric metric(s) "
        f"across {payload.get('n_rows')} row(s)."
    )

    metrics = compact_dict({
        "n_rows": payload.get("n_rows"),
        "n_columns": payload.get("n_columns"),
        "n_metric_columns": len(metric_columns),
        "n_id_columns": len(id_columns),
    })

    tables: Dict[str, Any] = {}

    if payload.get("kpi_rows"):
        tables["kpi_rows"] = payload.get("kpi_rows")

    if payload.get("distinct_count_rows"):
        tables["distinct_count_rows"] = payload.get("distinct_count_rows")

    metadata = {
        "metric_columns": metric_columns,
        "id_columns": id_columns,
    }

    return title, summary, metrics, tables, metadata


KPI_SUMMARY_DISPLAY = DisplayConfig(
    tables={
        "kpi_rows": TableDisplayConfig(
            column_labels={
                "metric": "Metric",
                "non_missing_count": "Non-missing",
                "missing_count": "Missing",
                "total": "Total",
                "mean": "Mean",
                "median": "Median",
                "min": "Min",
                "max": "Max",
            },
            column_order=[
                "metric",
                "non_missing_count",
                "missing_count",
                "total",
                "mean",
                "median",
                "min",
                "max",
            ],
            column_formatters={
                "total": format_number,
                "mean": format_number,
                "median": format_number,
                "min": format_number,
                "max": format_number,
            },
        ),
        "distinct_count_rows": TableDisplayConfig(
            column_labels={
                "column": "Column",
                "distinct_count": "Distinct count",
                "missing_count": "Missing",
            },
            column_order=["column", "distinct_count", "missing_count"],
        ),
    }
)


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="kpi_summary",
    display_name="KPI Summary",
    description="Compute high-level business KPI summaries for numeric metrics in the active DataFrame dataset.",
    usage_guidance=(
        "Use this near the beginning of a business or data analysis task to summarize key metrics such as "
        "revenue, sales, orders, quantity, cost, profit, or customer value. If metric_columns are not provided, "
        "the tool will infer likely numeric business metrics."
    ),
    use_when=[
        "The user asks for a business overview, KPI summary, dashboard-style summary, or key metrics.",
        "The user asks to analyze revenue, sales, orders, customer value, profit, cost, or other numeric business metrics.",
        "An active DataFrame dataset exists after uploading data or materializing a SQL query result.",
    ],
    do_not_use_when=[
        "No active DataFrame dataset exists.",
        "The user is asking to inspect a SQL database schema before materializing an analysis dataset.",
        "The user requests a grouped comparison; use groupby_summary instead.",
        "The user requests statistical modeling or regression; use the appropriate modeling tool instead.",
    ],
    requires_data_source="dataframe",
    produces_active_dataset=False,
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={},
        optional={
            "metric_columns": list,
            "id_columns": list,
            "max_metrics": int,
        },
        column_args=[],
        column_list_args=["metric_columns", "id_columns"],
        allow_all_columns=False,
    ),
    execute=execute_kpi_summary,
    extractor=extract_kpi_summary,
    guardrail_evaluators=[],
    display_config=KPI_SUMMARY_DISPLAY,
    examples=[
        {
            "user_request": "Give me a KPI summary for this dataset.",
            "arguments": {
                "metric_columns": ["total_revenue", "number_of_orders"],
                "id_columns": ["customer_id"],
            },
        },
        {
            "user_request": "Analyze the active revenue dataset and start with key metrics.",
            "arguments": {
                "metric_columns": ["total_revenue"],
                "id_columns": ["customer_id"],
            },
        },
        {
            "user_request": "What are the main business KPIs in this materialized SQL result?",
            "arguments": {},
        },
    ],
))