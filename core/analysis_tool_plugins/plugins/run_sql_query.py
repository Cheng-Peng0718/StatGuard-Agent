# core/analysis_tool_plugins/plugins/run_sql_query.py

from __future__ import annotations

from typing import Any

from core.analysis_tool_plugins.base import AnalysisToolPlugin, ArgumentSchema
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.sql_utils import (
    connect_duckdb_read_only,
    dataframe_preview_payload,
    limit_query,
    normalize_database_path,
    validate_read_only_sql,
)


DEFAULT_MAX_ROWS = 500


def _execute(context) -> dict[str, Any]:
    arguments = (
            getattr(context, "arguments", None)
            or getattr(context, "args", None)
            or {}
    )

    database_path = arguments.get("database_path")
    query = arguments.get("query")
    max_rows = arguments.get("max_rows", DEFAULT_MAX_ROWS)
    workspace_dir = getattr(context, "workspace_dir", None)

    try:
        max_rows = int(max_rows)
    except Exception:
        max_rows = DEFAULT_MAX_ROWS

    max_rows = max(1, min(max_rows, 5000))

    is_safe, safety_error = validate_read_only_sql(query)

    if not is_safe:
        return {
            "status": "blocked",
            "error_code": "UNSAFE_OR_INVALID_SQL",
            "message": safety_error,
            "recoverable": True,
            "details": {
                "query": query,
                "safety_error": safety_error,
            },
            "artifacts": [],
        }

    try:
        resolved_path = normalize_database_path(database_path, workspace_dir=workspace_dir)
        limited_query = limit_query(query, max_rows=max_rows)

        con = connect_duckdb_read_only(resolved_path)
        df = con.execute(limited_query).fetchdf()
        con.close()

        payload = dataframe_preview_payload(df)

        payload.update(
            {
                "database_path": resolved_path,
                "query": query,
                "executed_query": limited_query,
                "max_rows": max_rows,
            }
        )

        return {
            "status": "ok",
            "message": (
                f"SQL query executed successfully. "
                f"Returned {payload['n_rows_returned']} row(s) and {payload['n_cols_returned']} column(s)."
            ),
            "recoverable": False,
            "details": payload,
            "artifacts": [],
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_code": "SQL_QUERY_EXECUTION_FAILED",
            "message": f"SQL query execution failed: {exc}",
            "recoverable": True,
            "details": {
                "database_path": database_path,
                "query": query,
                "exception_type": type(exc).__name__,
                "error_message": str(exc),
            },
            "artifacts": [],
        }


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="run_sql_query",
    display_name="Run Safe SQL Query",
    evidence_categories=["sql_query_result"],
    description="Run a read-only SQL SELECT/WITH query against a DuckDB database and return a preview-style result.",
    usage_guidance=(
        "Use this for SQL previews, KPI summaries, trend summaries, top-N results, "
        "and business metric queries that do not need to become the active DataFrame dataset."
    ),
    use_when=[
        "The user asks for a SQL-derived metric or summary such as revenue by month, top products, or customer counts.",
        "The user wants to preview rows or check a query result.",
        "The result can be answered directly from a small SQL output.",
    ],
    do_not_use_when=[
        "The user wants the SQL query result to become the active dataset for downstream DataFrame/statistical tools; use materialize_sql_query_result instead.",
        "The user asks about the active/current/materialized DataFrame dataset.",
        "The query would mutate data.",
        "The database path is missing or looks like a placeholder.",
    ],
    requires_data_source="sql",
    produces_active_dataset=False,
    examples=[
        {
            "user_request": "Using demo_data/ecommerce_demo.duckdb, calculate monthly revenue.",
            "arguments": {
                "database_path": "demo_data/ecommerce_demo.duckdb",
                "query": "SELECT DATE_TRUNC('month', o.order_date::DATE) AS month, SUM(oi.net_revenue) AS revenue FROM orders o JOIN order_items oi ON o.order_id = oi.order_id GROUP BY 1 ORDER BY 1",
            },
        }
    ],
    execute=_execute,
    argument_schema=ArgumentSchema(
        required={
            "database_path": str,
            "query": str,
        },
        optional={
            "max_rows": int,
        },
    ),
    requires_confirmation=False,
))