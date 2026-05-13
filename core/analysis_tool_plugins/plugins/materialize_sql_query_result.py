from __future__ import annotations

from typing import Any

from core.analysis_tool_plugins.base import AnalysisToolPlugin, ArgumentSchema
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.sql_utils import (
    connect_duckdb_read_only,
    count_query_rows,
    materialization_query,
    normalize_database_path,
    sql_query_hash,
    validate_read_only_sql,
)
from core.data_versions import create_child_data_version, make_audit_event


DEFAULT_MAX_ROWS = 100_000
HARD_MAX_ROWS = 500_000


def _get_arguments(context) -> dict[str, Any]:
    return (
        getattr(context, "arguments", None)
        or getattr(context, "args", None)
        or {}
    )


def _execute(context) -> dict[str, Any]:
    arguments = _get_arguments(context)

    database_path = arguments.get("database_path")
    query = arguments.get("query")
    result_name = arguments.get("result_name") or "sql_query_result"
    max_rows = arguments.get("max_rows", DEFAULT_MAX_ROWS)

    workspace_dir = getattr(context, "workspace_dir", None)
    active_data_version_id = getattr(context, "active_data_version_id", None)

    try:
        max_rows = int(max_rows)
    except Exception:
        max_rows = DEFAULT_MAX_ROWS

    max_rows = max(1, min(max_rows, HARD_MAX_ROWS))

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

    if not workspace_dir:
        return {
            "status": "blocked",
            "error_code": "MISSING_WORKSPACE_DIR",
            "message": "Cannot materialize SQL result because workspace_dir is missing.",
            "recoverable": True,
            "details": {
                "database_path": database_path,
                "query": query,
            },
            "artifacts": [],
        }

    try:
        resolved_path = normalize_database_path(database_path, workspace_dir=workspace_dir)

        con = connect_duckdb_read_only(resolved_path)

        row_count_query = count_query_rows(query)
        n_rows = int(con.execute(row_count_query).fetchone()[0])

        if n_rows > max_rows:
            con.close()

            return {
                "status": "blocked",
                "error_code": "SQL_RESULT_TOO_LARGE",
                "message": (
                    f"The SQL query would materialize {n_rows:,} rows, "
                    f"which exceeds the current limit of {max_rows:,}. "
                    "Please filter, aggregate, or limit the query before materializing it."
                ),
                "recoverable": True,
                "details": {
                    "database_path": resolved_path,
                    "query": query,
                    "n_rows": n_rows,
                    "max_rows": max_rows,
                    "suggestion": "Add WHERE, GROUP BY, or LIMIT to reduce the result size.",
                },
                "artifacts": [],
            }

        df = con.execute(materialization_query(query)).fetchdf()
        con.close()

        query_hash = sql_query_hash(query)

        version = create_child_data_version(
            df=df,
            workspace_dir=workspace_dir,
            parent_version_id=active_data_version_id,
            operation="materialize_sql_query_result",
            created_by="sql_tool",
            description=f"Materialized SQL query result: {result_name}",
            metadata={
                "source_type": "sql",
                "source_database_path": resolved_path,
                "source_query": query,
                "query_hash": query_hash,
                "result_name": result_name,
                "max_rows": max_rows,
            },
        )

        audit_event = make_audit_event(
            event_type="sql_query_materialized",
            version_id=version["version_id"],
            parent_version_id=active_data_version_id,
            tool_name="materialize_sql_query_result",
            description=(
                f"Materialized SQL query result `{result_name}` "
                f"from {resolved_path} into workspace data version {version['version_id']}."
            ),
            details={
                "source_type": "sql",
                "source_database_path": resolved_path,
                "source_query": query,
                "query_hash": query_hash,
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "columns": list(df.columns),
            },
        )

        data_version_update = {
            "active_data_version_id": version["version_id"],
            "new_version": version,
            "audit_event": audit_event,
        }

        return {
            "status": "ok",
            "message": (
                f"Materialized SQL query result as active dataset "
                f"`{version['version_id']}` with {df.shape[0]} rows and {df.shape[1]} columns."
            ),
            "recoverable": False,
            "details": {
                "database_path": resolved_path,
                "query": query,
                "query_hash": query_hash,
                "result_name": result_name,
                "new_data_version_id": version["version_id"],
                "parquet_path": version["path"],
                "n_rows": int(df.shape[0]),
                "n_cols": int(df.shape[1]),
                "columns": list(df.columns),
                "data_version_update": data_version_update,
            },
            "data_version_update": data_version_update,
            "artifacts": [],
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_code": "SQL_MATERIALIZATION_FAILED",
            "message": f"Failed to materialize SQL query result: {exc}",
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
    tool_name="materialize_sql_query_result",
    display_name="Materialize SQL Query Result",
    description="Run a safe SQL SELECT/WITH query and save the result as a workspace DataFrame data version.",
    evidence_categories=["data_preparation"],
    usage_guidance=(
        "Use this when a SQL query result should become the active dataset for downstream "
        "DataFrame tools such as groupby_summary, summary statistics, regression, or plotting."
        "When materializing data for inferential statistical analysis such as t-tests, ANOVA, statistical group comparison, correlation tests, or regression, preserve the correct observational unit. Do not pre-aggregate to one row per group unless the user explicitly asks for descriptive group totals. For example, to test whether revenue differs by region, materialize customer-level or order-level rows with both region and revenue, not region-level totals only."
    ),
    use_when=[
        "The user asks to materialize, extract, prepare, or create a dataset from SQL.",
        "The user wants to analyze a SQL query result with DataFrame/statistical tools.",
        "The source is SQL-only and a selected query result is needed as the active workspace dataset.",
    ],
    do_not_use_when=[
        "The user only needs a quick KPI, preview, or summary; use run_sql_query instead.",
        "An active DataFrame dataset already exists and the user asks about the active/current/materialized dataset.",
        "The database path is missing or looks like a placeholder.",
        "The query returns an unnecessarily large table when filtering or aggregation would be more appropriate.",
    ],
    requires_data_source="sql",
    produces_active_dataset=True,
    examples=[
        {
            "user_request": "Materialize a customer-level revenue dataset from demo_data/ecommerce_demo.duckdb.",
            "arguments": {
                "database_path": "demo_data/ecommerce_demo.duckdb",
                "query": "SELECT c.customer_id, c.region, c.segment, COUNT(DISTINCT o.order_id) AS number_of_orders, SUM(oi.net_revenue) AS total_revenue FROM customers c JOIN orders o ON c.customer_id = o.customer_id JOIN order_items oi ON o.order_id = oi.order_id GROUP BY c.customer_id, c.region, c.segment",
                "result_name": "customer_level_revenue",
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
            "result_name": str,
            "max_rows": int,
        },
    ),
    requires_confirmation=False,
))

