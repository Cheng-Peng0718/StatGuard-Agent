# core/analysis_tool_plugins/plugins/inspect_sql_schema.py

from __future__ import annotations

from typing import Any

from core.analysis_tool_plugins.base import AnalysisToolPlugin, ArgumentSchema
from core.analysis_tool_plugins.registry import register_plugin
from core.analysis_tool_plugins.shared.sql_utils import (
    connect_duckdb_read_only,
    normalize_database_path,
)


def _execute(context) -> dict[str, Any]:
    arguments = (
            getattr(context, "arguments", None)
            or getattr(context, "args", None)
            or {}
    )
    database_path = arguments.get("database_path")
    workspace_dir = getattr(context, "workspace_dir", None)

    resolved_path = normalize_database_path(database_path, workspace_dir=workspace_dir)

    try:
        con = connect_duckdb_read_only(resolved_path)

        tables_df = con.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'main'
            ORDER BY table_name
            """
        ).fetchdf()

        tables = tables_df["table_name"].tolist()

        table_summaries = []

        for table in tables:
            columns_df = con.execute(
                """
                SELECT column_name, data_type, is_nullable
                FROM information_schema.columns
                WHERE table_schema = 'main'
                  AND table_name = ?
                ORDER BY ordinal_position
                """,
                [table],
            ).fetchdf()

            try:
                row_count = con.execute(f'SELECT COUNT(*) AS n FROM "{table}"').fetchone()[0]
            except Exception:
                row_count = None

            table_summaries.append(
                {
                    "table_name": table,
                    "row_count": int(row_count) if row_count is not None else None,
                    "columns": columns_df.to_dict(orient="records"),
                }
            )

        con.close()

        compact_schema_parts = []

        for table_summary in table_summaries:
            table_name = table_summary.get("table_name")
            columns = table_summary.get("columns", [])

            column_names = [
                col.get("column_name")
                for col in columns
                if isinstance(col, dict) and col.get("column_name")
            ]

            compact_schema_parts.append(
                f"{table_name}({', '.join(column_names)})"
            )

        compact_schema = "; ".join(compact_schema_parts)

        return {
            "status": "ok",
            "message": (
                f"Inspected SQL schema for {len(tables)} table(s). "
                f"Schema: {compact_schema}"
            ),
            "recoverable": False,
            "details": {
                "database_path": resolved_path,
                "n_tables": len(tables),
                "tables": table_summaries,
                "compact_schema": compact_schema,
            },
            "artifacts": [],
        }

    except Exception as exc:
        return {
            "status": "failed",
            "error_code": "SQL_SCHEMA_INSPECTION_FAILED",
            "message": f"Failed to inspect SQL schema: {exc}",
            "recoverable": True,
            "details": {
                "database_path": resolved_path,
                "exception_type": type(exc).__name__,
                "error_message": str(exc),
            },
            "artifacts": [],
        }


inspect_sql_schema_plugin = AnalysisToolPlugin(
    tool_name="inspect_sql_schema",
    display_name="Inspect SQL Database Schema",
    execute=_execute,
    argument_schema=ArgumentSchema(
        required={
            "database_path": str,
        },
        optional={},
    ),
    requires_confirmation=False,
)

register_plugin(inspect_sql_schema_plugin)