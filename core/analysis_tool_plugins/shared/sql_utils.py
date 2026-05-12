# core/analysis_tool_plugins/shared/sql_utils.py

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import sqlglot
from sqlglot import exp
import hashlib


FORBIDDEN_SQL_NODES = (
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.TruncateTable,
    exp.Command,
)


def normalize_database_path(database_path: str, workspace_dir: str | None = None) -> str:
    """
    Resolve database_path safely.

    For now this supports:
    - absolute paths
    - paths relative to workspace_dir
    - paths relative to project working directory

    Do not create the file here. Read-only SQL tools should fail clearly
    if the path does not exist.
    """
    if not database_path or not isinstance(database_path, str):
        raise ValueError("database_path must be a non-empty string.")

    raw = Path(database_path)

    if raw.is_absolute():
        resolved = raw
    elif workspace_dir:
        candidate = Path(workspace_dir) / raw
        resolved = candidate if candidate.exists() else raw
    else:
        resolved = raw

    if not resolved.exists():
        raise FileNotFoundError(f"Database file does not exist: {resolved}")

    return str(resolved)


def connect_duckdb_read_only(database_path: str):
    """
    Open DuckDB database in read-only mode when possible.
    """
    return duckdb.connect(database_path, read_only=True)


def validate_read_only_sql(query: str) -> tuple[bool, str | None]:
    """
    Validate that a query is read-only.

    We allow SELECT / WITH style analytical queries and reject mutation or DDL.
    This is intentionally parser-based rather than keyword-routing logic.
    """
    if not isinstance(query, str) or not query.strip():
        return False, "SQL query must be a non-empty string."

    try:
        expressions = sqlglot.parse(query, read="duckdb")
    except Exception as exc:
        return False, f"SQL parse error: {exc}"

    if len(expressions) != 1:
        return False, "Only one SQL statement is allowed."

    expression = expressions[0]

    for node in expression.walk():
        if isinstance(node, FORBIDDEN_SQL_NODES):
            return False, f"SQL statement contains forbidden operation: {type(node).__name__}"

    # Root should be a read-style expression. SELECT and UNION are enough for MVP.
    allowed_roots = (
        exp.Select,
        exp.Union,
        exp.Except,
        exp.Intersect,
    )

    if not isinstance(expression, allowed_roots):
        return False, f"Only SELECT/WITH read-only queries are allowed. Got: {type(expression).__name__}"

    return True, None


def limit_query(query: str, max_rows: int) -> str:
    """
    Wrap the user query and apply a hard row limit.

    This avoids modifying the user's query logic directly.
    """
    query = query.strip().rstrip(";")
    return f"SELECT * FROM ({query}) AS _agent_sql_result LIMIT {int(max_rows)}"


def dataframe_preview_payload(df, *, max_preview_rows: int = 20) -> dict[str, Any]:
    preview_df = df.head(max_preview_rows)

    return {
        "n_rows_returned": int(len(df)),
        "n_cols_returned": int(len(df.columns)),
        "columns": list(df.columns),
        "preview": preview_df.to_dict(orient="records"),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

def sql_query_hash(query: str) -> str:
    normalized = " ".join((query or "").strip().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def strip_sql_semicolon(query: str) -> str:
    return (query or "").strip().rstrip(";")


def count_query_rows(query: str) -> str:
    query = strip_sql_semicolon(query)
    return f"SELECT COUNT(*) AS n_rows FROM ({query}) AS _agent_count_result"


def materialization_query(query: str) -> str:
    query = strip_sql_semicolon(query)
    return f"SELECT * FROM ({query}) AS _agent_materialized_result"