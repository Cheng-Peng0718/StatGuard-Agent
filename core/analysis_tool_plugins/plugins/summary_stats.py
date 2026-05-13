from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd

from core.analysis_tool_plugins.base import (
    AnalysisToolPlugin,
    ArgumentSchema,
    DisplayConfig,
    compact_dict,
)
from core.analysis_tool_plugins.registry import register_plugin


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


def execute_summary_stats(context) -> Dict[str, Any]:
    """
    Compute descriptive summaries for numeric and categorical columns.
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

        df = df.copy().replace([np.inf, -np.inf], np.nan)

        numeric = df.select_dtypes(include=[np.number])
        categorical = df.select_dtypes(exclude=[np.number])

        numeric_summary = {}
        if numeric.shape[1] > 0:
            desc = numeric.describe().replace([np.inf, -np.inf], np.nan)
            numeric_summary = (
                desc.round(6)
                .where(pd.notna(desc), None)
                .to_dict()
            )

        categorical_summary = {}
        for col in categorical.columns:
            vc = categorical[col].value_counts(dropna=True).head(10)

            categorical_summary[str(col)] = {
                "missing_count": int(categorical[col].isna().sum()),
                "unique_count": int(categorical[col].nunique(dropna=True)),
                "top_values": {str(k): int(v) for k, v in vc.items()},
            }

        return _ok(
            "Summary statistics completed.",
            {
                "numeric_summary": numeric_summary,
                "categorical_summary": categorical_summary,
                "n_rows": int(df.shape[0]),
                "n_columns": int(df.shape[1]),
                "n_numeric_columns": int(numeric.shape[1]),
                "n_categorical_columns": int(categorical.shape[1]),
            },
        )

    except Exception as e:
        return _failed(
            "SUMMARY_STATS_EXCEPTION",
            "Summary statistics failed.",
            e,
        )


def extract_summary_stats(
    *,
    payload: Dict[str, Any],
    arguments: Dict[str, Any],
    default_title: str,
    default_summary: str,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    title = "Summary Statistics"

    metrics = compact_dict({
        "n_rows": payload.get("n_rows"),
        "n_columns": payload.get("n_columns"),
        "n_numeric_columns": payload.get("n_numeric_columns"),
        "n_categorical_columns": payload.get("n_categorical_columns"),
    })

    tables: Dict[str, Any] = {}

    numeric_summary = payload.get("numeric_summary", {})
    categorical_summary = payload.get("categorical_summary", {})

    if numeric_summary:
        tables["numeric_summary"] = numeric_summary

    if categorical_summary:
        tables["categorical_summary"] = categorical_summary

    metadata: Dict[str, Any] = {}

    summary = "Computed descriptive summary statistics for the active dataset."

    return title, summary, metrics, tables, metadata


SUMMARY_STATS_DISPLAY = DisplayConfig()


PLUGIN = register_plugin(AnalysisToolPlugin(
    tool_name="get_summary_stats",
    display_name="Summary Statistics",
    evidence_categories=["dataset_overview", "summary_statistics"],
    requires_confirmation=False,
    argument_schema=ArgumentSchema(
        required={},
        optional={},
        column_args=[],
        column_list_args=[],
        allow_all_columns=False,
    ),
    execute=execute_summary_stats,
    extractor=extract_summary_stats,
    guardrail_evaluators=[],
    display_config=SUMMARY_STATS_DISPLAY,
))