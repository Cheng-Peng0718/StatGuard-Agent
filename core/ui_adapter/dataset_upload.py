from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.data_versions import (
    create_initial_data_version,
    make_audit_event,
)


def _semantic_type_for_series(series: pd.Series) -> str:
    non_missing = series.dropna()

    if pd.api.types.is_numeric_dtype(series):
        return "continuous_numeric"

    n_unique = int(non_missing.nunique())

    if n_unique == 2:
        return "binary_categorical"

    return "nominal_categorical"


def _series_dtype(series: pd.Series) -> str:
    return str(series.dtype)


def build_legacy_dataset_profile_from_df(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Build the legacy dataset_profile used by current verify_node.

    This is intentionally conservative and UI-safe. A future phase can replace
    this with the full DatasetProfileV2/capability-map builder.
    """
    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    columns = []

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_unique = int(series.dropna().nunique())

        columns.append({
            "name": str(col),
            "dtype": _series_dtype(series),
            "semantic_type": _semantic_type_for_series(series),
            "missing_count": n_missing,
            "missing_rate": float(n_missing / n_rows) if n_rows else 0.0,
            "n_unique": n_unique,
        })

    return {
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
    }


def make_uploaded_dataset_info(
    *,
    df: pd.DataFrame,
    filename: Optional[str],
    workspace_dir: str,
    data_version: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "filename": filename,
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "columns": [str(col) for col in df.columns],
        "workspace_dir": str(workspace_dir),
        "active_data_version_id": data_version.get("version_id"),
        "data_path": data_version.get("path"),
    }

def _measurement_scale_for_series(series: pd.Series) -> str:
    if pd.api.types.is_numeric_dtype(series):
        return "continuous"
    return "nominal"


def build_dataset_profile_v2_from_df(
    df: pd.DataFrame,
    *,
    data_version_id: str,
) -> Dict[str, Any]:
    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    columns = {}

    for col in df.columns:
        series = df[col]
        n_missing = int(series.isna().sum())
        n_unique = int(series.dropna().nunique())

        columns[str(col)] = {
            "name": str(col),
            "semantic_type": _semantic_type_for_series(series),
            "raw_dtype": _series_dtype(series),
            "measurement_scale": _measurement_scale_for_series(series),
            "n_missing": n_missing,
            "missing_rate": float(n_missing / n_rows) if n_rows else 0.0,
            "n_unique": n_unique,
            "unique_rate": float(n_unique / n_rows) if n_rows else 0.0,

            # compatibility aliases
            "dtype": _series_dtype(series),
            "missing_count": n_missing,
        }

    return {
        "data_version_id": data_version_id,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "columns": columns,
    }


def build_basic_capability_map_from_df(
    df: pd.DataFrame,
    *,
    data_version_id: str,
) -> Dict[str, Any]:
    numeric_cols = [
        str(col)
        for col in df.columns
        if pd.api.types.is_numeric_dtype(df[col])
    ]

    categorical_cols = [
        str(col)
        for col in df.columns
        if not pd.api.types.is_numeric_dtype(df[col])
    ]

    has_numeric = len(numeric_cols) > 0
    has_two_numeric = len(numeric_cols) >= 2
    has_categorical = len(categorical_cols) > 0

    return {
        "data_version_id": data_version_id,
        "capabilities": [
            {
                "tool_name": "get_summary_stats",
                "display_name": "Summary Statistics",
                "status": "ready" if has_numeric else "needs_user_choice",
                "method_family": "eda",
                "reason": "Summarize numeric columns in the uploaded dataset.",
                "required_roles": [],
                "optional_roles": ["columns"],
            },
            {
                "tool_name": "missingness_report",
                "display_name": "Missingness Report",
                "status": "ready",
                "method_family": "eda",
                "reason": "Report missing values in the uploaded dataset.",
                "required_roles": [],
                "optional_roles": ["columns"],
            },
            {
                "tool_name": "get_correlation_matrix",
                "display_name": "Correlation Matrix",
                "status": "ready" if has_two_numeric else "not_applicable",
                "method_family": "association_screening",
                "reason": "Requires at least two numeric columns.",
                "required_roles": [],
                "optional_roles": ["columns"],
            },
            {
                "tool_name": "run_multiple_regression",
                "display_name": "Linear Model",
                "status": "needs_user_choice" if has_two_numeric else "not_applicable",
                "method_family": "regression",
                "reason": "Requires user-selected outcome and predictor columns.",
                "required_roles": ["target_col", "feature_cols"],
                "optional_roles": [],
            },
            {
                "tool_name": "run_anova",
                "display_name": "One-way ANOVA",
                "status": "needs_user_choice" if has_numeric and has_categorical else "not_applicable",
                "method_family": "group_comparison",
                "reason": "Requires numeric outcome and categorical group column.",
                "required_roles": ["target_col", "group_col"],
                "optional_roles": [],
            },
            {
                "tool_name": "clean_data",
                "display_name": "Clean Data",
                "status": "ready",
                "method_family": "data_preparation",
                "reason": "Can clean missing values or selected columns with confirmation.",
                "required_roles": [],
                "optional_roles": ["columns"],
            },
        ],
    }

def build_dataset_summary_from_df(
    df: pd.DataFrame,
    *,
    data_version_id: str,
) -> Dict[str, Any]:
    n_rows = int(len(df))
    n_cols = int(len(df.columns))

    numeric_columns = []
    categorical_columns = []
    binary_columns = []
    id_like_columns = []

    n_columns_with_missing = 0
    missing_by_column = {}

    for col in df.columns:
        col_name = str(col)
        series = df[col]
        non_missing = series.dropna()

        n_missing = int(series.isna().sum())
        n_unique = int(non_missing.nunique())

        if n_missing > 0:
            n_columns_with_missing += 1
            missing_by_column[col_name] = {
                "n_missing": n_missing,
                "missing_rate": float(n_missing / n_rows) if n_rows else 0.0,
            }

        semantic_type = _semantic_type_for_series(series)

        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(col_name)
        else:
            categorical_columns.append(col_name)

        if semantic_type == "binary_categorical":
            binary_columns.append(col_name)

        # Simple conservative ID-like heuristic.
        if n_rows > 0:
            unique_rate = float(n_unique / n_rows)
            if unique_rate >= 0.95 and n_unique >= max(3, int(0.8 * n_rows)):
                id_like_columns.append(col_name)

    return {
        "data_version_id": data_version_id,
        "n_rows": n_rows,
        "n_cols": n_cols,
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "binary_columns": binary_columns,
        "id_like_columns": id_like_columns,
        "missingness_summary": {
            "n_columns_with_missing": n_columns_with_missing,
            "missing_by_column": missing_by_column,
        },
    }


def prepare_uploaded_dataset_state(
    *,
    df: pd.DataFrame,
    workspace_dir: str,
    filename: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Prepare backend state updates for a newly uploaded dataset.

    This is the only upload boundary the new UI should call.

    It:
    - creates initial data version
    - creates legacy dataset_profile
    - resets observations/analysis/runtimes
    - sets active_data_version_id
    - records uploaded_dataset_info

    It does not call LLMs, execute tools, or mutate the input state.
    """
    if not isinstance(df, pd.DataFrame):
        raise TypeError("prepare_uploaded_dataset_state requires a pandas DataFrame.")

    if df.empty:
        raise ValueError("Uploaded dataset is empty.")

    workspace_path = Path(workspace_dir)
    workspace_path.mkdir(parents=True, exist_ok=True)

    data_version = create_initial_data_version(
        df=df,
        workspace_dir=str(workspace_path),
        created_by="upload",
        description=f"Initial uploaded dataset: {filename or 'uploaded dataset'}",
    )

    active_data_version_id = data_version["version_id"]

    audit_event = make_audit_event(
        event_type="data_version_created",
        description="Initial data version created from uploaded dataset.",
        version_id=active_data_version_id,
        parent_version_id=None,
        tool_name="dataset_upload",
        action_id=None,
        details={
            "filename": filename,
            "n_rows": int(len(df)),
            "n_cols": int(len(df.columns)),
        },
    )

    dataset_profile = build_legacy_dataset_profile_from_df(df)

    dataset_profile_v2 = build_dataset_profile_v2_from_df(
        df,
        data_version_id=active_data_version_id,
    )

    dataset_summary = build_dataset_summary_from_df(
        df,
        data_version_id=active_data_version_id,
    )

    capability_map = build_basic_capability_map_from_df(
        df,
        data_version_id=active_data_version_id,
    )

    uploaded_dataset_info = make_uploaded_dataset_info(
        df=df,
        filename=filename,
        workspace_dir=str(workspace_path),
        data_version=data_version,
    )

    return {
        "workspace_dir": str(workspace_path),

        # Legacy profile for verify_node compatibility.
        "dataset_profile": dataset_profile,

        # Dataset Intelligence fields used by advisory / plan-only flows.
        "dataset_profile_v2": dataset_profile_v2,
        "dataset_summary": dataset_summary,
        "capability_map": capability_map,

        "data_versions": [data_version],
        "data_audit_log": [audit_event],
        "active_data_version_id": active_data_version_id,
        "uploaded_dataset_info": uploaded_dataset_info,

        # Reset analysis/runtime state for the new dataset.
        "observations": [],
        "analysis_runs": [],
        "pending_plan": None,
        "plan_status": None,
        "plan_execution_status": None,
        "current_plan_step_id": None,
        "current_action": None,
        "current_execution": None,
        "current_verification": None,
        "human_review_required": False,
        "pending_action": None,
        "repair_decision": None,
        "repair_proposal": None,
        "repair_attempts": [],
        "deliverable_check": None,
        "execution_audit": None,
        "state_serialization_audit": None,
        "assistant_response": {
            "response_type": "dataset_loaded",
            "content": (
                f"Dataset `{filename or 'uploaded dataset'}` loaded successfully "
                f"with {len(df)} rows and {len(df.columns)} columns."
            ),
            "source_node": "dataset_upload",
            "metadata": {
                "active_data_version_id": active_data_version_id,
            },
        },
    }