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

    uploaded_dataset_info = make_uploaded_dataset_info(
        df=df,
        filename=filename,
        workspace_dir=str(workspace_path),
        data_version=data_version,
    )

    return {
        "workspace_dir": str(workspace_path),
        "dataset_profile": dataset_profile,
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