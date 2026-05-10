from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.app_backend.snapshot import build_ui_snapshot
from core.context_builder import generate_profile
from core.data.context_refresh import refresh_dataset_context_from_path
from core.data_versions import create_initial_data_version, make_audit_event


SUPPORTED_UPLOAD_EXTENSIONS = {".csv", ".xlsx", ".xls", ".parquet"}


def read_uploaded_dataframe(source_path: str) -> pd.DataFrame:
    """
    Read a user-uploaded tabular file into a DataFrame.

    This function is UI-framework-agnostic. Streamlit/FastAPI should save the
    uploaded file to a temporary path first, then call this backend function.
    """
    path = Path(source_path)
    suffix = path.suffix.lower()

    if suffix not in SUPPORTED_UPLOAD_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_UPLOAD_EXTENSIONS))
        raise ValueError(
            f"Unsupported uploaded file type '{suffix}'. "
            f"Supported types: {supported}."
        )

    if suffix == ".csv":
        return pd.read_csv(path)

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)

    if suffix == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(f"Unsupported uploaded file type '{suffix}'.")


def _ensure_workspace_dir(workspace_dir: str) -> str:
    workspace = Path(workspace_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    return str(workspace)


def _dataset_name_from_path(source_path: str, dataset_name: Optional[str]) -> str:
    if dataset_name:
        return dataset_name

    name = Path(source_path).stem.strip()

    return name or "uploaded_dataset"


def initialize_dataset_session_from_file(
    source_path: str,
    *,
    workspace_dir: str,
    dataset_name: Optional[str] = None,
    user_request: str = "Dataset uploaded.",
    max_steps: int = 12,
) -> Dict[str, Any]:
    """
    Initialize graph state from an uploaded dataset file.

    This function does not invoke LangGraph and does not execute analysis tools.
    It prepares the canonical backend state that the UI can store in session.
    """
    workspace_dir = _ensure_workspace_dir(workspace_dir)
    dataset_name = _dataset_name_from_path(source_path, dataset_name)

    df = read_uploaded_dataframe(source_path)

    raw_version = create_initial_data_version(
        df,
        workspace_dir=workspace_dir,
        created_by="upload",
        description=f"Initial uploaded dataset: {dataset_name}",
    )

    active_data_version_id = raw_version["version_id"]
    active_data_path = raw_version["path"]

    # Context/profile objects used by the modern planner and UI snapshot.
    refreshed_context = refresh_dataset_context_from_path(
        active_data_path,
        dataset_name=dataset_name,
        data_version_id=active_data_version_id,
    )

    # Context profile still used by build_context/supervisor prompt path.
    state_dataset_profile = generate_profile(active_data_path)

    audit_event = make_audit_event(
        event_type="dataset_uploaded",
        description=f"Uploaded dataset `{dataset_name}` as `{active_data_version_id}`.",
        version_id=active_data_version_id,
        parent_version_id=None,
        tool_name=None,
        action_id=None,
        details={
            "source_path": str(source_path),
            "stored_path": active_data_path,
            "n_rows": raw_version.get("n_rows"),
            "n_cols": raw_version.get("n_cols"),
        },
    )

    state: Dict[str, Any] = {
        "user_request": user_request,
        "dataset_name": dataset_name,
        "workspace_dir": workspace_dir,

        "current_step": 0,
        "max_steps": max_steps,

        "dataset_profile": state_dataset_profile,
        "data_versions": [raw_version],
        "active_data_version_id": active_data_version_id,
        "data_audit_log": [audit_event],

        "observations": [],
        "analysis_runs": [],

        "current_action": None,
        "current_execution": None,
        "current_verification": None,

        "task_contract": None,
        "deliverable_check": None,
        "deliverable_gate_attempts": 0,

        "interaction_intent": None,
        "intent_decision": {},
        "task_spec": {},

        "pending_plan": None,
        "plan_status": None,
        "plan_execution_status": None,
        "current_plan_step_id": None,

        "final_answer": None,
        "assistant_response": {
            "response_type": "dataset_loaded",
            "content": (
                f"Dataset `{dataset_name}` loaded successfully with "
                f"{raw_version['n_rows']} rows and {raw_version['n_cols']} columns."
            ),
            "source_node": "dataset_upload",
            "data_version_id": active_data_version_id,
            "metadata": {
                "active_data_version_id": active_data_version_id,
                "path": active_data_path,
            },
        },

        "execution_audit": {},
        "repair_decision": {},
        "repair_attempts": [],
        "repair_proposal": {},
        "state_serialization_audit": {},

        "latest_ui_event": {},
        "human_review_required": False,
        "pending_action": None,
        "human_review_action_hash": None,
        "human_review_rejection_reason": None,
        "selected_plan_step_id": None,
    }

    state.update(
        refreshed_context.to_state_updates(
            include_dataset_context=True,
            dataset_name=dataset_name,
            state_dataset_profile=state_dataset_profile.model_dump(),
            source="upload",
        )
    )

    return {
        "state": state,
        "snapshot": build_ui_snapshot(state),
    }