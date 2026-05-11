import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import pandas as pd


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def make_version_id(prefix: str = "data") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def get_versions_dir(workspace_dir: str) -> str:
    versions_dir = os.path.join(workspace_dir, "data_versions")
    ensure_dir(versions_dir)
    return versions_dir


def create_initial_data_version(
    df: pd.DataFrame,
    workspace_dir: str,
    created_by: str = "upload",
    description: str = "Initial uploaded dataset",
) -> Dict[str, Any]:
    """
    Create the first immutable raw data version.
    """
    versions_dir = get_versions_dir(workspace_dir)
    version_id = "raw_v1"
    path = os.path.join(versions_dir, f"{version_id}.parquet")

    df.to_parquet(path, index=False)

    return {
        "version_id": version_id,
        "parent_version_id": None,
        "path": path,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "created_by": created_by,
        "created_at": utc_now_iso(),
        "operation": "initial_load",
        "description": description,
        "metadata": {},
    }


def create_child_data_version(
    df: pd.DataFrame,
    workspace_dir: str,
    parent_version_id: Optional[str],
    operation: str,
    created_by: str = "tool",
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Create a new immutable child data version.
    """
    versions_dir = get_versions_dir(workspace_dir)
    version_id = make_version_id("data_v")
    path = os.path.join(versions_dir, f"{version_id}.parquet")

    df.to_parquet(path, index=False)

    return {
        "version_id": version_id,
        "parent_version_id": parent_version_id,
        "path": path,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "created_by": created_by,
        "created_at": utc_now_iso(),
        "operation": operation,
        "description": description or operation,
        "metadata": metadata or {},
    }


def make_audit_event(
    event_type: str,
    description: str,
    version_id: Optional[str] = None,
    parent_version_id: Optional[str] = None,
    tool_name: Optional[str] = None,
    action_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "event_id": f"audit_{uuid.uuid4().hex[:8]}",
        "event_type": event_type,
        "version_id": version_id,
        "parent_version_id": parent_version_id,
        "tool_name": tool_name,
        "action_id": action_id,
        "description": description,
        "details": details or {},
        "created_at": utc_now_iso(),
    }


def find_data_version(data_versions: List[Dict[str, Any]], version_id: str) -> Optional[Dict[str, Any]]:
    for v in data_versions or []:
        if v.get("version_id") == version_id:
            return v
    return None


def get_active_data_path(
    workspace_dir: str,
    data_versions: Optional[List[Dict[str, Any]]] = None,
    active_data_version_id: Optional[str] = None,
    fallback_file: str = "working_data.parquet",
) -> str:
    """
    Resolve active data path.

    If active_data_version_id exists, use that version.
    Otherwise fall back to workspace/working_data.parquet for backward compatibility.
    """
    if data_versions and active_data_version_id:
        version = find_data_version(data_versions, active_data_version_id)
        if version and version.get("path"):
            return version["path"]

    return os.path.join(workspace_dir, fallback_file)