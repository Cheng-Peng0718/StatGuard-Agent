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
    data_versions: List[Dict[str, Any]] | None = None,
    active_data_version_id: str | None = None,
    fallback_file: str = "working_data.parquet",
) -> str | None:
    data_versions = data_versions or []

    # If an active version is explicitly declared, it must resolve.
    # Do not silently fall back to working_data.parquet.
    if active_data_version_id:
        for version in data_versions:
            if not isinstance(version, dict):
                continue

            version_id = (
                version.get("version_id")
                or version.get("id")
                or version.get("data_version_id")
            )

            if version_id == active_data_version_id:
                return version.get("path")

        return None

    # Fallback is allowed only when there is no declared active version.
    if fallback_file:
        import os
        return os.path.join(workspace_dir, fallback_file)

    return None


def extract_data_version_update(raw_result: Any) -> Optional[Dict[str, Any]]:
    """
    Extract data_version_update from plugin execution output.

    Supported locations:
    1. raw_result["data_version_update"]
    2. raw_result["payload"]["data_version_update"]
    3. raw_result["details"]["data_version_update"]

    Returns None if no data_version_update is present.
    """
    if not isinstance(raw_result, dict):
        return None

    top_level = raw_result.get("data_version_update")
    if top_level is not None:
        return top_level

    payload = raw_result.get("payload")
    if isinstance(payload, dict):
        from_payload = payload.get("data_version_update")
        if from_payload is not None:
            return from_payload

    details = raw_result.get("details")
    if isinstance(details, dict):
        from_details = details.get("data_version_update")
        if from_details is not None:
            return from_details

    return None


def validate_data_version_update(
    data_version_update: Any,
) -> Optional[Dict[str, Any]]:
    """
    Validate mutating analysis_tool_plugins data_version_update.

    Contract:
    - valid update -> normalized dict
    - invalid/malformed update -> None

    Invalid updates must never set active_data_version_id to None.
    """
    if data_version_update is None:
        return None

    if not isinstance(data_version_update, dict):
        return None

    new_version = data_version_update.get("new_version")
    active_data_version_id = data_version_update.get("active_data_version_id")

    if not isinstance(new_version, dict):
        return None

    new_version_id = (
        data_version_update.get("new_version_id")
        or new_version.get("version_id")
    )

    if not new_version_id:
        return None

    if not active_data_version_id:
        return None

    if active_data_version_id != new_version_id:
        return None

    if not new_version.get("version_id"):
        new_version["version_id"] = new_version_id

    return {
        **data_version_update,
        "new_version_id": new_version_id,
        "active_data_version_id": active_data_version_id,
        "new_version": new_version,
    }