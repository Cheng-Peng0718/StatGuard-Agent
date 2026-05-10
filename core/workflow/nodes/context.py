from __future__ import annotations

import os

from core.context_builder import build_context, generate_profile
from core.data.context_refresh import refresh_dataset_context_from_path
from core.data_versions import get_active_data_path


def build_context_node(state: dict):
    step = state.get("current_step", 0) + 1

    current_workspace = state["workspace_dir"]

    current_data_path = get_active_data_path(
        workspace_dir=current_workspace,
        data_versions=state.get("data_versions", []) or [],
        active_data_version_id=state.get("active_data_version_id"),
        fallback_file="working_data.parquet",
    )

    if not current_data_path or not os.path.exists(current_data_path):
        raise FileNotFoundError(
            f"No active data file found. "
            f"workspace={current_workspace}, "
            f"active_data_version_id={state.get('active_data_version_id')}, "
            f"resolved_path={current_data_path}"
        )

    new_profile = generate_profile(current_data_path)

    active_data_version_id = state.get("active_data_version_id") or "unknown"

    refreshed_context = refresh_dataset_context_from_path(
        current_data_path,
        dataset_name=state.get("dataset_name", "uploaded_dataset"),
        data_version_id=active_data_version_id,
    )

    context = build_context(
        step=step,
        max_steps=state["max_steps"],
        user_request=state["user_request"],
        profile=new_profile,
        observations=state.get("observations", []),
        workspace_dir=state.get("workspace_dir", "./"),
        deliverable_check=state.get("deliverable_check"),
        data_versions=state.get("data_versions", []),
        active_data_version_id=state.get("active_data_version_id"),
        data_audit_log=state.get("data_audit_log", []),
        task_contract=state.get("task_contract"),
    )

    return {
        "current_step": step,
        "current_context_text": context.context_text,

        "dataset_profile": new_profile,

        **refreshed_context.to_state_updates(
            include_dataset_context=True,
            dataset_name=state.get("dataset_name", "uploaded_dataset"),
            state_dataset_profile=new_profile.model_dump(),
            source="build_context",
        ),
    }
