from __future__ import annotations

import os

import pandas as pd

from core.context_builder import build_context, generate_profile
from core.data_versions import get_active_data_path
from core.dataset_intelligence.capability_map import build_capability_map
from core.dataset_intelligence.profiler import profile_dataframe, summarize_profile


def _load_dataframe_for_dataset_intelligence(path: str) -> pd.DataFrame:
    """
    Load the active dataset for Dataset Intelligence.

    This is separate from generate_profile(), because generate_profile()
    returns the legacy DatasetProfile, while Dataset Intelligence needs
    the actual DataFrame.
    """
    lower_path = str(path).lower()

    if lower_path.endswith(".parquet"):
        return pd.read_parquet(path)

    if lower_path.endswith(".csv"):
        return pd.read_csv(path)

    if lower_path.endswith(".xlsx") or lower_path.endswith(".xls"):
        return pd.read_excel(path)

    raise ValueError(f"Unsupported active data file type for profiling: {path}")


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

    df = _load_dataframe_for_dataset_intelligence(current_data_path)

    active_data_version_id = state.get("active_data_version_id") or "unknown"

    dataset_profile_v2 = profile_dataframe(
        df,
        dataset_name=state.get("dataset_name", "uploaded_dataset"),
        data_version_id=active_data_version_id,
    )

    dataset_summary = summarize_profile(dataset_profile_v2)
    capability_map = build_capability_map(dataset_profile_v2)

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
    )

    return {
        "current_step": step,
        "current_context_text": context.context_text,

        "dataset_profile": new_profile,

        "dataset_profile_v2": dataset_profile_v2.model_dump(),
        "dataset_summary": dataset_summary.model_dump(),
        "capability_map": capability_map.model_dump(),
    }