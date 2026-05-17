import os

import pandas as pd
import streamlit as st

from core.context_builder import generate_profile
from core.data_versions import create_initial_data_version, make_audit_event
from ui.session_state import reset_for_new_uploaded_dataset, build_graph_config


def _profile_to_dict(profile):
    if hasattr(profile, "model_dump"):
        return profile.model_dump()
    if hasattr(profile, "dict"):
        return profile.dict()
    if isinstance(profile, dict):
        return profile
    return None


def render_data_import_panel(config):
    st.header("Data import")
    uploaded_file = st.file_uploader("Upload data", type=["csv", "xls", "xlsx"])

    if not uploaded_file:
        return config

    file_signature = f"{uploaded_file.name}:{uploaded_file.size}"

    if st.session_state.get("uploaded_file_signature") == file_signature:
        st.success("✅ Data already mounted in the sandbox")
        return config

    st.session_state.uploaded_file_signature = file_signature

    # New dataset = new graph thread and clean analysis UI state.
    reset_for_new_uploaded_dataset()
    config = build_graph_config()

    try:
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            df_temp = pd.read_excel(uploaded_file)
        elif uploaded_file.name.endswith(".csv"):
            df_temp = pd.read_csv(uploaded_file)
        else:
            st.error("Unsupported file format")
            st.stop()

        initial_version = create_initial_data_version(
            df=df_temp,
            workspace_dir=st.session_state.workspace,
            created_by="upload",
            description=f"Initial uploaded dataset: {uploaded_file.name}",
        )

        st.session_state.data_versions = [initial_version]
        st.session_state.active_data_version_id = initial_version["version_id"]
        st.session_state.data_audit_log = [
            make_audit_event(
                event_type="data_loaded",
                version_id=initial_version["version_id"],
                description=f"Uploaded dataset {uploaded_file.name}",
                details={
                    "filename": uploaded_file.name,
                    "n_rows": int(df_temp.shape[0]),
                    "n_cols": int(df_temp.shape[1]),
                },
            )
        ]

        profile = generate_profile(initial_version["path"])
        st.session_state.dataset_profile = _profile_to_dict(profile)

        st.success("✅ Data converted to Parquet and mounted in the sandbox")

    except Exception as e:
        st.error(f"Data processing failed: {str(e)}")

    return config