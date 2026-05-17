import os
import uuid

import streamlit as st


def init_session_state():
    if "thread_id" not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())

    if "session_id" not in st.session_state:
        st.session_state.session_id = f"web_{uuid.uuid4().hex[:8]}"
        st.session_state.workspace = os.path.join("workspaces", st.session_state.session_id)
        os.makedirs(st.session_state.workspace, exist_ok=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "data_versions" not in st.session_state:
        st.session_state.data_versions = []

    if "active_data_version_id" not in st.session_state:
        st.session_state.active_data_version_id = None

    if "data_audit_log" not in st.session_state:
        st.session_state.data_audit_log = []

    if "analysis_runs" not in st.session_state:
        st.session_state.analysis_runs = []

    if "dataset_profile" not in st.session_state:
        st.session_state.dataset_profile = None

    if "resume_stream" not in st.session_state:
        st.session_state.resume_stream = False


def build_graph_config():
    return {
        "configurable": {
            "thread_id": st.session_state.thread_id
        }
    }


def reset_for_new_uploaded_dataset():
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.messages = []
    st.session_state.resume_stream = False

    st.session_state.analysis_runs = []
    st.session_state.dataset_profile = None

    st.session_state.data_versions = []
    st.session_state.active_data_version_id = None
    st.session_state.data_audit_log = []