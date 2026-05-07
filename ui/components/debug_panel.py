from __future__ import annotations

from typing import Dict

import streamlit as st


def render_debug_panel(snapshot: Dict) -> None:
    st.subheader("Debug")

    with st.expander("Audit state", expanded=False):
        st.json(snapshot.get("audits") or {})

    with st.expander("Repair state", expanded=False):
        st.json(snapshot.get("repair") or {})

    with st.expander("Raw UI snapshot", expanded=False):
        st.json(snapshot)