import streamlit as st


def configure_page():
    st.set_page_config(page_title="StatGuard Agent", layout="wide")
    st.title("SQL-connected AI Data Analyst")
    st.caption(
        "From SQL databases and uploaded datasets to EDA, KPI analysis, "
        "statistical modeling, visualization, and evidence-based reports."
    )