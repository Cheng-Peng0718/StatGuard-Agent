import streamlit as st


def render_data_version_panel():
    if not st.session_state.get("active_data_version_id"):
        st.divider()
        st.info(
            "No active analysis dataset loaded yet. You can upload a CSV/Excel file, "
            "or start from a SQL database and let the agent build an analysis-ready dataset."
        )
        st.caption(
            "Try: `Analyze the ecommerce database in demo_data/ecommerce_demo.duckdb and identify what drives revenue.`"
        )
        return

    st.divider()
    st.subheader("Data version")
    st.caption(f"Active version: `{st.session_state.active_data_version_id}`")

    versions = st.session_state.get("data_versions", [])
    if versions:
        latest = versions[-1]
        st.write(f"Rows: {latest.get('n_rows')}, Columns: {latest.get('n_cols')}")
        st.write(f"Operation: {latest.get('operation')}")

    audit_log = st.session_state.get("data_audit_log", [])

    if audit_log:
        with st.expander("Data audit trail"):
            for event in audit_log:
                event_type = event.get("event_type", "unknown_event")
                version_id = event.get("version_id")
                parent_version_id = event.get("parent_version_id")
                created_at = event.get("created_at", "")

                st.markdown(f"**{event_type}**")
                if version_id:
                    st.caption(f"version: `{version_id}`")
                if parent_version_id:
                    st.caption(f"parent: `{parent_version_id}`")
                if created_at:
                    st.caption(created_at)

                st.write(event.get("description", ""))

                details = event.get("details", {})
                if details:
                    st.json(details)

                st.divider()