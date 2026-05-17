import streamlit as st


def _as_dict(value):
    if hasattr(value, "model_dump"):
        return value.model_dump()
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return value
    return {}


def render_dataset_overview_panel():
    profile = st.session_state.get("dataset_profile")

    if not profile:
        return

    profile = _as_dict(profile)

    columns = profile.get("columns", {}) or {}
    n_rows = profile.get("n_rows", 0)
    n_cols = profile.get("n_cols", 0)

    numeric_cols = []
    categorical_cols = []
    datetime_cols = []
    id_like_cols = []
    high_missing_cols = []

    for name, info in columns.items():
        info = _as_dict(info)

        semantic_type = info.get("semantic_type", "unknown")
        missing_rate = float(info.get("missing_rate", 0) or 0)

        if info.get("is_numeric_like") or semantic_type in {"numeric", "numeric_like"}:
            numeric_cols.append(name)

        if semantic_type == "categorical":
            categorical_cols.append(name)

        if semantic_type == "datetime":
            datetime_cols.append(name)

        if info.get("is_id_like"):
            id_like_cols.append(name)

        if missing_rate >= 0.2:
            high_missing_cols.append((name, missing_rate))

    st.divider()
    st.subheader("Dataset Overview")

    c1, c2 = st.columns(2)
    c1.metric("Rows", n_rows)
    c2.metric("Columns", n_cols)

    c3, c4 = st.columns(2)
    c3.metric("Numeric-like", len(numeric_cols))
    c4.metric("Categorical", len(categorical_cols))

    if high_missing_cols:
        st.warning(
            "High-missing columns: "
            + ", ".join(
                f"{name} ({rate:.1%})"
                for name, rate in high_missing_cols[:8]
            )
        )

    with st.expander("Column roles inferred from the uploaded data", expanded=False):
        st.markdown("**Numeric-like columns**")
        st.write(numeric_cols or "None detected")

        st.markdown("**Categorical columns**")
        st.write(categorical_cols or "None detected")

        st.markdown("**Datetime columns**")
        st.write(datetime_cols or "None detected")

        st.markdown("**ID-like columns**")
        st.write(id_like_cols or "None detected")