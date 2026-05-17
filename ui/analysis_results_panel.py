import streamlit as st


def render_analysis_results_panel():
    analysis_runs = st.session_state.get("analysis_runs", [])

    if not analysis_runs:
        return

    st.divider()
    st.subheader("Analysis Results")

    for run in analysis_runs[-10:]:
        title = run.get("title") or run.get("tool_name", "Analysis")
        status = run.get("status", "unknown")
        data_version_id = run.get("data_version_id")

        with st.expander(f"{title} · {status}"):
            if data_version_id:
                st.caption(f"data version: `{data_version_id}`")

            summary_text = run.get("summary", "")
            if summary_text:
                st.write(summary_text)

            guardrails = run.get("guardrails", [])
            if guardrails:
                st.markdown("**Guardrails**")

                for finding in guardrails:
                    severity = finding.get("severity", "info")
                    title = finding.get("title", "Guardrail finding")
                    message = finding.get("message", "")
                    recommendation = finding.get("recommendation")

                    if severity == "critical":
                        st.error(f"**{title}** — {message}")
                    elif severity == "warning":
                        st.warning(f"**{title}** — {message}")
                    else:
                        st.info(f"**{title}** — {message}")

                    if recommendation:
                        st.caption(f"Recommendation: {recommendation}")

            metrics = run.get("metrics", {})
            if metrics:
                st.markdown("**Metrics**")
                st.json(metrics)

            tables = run.get("tables", {})
            if tables:
                st.markdown("**Tables**")
                for table_name, table_data in tables.items():
                    st.caption(table_name)
                    st.json(table_data)

            args = run.get("arguments", {})
            if args:
                st.markdown("**Arguments**")
                st.json(args)

            artifacts = run.get("artifacts", [])
            if artifacts:
                st.markdown("**Artifacts**")
                for artifact in artifacts:
                    artifact_type = artifact.get("type")
                    path = artifact.get("path")
                    name = artifact.get("name", path)

                    if artifact_type == "png" and path:
                        st.caption(name)
                        st.image(path)
                    else:
                        st.json(artifact)