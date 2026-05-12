import os
os.environ["LANGGRAPH_ALLOWED_MSGPACK_MODULES"] = "core.schema"
import streamlit as st
import uuid
from core.graph import app
import pandas as pd
import time
from core.data_versions import create_initial_data_version, make_audit_event
from core.report_builder import (
    build_markdown_report,
    build_html_report_from_state,
)

def typewriter_effect(text, speed=0.015):
    """Character-by-character typewriter streaming."""
    for char in text:
        yield char
        time.sleep(speed)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("SQL-backed LLM Business Analytics Agent")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_{uuid.uuid4().hex[:8]}"
    st.session_state.workspace = os.path.join("workspaces", st.session_state.session_id)
    os.makedirs(st.session_state.workspace, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

# Data-related session defaults.
# Chat should work even before a CSV/DataFrame dataset is uploaded,
# because SQL tools may operate directly on a database path.
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

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

with st.sidebar:
    st.header("Data import")
    uploaded_file = st.file_uploader("Upload data", type=['csv', 'xls', 'xlsx'])
    if uploaded_file:
        file_signature = f"{uploaded_file.name}:{uploaded_file.size}"

        # Only process a newly uploaded file once.
        if st.session_state.get("uploaded_file_signature") != file_signature:
            st.session_state.uploaded_file_signature = file_signature

            # New dataset = new graph thread, new messages, new workspace state.
            st.session_state.thread_id = str(uuid.uuid4())
            st.session_state.messages = []
            st.session_state.resume_stream = False
            st.session_state.analysis_runs = []
            st.session_state.thread_id = str(uuid.uuid4())

            config = {
                "configurable": {
                    "thread_id": st.session_state.thread_id
                }
            }

            parquet_path = os.path.join(st.session_state.workspace, "working_data.parquet")

            try:
                if uploaded_file.name.endswith(('.xls', '.xlsx')):
                    df_temp = pd.read_excel(uploaded_file)
                elif uploaded_file.name.endswith('.csv'):
                    df_temp = pd.read_csv(uploaded_file)
                else:
                    st.error("Unsupported file format")
                    st.stop()

                df_temp.to_parquet(parquet_path, engine='pyarrow', index=False)

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

                st.success("✅ Data converted to Parquet and mounted in the sandbox")

            except Exception as e:
                st.error(f"Data processing failed: {str(e)}")

        else:
            st.success("✅ Data already mounted in the sandbox")

    if st.session_state.get("active_data_version_id"):
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

        analysis_runs = st.session_state.get("analysis_runs", [])

        if analysis_runs:
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

            st.divider()
            st.subheader("Export Report")

            report_user_request = ""
            if st.session_state.get("messages"):
                user_messages = [
                    m.get("content", "")
                    for m in st.session_state.messages
                    if m.get("role") == "user"
                ]
                report_user_request = user_messages[-1] if user_messages else ""

            markdown_report = build_markdown_report(
                user_request=report_user_request,
                active_data_version_id=st.session_state.get("active_data_version_id"),
                data_versions=st.session_state.get("data_versions", []),
                data_audit_log=st.session_state.get("data_audit_log", []),
                analysis_runs=st.session_state.get("analysis_runs", []),
                title="Data Analysis Report",
            )

            html_report = build_html_report_from_state(
                user_request=report_user_request,
                active_data_version_id=st.session_state.get("active_data_version_id"),
                data_versions=st.session_state.get("data_versions", []),
                data_audit_log=st.session_state.get("data_audit_log", []),
                analysis_runs=st.session_state.get("analysis_runs", []),
                title="Data Analysis Report",
            )

            st.download_button(
                label="Download Markdown Report",
                data=markdown_report,
                file_name="analysis_report.md",
                mime="text/markdown",
                key="download_markdown_report_main",
            )

            st.download_button(
                label="Download HTML Report",
                data=html_report,
                file_name="analysis_report.html",
                mime="text/html",
                key="download_html_report_main",
            )

    else:
        st.divider()
        st.info(
            "No CSV/DataFrame dataset loaded yet. You can upload a CSV/Excel file, "
            "or ask the agent to inspect a SQL database path."
        )
        st.caption(
            "Try: `Inspect the SQL schema for demo_data/ecommerce_demo.duckdb`"
        )

    state_snapshot = app.get_state(config)
    plan = state_snapshot.values.get("analysis_plan")
    if plan:
        st.divider()
        st.subheader("Current analysis plan")
        for i, step in enumerate(plan):
            st.info(f"{i + 1}. {step}")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

        image_path = msg.get("image_path")
        if image_path and os.path.exists(image_path):
            st.image(image_path)
        if "image" in msg:
            st.image(msg["image"])

state_snapshot = app.get_state(config)
is_interrupted = bool(state_snapshot.next and "human_review" in state_snapshot.next)

if is_interrupted:
    values = state_snapshot.values
    action = values.get("current_action")
    vr = values.get("current_verification")

    with st.chat_message("assistant"):
        tool_name = getattr(action, "tool_name", "unknown_tool") if action else "unknown_tool"
        reasoning = getattr(action, "reasoning_summary", "") if action else ""

        st.error(f"**High-risk action requires approval**: Agent will run `{tool_name}`")

        if reasoning:
            st.info(f"**Rationale**: {reasoning}")

        if action:
            action_dump = action.model_dump() if hasattr(action, "model_dump") else {}
            args = action_dump.get("arguments", action_dump.get("args", {}))

            with st.expander("Tool arguments"):
                st.json(args)

            if isinstance(args, dict) and "code" in args:
                st.code(args["code"], language="python")

        if vr is None:
            st.warning("No current_verification. Ensure verify_node always returns current_verification.")
        else:
            vr_status = getattr(vr, "status", None) if not isinstance(vr, dict) else vr.get("status")
            vr_feedback = getattr(vr, "feedback", None) if not isinstance(vr, dict) else vr.get("feedback")

            st.caption(f"verification_status = {vr_status}")
            if vr_feedback:
                st.info(f"**Verifier feedback**: {vr_feedback}")

        c1, c2 = st.columns(2)

        if c1.button("✅ Approve", use_container_width=True):
            vr = state_snapshot.values.get("current_verification")

            if vr is None:
                st.error("Cannot approve: current_verification is missing.")
                st.stop()

            if hasattr(vr, "model_copy"):
                vr = vr.model_copy(update={"status": "allowed"})
            elif hasattr(vr, "copy"):
                vr = vr.copy(update={"status": "allowed"})
            elif isinstance(vr, dict):
                vr = {**vr, "status": "allowed"}
            else:
                vr.status = "allowed"

            app.update_state(config, {
                "current_verification": vr,

                # Preserve data version state across human-review resume.
                "data_versions": st.session_state.get("data_versions", []),
                "active_data_version_id": st.session_state.get("active_data_version_id"),
                "data_audit_log": st.session_state.get("data_audit_log", []),
            })

            st.session_state.resume_stream = True
            st.rerun()

        if c2.button("❌ Reject and rethink", use_container_width=True):
            values = state_snapshot.values
            vr = values.get("current_verification")
            action = values.get("current_action")

            if vr is None:
                st.error("Cannot reject: current_verification is missing.")
                st.stop()

            feedback = (
                "Human reviewer rejected this action. Do not resubmit the same tool with the same arguments. "
                "Explain why it was rejected or propose a non-mutating alternative."
            )

            if hasattr(vr, "model_copy"):
                vr = vr.model_copy(update={
                    "status": "rejected_recoverable",
                    "feedback": feedback
                })
            elif hasattr(vr, "copy"):
                vr = vr.copy(update={
                    "status": "rejected_recoverable",
                    "feedback": feedback
                })
            elif isinstance(vr, dict):
                vr = {
                    **vr,
                    "status": "rejected_recoverable",
                    "feedback": feedback,
                }
            else:
                vr.status = "rejected_recoverable"
                vr.feedback = feedback

            if action is not None:
                if hasattr(action, "model_dump"):
                    action_dump = action.model_dump()
                elif isinstance(action, dict):
                    action_dump = action
                else:
                    action_dump = {}

                tool_name = action_dump.get("tool_name") or getattr(action, "tool_name", None)
                arguments = action_dump.get("arguments") or getattr(action, "arguments", {}) or {}
                action_id = action_dump.get("action_id") or getattr(action, "action_id", f"act_{uuid.uuid4().hex[:8]}")

                rejection_observation = {
                    "observation_id": f"obs_{uuid.uuid4().hex[:8]}",
                    "source_action_id": action_id,
                    "tool_name": tool_name,
                    "arguments": arguments,
                    "status": "rejected",
                    "success": False,
                    "error_code": "HUMAN_REJECTED_ACTION",
                    "message": feedback,
                    "artifacts": [],
                    "summary": (
                        f"Human rejected tool call {tool_name} with arguments {arguments}. "
                        f"The same tool call should not be proposed again unless the user explicitly asks for it."
                    ),
                    "structured_data": {
                        "status": "rejected",
                        "success": False,
                        "error_code": "HUMAN_REJECTED_ACTION",
                        "message": feedback,
                        "tool_name": tool_name,
                        "arguments": arguments,
                    },
                    "raw_data": {
                        "verification": vr.model_dump() if hasattr(vr, "model_dump") else vr,
                        "rejected_action": action_dump,
                    },
                }

                app.update_state(config, {
                    "current_verification": vr,
                    "observations": [rejection_observation],
                    "human_review_required": False,
                    "pending_action": None,
                })

            else:
                app.update_state(config, {
                    "current_verification": vr,
                    "human_review_required": False,
                    "pending_action": None,
                })

            st.session_state.resume_stream = True
            st.rerun()

elif prompt := st.chat_input("Enter your analysis request..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()

is_new_task = (st.session_state.messages and st.session_state.messages[-1]["role"] == "user")
is_resuming = st.session_state.get("resume_stream", False)

if (is_new_task or is_resuming) and not is_interrupted:
    with st.chat_message("assistant"):

        state_input = None
        if is_new_task and not is_resuming:
            real_profile = st.session_state.get("dataset_profile")

            if hasattr(real_profile, "model_dump"):
                profile_dict = real_profile.model_dump()
            elif hasattr(real_profile, "dict"):
                profile_dict = real_profile.dict()
            elif isinstance(real_profile, dict):
                profile_dict = real_profile
            else:
                profile_dict = None

            state_input = {
                "user_request": st.session_state.messages[-1]["content"],
                "max_steps": 20,
                "workspace_dir": st.session_state.get("workspace", "./"),
                "dataset_profile": profile_dict,
                "deliverable_gate_attempts": 0,
                "deliverable_check": None,
                "task_contract": None,

                "data_versions": st.session_state.get("data_versions", []),
                "active_data_version_id": st.session_state.get("active_data_version_id"),
                "data_audit_log": st.session_state.get("data_audit_log", []),
                "analysis_runs": st.session_state.get("analysis_runs", []),
            }

        st.session_state.resume_stream = False
        workspace_path = st.session_state.get("workspace", "./")
        existing_imgs = set([f for f in os.listdir(workspace_path) if f.endswith('.png')]) if os.path.exists(
            workspace_path) else set()

        live_display = st.empty()

        pending_final_answer = None
        deliverable_gate_status = None
        deliverable_gate_allows_final = False

        for event in app.stream(state_input, config):
            for node_name, state_data in event.items():

                if not isinstance(state_data, dict):
                    continue

                if state_data.get("data_versions") is not None:
                    st.session_state.data_versions = state_data.get("data_versions")

                if state_data.get("active_data_version_id") is not None:
                    st.session_state.active_data_version_id = state_data.get("active_data_version_id")

                if state_data.get("data_audit_log") is not None:
                    st.session_state.data_audit_log = state_data.get("data_audit_log")

                if state_data.get("analysis_runs") is not None:
                    st.session_state.analysis_runs = state_data.get("analysis_runs")

                deliverable_check = state_data.get("deliverable_check") if isinstance(state_data, dict) else None

                if deliverable_check:
                    deliverable_gate_status = deliverable_check.get("status")
                    deliverable_gate_allows_final = deliverable_gate_status in {"ok", "blocked"}

                if node_name == "supervisor_node" or node_name == "supervisor":
                    action = state_data.get("current_action")
                    if action:
                        reasoning = getattr(action, 'reasoning_summary', '')
                        action_type = getattr(action, 'action_type', '')
                        tool_name = getattr(action, 'tool_name', '')

                        with live_display.container():
                            if reasoning:
                                st.markdown("**Agent reasoning:**")
                                st.write_stream(typewriter_effect(f"> *{reasoning}*"))

                            if action_type == "tool_call":
                                st.info(f"Scheduling tool: `{tool_name}`")
                            elif action_type == "final_answer":
                                st.success("Reasoning complete, preparing report.")


                elif node_name == "execute_node":

                    execution = state_data.get("current_execution")

                    with live_display.container():

                        if isinstance(execution, dict):

                            status = execution.get("status")

                            message = execution.get("message", "")

                            if status in {"blocked", "failed"}:

                                st.warning(f"Tool returned `{status}`: {message}", icon="⚠️")

                            else:

                                st.success("✅ Tool finished; syncing to memory...")

                        elif isinstance(execution, str):

                            st.warning(f"System message: {execution}", icon="⚠️")

                        else:

                            st.success("✅ Tool finished; syncing to memory...")

                        time.sleep(0.5)

                current_action = state_data.get("current_action")

                if current_action and hasattr(current_action, "action_type"):

                    if current_action.action_type == "final_answer":
                        pending_final_answer = current_action.reasoning_summary
                        continue

                    elif current_action.action_type == "ask_user":
                        live_display.empty()
                        st.warning(f"Agent asks for input: {current_action.reasoning_summary}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Agent asks for input: {current_action.reasoning_summary}"
                        })
                        st.rerun()

        post_stream_state = app.get_state(config)

        if post_stream_state and post_stream_state.values:
            values = post_stream_state.values

            if values.get("data_versions") is not None:
                st.session_state.data_versions = values.get("data_versions")

            if values.get("active_data_version_id") is not None:
                st.session_state.active_data_version_id = values.get("active_data_version_id")

            if values.get("data_audit_log") is not None:
                st.session_state.data_audit_log = values.get("data_audit_log")

            if values.get("analysis_runs") is not None:
                st.session_state.analysis_runs = values.get("analysis_runs")

        if post_stream_state.next and "human_review" in post_stream_state.next:
            live_display.empty()
            st.session_state.resume_stream = False
            st.rerun()

        if pending_final_answer and deliverable_gate_allows_final:
            live_display.empty()

            if deliverable_gate_status == "blocked":
                st.warning("【Final conclusion with limitations】")
            else:
                st.success("【Final conclusion】")

            st.markdown(pending_final_answer)

            current_imgs = set([f for f in os.listdir(workspace_path) if f.endswith(".png")])
            new_imgs = current_imgs - existing_imgs

            for img in new_imgs:
                img_path = os.path.join(workspace_path, img)

                if deliverable_gate_status == "blocked":
                    chart_msg = (
                        f"Generated chart artifact: {img}\n\n"
                        "Note: at least one deliverable was blocked or incomplete, "
                        "so this artifact should be interpreted with the limitations stated above."
                    )
                else:
                    chart_msg = f"Generated chart: {img}"

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": chart_msg,
                    "image_path": img_path,
                })

            st.session_state.messages.append({
                "role": "assistant",
                "content": pending_final_answer,
            })

            st.rerun()


        elif pending_final_answer and not deliverable_gate_allows_final:
            live_display.empty()
            st.info("Final answer was held back because required deliverables are not yet satisfied.")
            # Do not rerun here. The graph should continue through build_context while streaming.
            # If the stream ended here, avoid triggering an infinite UI rerun loop.
