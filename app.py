import os
os.environ["LANGGRAPH_ALLOWED_MSGPACK_MODULES"] = "core.schema"

import time
import uuid

import streamlit as st

from core.graph import app as graph_app

from ui.app_config import configure_page
from ui.session_state import (
    init_session_state,
    build_graph_config,
)
from ui.rendering import typewriter_effect
from ui.chat_history import render_chat_history
from ui.data_import_panel import render_data_import_panel
from ui.data_version_panel import render_data_version_panel
from ui.analysis_results_panel import render_analysis_results_panel
from ui.report_export_panel import render_report_export_panel
from ui.dataset_overview_panel import render_dataset_overview_panel


configure_page()
init_session_state()
config = build_graph_config()


with st.sidebar:
    config = render_data_import_panel(config)
    render_data_version_panel()
    render_dataset_overview_panel()
    render_analysis_results_panel()
    render_report_export_panel()

    # Old workflow-style plan display is intentionally disabled.
    # The active architecture is a workbench-style analyst loop:
    # context -> supervisor -> verify -> execute -> summarize -> evidence -> answer.

render_chat_history()


state_snapshot = graph_app.get_state(config)
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

            graph_app.update_state(config, {
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
                    "feedback": feedback,
                })
            elif hasattr(vr, "copy"):
                vr = vr.copy(update={
                    "status": "rejected_recoverable",
                    "feedback": feedback,
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
                action_id = action_dump.get("action_id") or getattr(
                    action,
                    "action_id",
                    f"act_{uuid.uuid4().hex[:8]}",
                )

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

                graph_app.update_state(config, {
                    "current_verification": vr,
                    "observations": [rejection_observation],
                    "human_review_required": False,
                    "pending_action": None,
                })

            else:
                graph_app.update_state(config, {
                    "current_verification": vr,
                    "human_review_required": False,
                    "pending_action": None,
                })

            st.session_state.resume_stream = True
            st.rerun()


elif prompt := st.chat_input("Ask for an analysis."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()


is_new_task = (
    st.session_state.messages
    and st.session_state.messages[-1]["role"] == "user"
)
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
        existing_imgs = (
            set(f for f in os.listdir(workspace_path) if f.endswith(".png"))
            if os.path.exists(workspace_path)
            else set()
        )

        live_display = st.empty()

        pending_final_answer = None
        deliverable_gate_status = None
        deliverable_gate_allows_final = False

        for event in graph_app.stream(state_input, config):
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

                deliverable_check = state_data.get("deliverable_check")

                if deliverable_check:
                    deliverable_gate_status = deliverable_check.get("status")
                    deliverable_gate_allows_final = deliverable_gate_status in {"ok", "blocked"}

                if node_name in {"supervisor_node", "supervisor"}:
                    action = state_data.get("current_action")
                    if action:
                        reasoning = getattr(action, "reasoning_summary", "")
                        action_type = getattr(action, "action_type", "")
                        tool_name = getattr(action, "tool_name", "")

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

                    if current_action.action_type == "ask_user":
                        live_display.empty()
                        st.warning(f"Agent asks for input: {current_action.reasoning_summary}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"Agent asks for input: {current_action.reasoning_summary}",
                        })
                        st.rerun()

        post_stream_state = graph_app.get_state(config)

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

        if post_stream_state and post_stream_state.next and "human_review" in post_stream_state.next:
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

            current_imgs = (
                set(f for f in os.listdir(workspace_path) if f.endswith(".png"))
                if os.path.exists(workspace_path)
                else set()
            )
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