import os
os.environ["LANGGRAPH_ALLOWED_MSGPACK_MODULES"] = "core.schema"
import streamlit as st
import uuid
from core.graph import app
import pandas as pd
import time

def typewriter_effect(text, speed=0.015):
    """Character-by-character typewriter streaming."""
    for char in text:
        yield char
        time.sleep(speed)

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

st.set_page_config(page_title="AI Data Analyst", page_icon="📊", layout="wide")
st.title("Enterprise intelligent data analysis workstation")

if "session_id" not in st.session_state:
    st.session_state.session_id = f"web_{uuid.uuid4().hex[:8]}"
    st.session_state.workspace = os.path.join("workspaces", st.session_state.session_id)
    os.makedirs(st.session_state.workspace, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []

config = {
    "configurable": {
        "thread_id": st.session_state.thread_id
    }
}

with st.sidebar:
    st.header("Data import")
    uploaded_file = st.file_uploader("Upload data", type=['csv', 'xls', 'xlsx'])
    if uploaded_file:
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
            st.success("✅ Data converted to Parquet and mounted in the sandbox")

        except Exception as e:
            st.error(f"Data processing failed: {str(e)}")

    state_snapshot = app.get_state(config)
    plan = state_snapshot.values.get("analysis_plan")
    if plan:
        st.divider()
        st.subheader("📝 Current analysis plan")
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

            app.update_state(config, {"current_verification": vr})
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
    if not uploaded_file:
        st.warning("Please upload data first")
        st.stop()
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
                profile_dict = {"n_rows": "unknown", "columns": {"unknown": "unknown type"}}

            state_input = {
                "user_request": st.session_state.messages[-1]["content"],
                "max_steps": 30,
                "workspace_dir": st.session_state.get("workspace", "./"),
                "dataset_profile": profile_dict,
                "deliverable_gate_attempts": 0,
                "deliverable_check": None,
                "task_contract": None
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
                                st.markdown("🧠 **Agent reasoning:**")
                                st.write_stream(typewriter_effect(f"> *{reasoning}*"))

                            if action_type == "tool_call":
                                st.info(f"🛠️ Scheduling tool: `{tool_name}`", icon="⚙️")
                            elif action_type == "final_answer":
                                st.success("✨ Reasoning complete, preparing report.", icon="🎉")

                elif node_name == "execute_node":
                    execution = state_data.get("current_execution")

                    with live_display.container():
                        if isinstance(execution, str) and (
                            "System intervention" in execution or "Fingerprint gate" in execution or "❌" in execution
                        ):
                            st.warning(f"🚧 System message: {execution}", icon="⚠️")
                        else:
                            st.success(f"✅ Tool finished; syncing to memory...")
                        time.sleep(0.5)

                current_action = state_data.get("current_action")

                if current_action and hasattr(current_action, "action_type"):

                    if current_action.action_type == "final_answer":
                        pending_final_answer = current_action.reasoning_summary
                        continue

                    elif current_action.action_type == "ask_user":
                        live_display.empty()
                        st.warning(f"🤖 Agent asks for input: {current_action.reasoning_summary}")

                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": f"🤖 Agent asks for input: {current_action.reasoning_summary}"
                        })
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
