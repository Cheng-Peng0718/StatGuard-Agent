from __future__ import annotations

from typing import Callable, Dict, List

import streamlit as st


def render_chat_panel(
    *,
    snapshot: Dict,
    chat_history: List[Dict],
    on_user_message: Callable[[str], None],
) -> None:
    st.subheader("Chat")

    with st.container(height=420):
        if not chat_history:
            st.info("Ask about your dataset or request a plan.")

        for message in chat_history:
            role = message.get("role", "assistant")
            content = message.get("content", "")

            if role == "user":
                st.markdown(f"**You:** {content}")
            else:
                st.markdown(f"**Agent:** {content}")

    with st.form("app_v3_chat_form", clear_on_submit=True):
        user_text = st.text_input(
            "Message",
            placeholder="Ask about your dataset...",
            label_visibility="collapsed",
        )

        submitted = st.form_submit_button("Send", use_container_width=True)

    if submitted and user_text.strip():
        on_user_message(user_text.strip())
        st.rerun()