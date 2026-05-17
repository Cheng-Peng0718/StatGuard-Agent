import os
import streamlit as st


def render_chat_history():
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

            image_path = msg.get("image_path")
            if image_path and os.path.exists(image_path):
                st.image(image_path)

            if "image" in msg:
                st.image(msg["image"])