from __future__ import annotations

from typing import Any, Dict, List

import streamlit as st

from core.app_backend import create_app_session


def ensure_session_state() -> None:
    if "app_session" not in st.session_state:
        st.session_state["app_session"] = create_app_session()

    if "graph_state" not in st.session_state:
        st.session_state["graph_state"] = None

    if "snapshot" not in st.session_state:
        st.session_state["snapshot"] = None

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


def current_snapshot() -> Dict[str, Any]:
    return st.session_state.get("snapshot") or {}


def has_dataset(snapshot: Dict[str, Any]) -> bool:
    dataset = snapshot.get("dataset") or {}
    return bool(dataset.get("dataset_name"))


def add_message(role: str, content: str) -> None:
    if not content:
        return

    messages: List[Dict[str, str]] = st.session_state["messages"]

    if (
        messages
        and messages[-1].get("role") == role
        and messages[-1].get("content") == content
    ):
        return

    messages.append({
        "role": role,
        "content": content,
    })


def sync_assistant_response_to_chat(snapshot: Dict[str, Any]) -> None:
    response = snapshot.get("assistant_response") or {}
    content = response.get("content")

    if content:
        add_message("assistant", content)