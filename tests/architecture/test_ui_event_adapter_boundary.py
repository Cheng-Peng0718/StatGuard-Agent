from pathlib import Path


def test_ui_event_adapter_does_not_import_streamlit_or_app():
    text = Path("core/ui_adapter/events.py").read_text(encoding="utf-8")

    forbidden = [
        "streamlit",
        "import app",
        "from app",
    ]

    offenders = [item for item in forbidden if item in text]

    assert offenders == []


def test_ui_event_adapter_does_not_execute_tools_or_call_llm():
    text = Path("core/ui_adapter/events.py").read_text(encoding="utf-8")

    forbidden = [
        "execute_node",
        "execute_tool",
        "run_tool",
        "client.chat",
        "responses.create",
        ".invoke(",
    ]

    offenders = [item for item in forbidden if item in text]

    assert offenders == []


def test_ui_event_adapter_exposes_event_contract():
    text = Path("core/ui_adapter/events.py").read_text(encoding="utf-8")

    assert "class UIEvent" in text
    assert "def normalize_ui_event" in text
    assert "def apply_ui_event_to_state" in text
    assert "user_message" in text
    assert "approve_human_review" in text
    assert "reject_human_review" in text
    assert "run_plan" in text