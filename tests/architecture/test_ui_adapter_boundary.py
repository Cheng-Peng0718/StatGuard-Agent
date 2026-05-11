from pathlib import Path


def test_ui_adapter_does_not_import_streamlit_or_app():
    text = Path("core/ui_adapter/snapshot.py").read_text(encoding="utf-8")

    forbidden = [
        "streamlit",
        "import app",
        "from app",
    ]

    offenders = [item for item in forbidden if item in text]

    assert offenders == []


def test_ui_adapter_does_not_execute_tools_or_call_llm():
    text = Path("core/ui_adapter/snapshot.py").read_text(encoding="utf-8")

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


def test_ui_adapter_exposes_build_ui_snapshot():
    text = Path("core/ui_adapter/snapshot.py").read_text(encoding="utf-8")

    assert "def build_ui_snapshot" in text
    assert "schema_version" in text
    assert "assistant_response" in text
    assert "human_review" in text
    assert "runtime" in text