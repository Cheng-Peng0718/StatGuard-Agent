from pathlib import Path


def test_dataset_upload_adapter_does_not_import_streamlit_or_app():
    text = Path("core/ui_adapter/dataset_upload.py").read_text(encoding="utf-8")

    forbidden = [
        "streamlit",
        "import app",
        "from app",
    ]

    offenders = [
        item
        for item in forbidden
        if item in text
    ]

    assert offenders == []


def test_dataset_upload_adapter_does_not_execute_tools_or_call_llm():
    text = Path("core/ui_adapter/dataset_upload.py").read_text(encoding="utf-8")

    forbidden = [
        "execute_node",
        "execute_tool",
        "run_tool",
        "client.chat",
        "responses.create",
        ".invoke(",
    ]

    offenders = [
        item
        for item in forbidden
        if item in text
    ]

    assert offenders == []


def test_dataset_upload_adapter_owns_dataset_upload_boundary():
    text = Path("core/ui_adapter/dataset_upload.py").read_text(encoding="utf-8")

    assert "def prepare_uploaded_dataset_state" in text
    assert "create_initial_data_version" in text
    assert "build_legacy_dataset_profile_from_df" in text
    assert "uploaded_dataset_info" in text