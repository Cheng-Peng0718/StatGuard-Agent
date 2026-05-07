from pathlib import Path


def test_manual_smoke_checklist_does_not_authorize_legacy_app_usage():
    text = Path("docs/18_app_v2_manual_smoke_checklist.md").read_text(
        encoding="utf-8"
    )

    assert "legacy `app.py`" in text
    assert "Do not add business logic to `ui/app_v2.py`" in text


def test_manual_smoke_checklist_uses_app_v2_entrypoint():
    text = Path("docs/18_app_v2_manual_smoke_checklist.md").read_text(
        encoding="utf-8"
    )

    assert "streamlit run ui/app_v2.py" in text
    assert "streamlit run app.py" not in text