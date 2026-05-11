from pathlib import Path


def test_app_v2_manual_smoke_checklist_exists():
    path = Path("docs/18_app_v2_manual_smoke_checklist.md")

    assert path.exists()


def test_app_v2_manual_smoke_checklist_mentions_core_flows():
    text = Path("docs/18_app_v2_manual_smoke_checklist.md").read_text(
        encoding="utf-8"
    )

    required = [
        "streamlit run ui/app_v2.py",
        "Dataset upload smoke",
        "Advisory chat smoke",
        "Plan request smoke",
        "Run safe plan step smoke",
        "Human review smoke for clean_data",
        "Reject human review smoke",
        "Debug panel smoke",
        "Browser refresh smoke",
    ]

    for item in required:
        assert item in text


def test_app_v2_manual_smoke_checklist_mentions_failure_protocol():
    text = Path("docs/18_app_v2_manual_smoke_checklist.md").read_text(
        encoding="utf-8"
    )

    required = [
        "Do not patch UI immediately.",
        "UI rendering issue",
        "UI adapter issue",
        "backend controller issue",
        "graph node issue",
        "plugin issue",
        "state serialization issue",
        "Do not add business logic to `ui/app_v2.py`",
    ]

    for item in required:
        assert item in text