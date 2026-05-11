from pathlib import Path


def test_app_v2_ui_polish_doc_exists():
    path = Path("docs/20_app_v2_ui_polish_notes.md")

    assert path.exists()


def test_app_v2_ui_polish_doc_preserves_boundary():
    text = Path("docs/20_app_v2_ui_polish_notes.md").read_text(
        encoding="utf-8"
    )

    required = [
        "presentation polish only",
        "must not move backend business logic",
        "apply_ui_event_to_state",
        "run_backend_turn",
        "build_ui_snapshot",
        "prepare_uploaded_dataset_state",
        "UI should show state clearly, not decide backend behavior",
    ]

    for item in required:
        assert item in text