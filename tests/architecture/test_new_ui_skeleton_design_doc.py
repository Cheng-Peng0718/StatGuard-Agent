from pathlib import Path


def test_new_ui_skeleton_design_doc_exists():
    path = Path("docs/17_new_ui_skeleton_design.md")

    assert path.exists()


def test_new_ui_skeleton_design_mentions_adapter_boundaries():
    text = Path("docs/17_new_ui_skeleton_design.md").read_text(
        encoding="utf-8"
    )

    required = [
        "apply_ui_event_to_state",
        "build_ui_snapshot",
        "backend_state",
        "ui_snapshot",
        "chat_history",
        "human_review",
        "analysis_runs",
        "data_versions",
        "repair",
        "audits",
    ]

    for item in required:
        assert item in text


def test_new_ui_skeleton_design_forbids_backend_logic_in_ui():
    text = Path("docs/17_new_ui_skeleton_design.md").read_text(
        encoding="utf-8"
    )

    forbidden_contracts = [
        "The UI is not responsible for:",
        "Tool selection.",
        "Verification.",
        "Data-version logic.",
        "Deliverable-gate logic.",
        "Repair decisions.",
        "The new UI must not:",
        "Import tool plugins directly.",
        "Implement deliverable-gate logic.",
        "Implement repair logic.",
    ]

    for item in forbidden_contracts:
        assert item in text