from pathlib import Path


def test_ui_adapter_contract_doc_exists():
    path = Path("docs/16_ui_adapter_contract.md")

    assert path.exists()


def test_ui_adapter_contract_doc_mentions_required_boundaries():
    text = Path("docs/16_ui_adapter_contract.md").read_text(encoding="utf-8")

    required = [
        "build_ui_snapshot",
        "apply_ui_event_to_state",
        "ui_snapshot_v1",
        "user_message",
        "run_plan",
        "approve_human_review",
        "reject_human_review",
        "Old UI freeze",
    ]

    for item in required:
        assert item in text


def test_ui_adapter_contract_doc_forbids_raw_graphstate_dependency():
    text = Path("docs/16_ui_adapter_contract.md").read_text(encoding="utf-8")

    forbidden_dependency_mentions = [
        "must not directly inspect or mutate raw GraphState",
        "Do not add new business logic to app.py",
        "Do not make backend modules import app.py or Streamlit",
    ]

    for item in forbidden_dependency_mentions:
        assert item in text