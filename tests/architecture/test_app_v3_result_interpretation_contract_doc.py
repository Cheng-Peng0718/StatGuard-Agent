from pathlib import Path


def test_app_v3_result_interpretation_contract_doc_exists():
    path = Path("docs/23_app_v3_result_interpretation_contract.md")

    assert path.exists()


def test_app_v3_result_interpretation_contract_mentions_required_sections():
    text = Path("docs/23_app_v3_result_interpretation_contract.md").read_text(
        encoding="utf-8"
    )

    required = [
        "What was computed",
        "Key findings",
        "Caveats",
        "Recommended next steps",
        "core.ui_adapter.insight_cards",
        "does not call an LLM",
        "Failed runs should be archived and explained",
    ]

    for item in required:
        assert item in text