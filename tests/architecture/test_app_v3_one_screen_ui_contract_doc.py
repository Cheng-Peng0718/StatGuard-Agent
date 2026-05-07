from pathlib import Path


def test_app_v3_one_screen_ui_contract_doc_exists():
    path = Path("docs/21_app_v3_one_screen_ui_contract.md")

    assert path.exists()


def test_app_v3_contract_mentions_one_screen_layout():
    text = Path("docs/21_app_v3_one_screen_ui_contract.md").read_text(
        encoding="utf-8"
    )

    required = [
        "one-screen layout",
        "Chat Panel",
        "Active Workspace",
        "Plan Timeline",
        "Bottom Action Bar",
        "fixed height",
        "internal scrolling",
        "Human Review Priority",
        "Review Mode",
    ]

    for item in required:
        assert item in text


def test_app_v3_contract_preserves_backend_boundaries():
    text = Path("docs/21_app_v3_one_screen_ui_contract.md").read_text(
        encoding="utf-8"
    )

    required = [
        "apply_ui_event_to_state",
        "run_backend_turn",
        "build_ui_snapshot",
        "prepare_uploaded_dataset_state",
        "must not add backend business logic",
        "directly call graph nodes",
        "directly execute tools",
        "hide backend failures with UI workarounds",
    ]

    for item in required:
        assert item in text


def test_app_v3_contract_mentions_future_interpretation_and_planner_layers():
    text = Path("docs/21_app_v3_one_screen_ui_contract.md").read_text(
        encoding="utf-8"
    )

    required = [
        "Result Interpreter",
        "Insight Synthesizer",
        "deterministic baseline planning",
        "LLM-guided adaptive planning",
        "What was computed",
        "Key findings",
        "Interpretation",
        "Caveats",
        "Recommended next step",
    ]

    for item in required:
        assert item in text