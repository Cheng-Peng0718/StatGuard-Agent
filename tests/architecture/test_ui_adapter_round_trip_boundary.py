from pathlib import Path


def test_ui_adapter_round_trip_smoke_does_not_import_ui_app():
    text = Path(
        "tests/integration/test_ui_adapter_round_trip_smoke.py"
    ).read_text(encoding="utf-8")

    assert "import app" not in text
    assert "from app" not in text
    assert "streamlit" not in text.lower()


def test_ui_adapter_round_trip_uses_ui_event_and_snapshot_boundaries():
    text = Path(
        "tests/integration/test_ui_adapter_round_trip_smoke.py"
    ).read_text(encoding="utf-8")

    assert "apply_ui_event_to_state" in text
    assert "build_ui_snapshot" in text
    assert "make_user_message_event" in text
    assert "make_run_plan_event" in text
    assert "make_approve_human_review_event" in text


def test_ui_adapter_round_trip_does_not_mutate_state_directly_for_user_inputs():
    text = Path(
        "tests/integration/test_ui_adapter_round_trip_smoke.py"
    ).read_text(encoding="utf-8")

    # User-originating operations should be represented as UI events.
    assert "human_review_decision\"] = \"approved\"" not in text
    assert "user_request\"] = \"run the plan\"" not in text