from pathlib import Path


def test_state_object_leak_regression_does_not_import_ui_app():
    text = Path(
        "tests/integration/test_state_object_leak_regression.py"
    ).read_text(encoding="utf-8")

    assert "import app" not in text
    assert "from app" not in text
    assert "streamlit" not in text.lower()


def test_state_object_leak_regression_checks_stable_backend_fields():
    text = Path(
        "tests/integration/test_state_object_leak_regression.py"
    ).read_text(encoding="utf-8")

    required_fields = [
        "assistant_response",
        "pending_plan",
        "observations",
        "analysis_runs",
        "data_versions",
        "execution_audit",
        "state_serialization_audit",
        "deliverable_check",
        "repair_decision",
        "repair_proposal",
        "repair_attempts",
    ]

    for field in required_fields:
        assert field in text