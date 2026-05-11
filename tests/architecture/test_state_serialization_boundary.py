from pathlib import Path


def test_state_serialization_audit_lives_in_audit_package():
    text = Path("core/audit/state_serialization.py").read_text(encoding="utf-8")

    assert "def audit_state_serialization" in text
    assert "def make_checkpoint_safe_state" in text
    assert "def to_jsonable" in text


def test_state_serialization_audit_does_not_import_ui_or_graph_nodes():
    text = Path("core/audit/state_serialization.py").read_text(encoding="utf-8")

    forbidden_fragments = [
        "streamlit",
        "from app",
        "import app",
        "summarize_node",
        "execute_node",
        "verify_node",
    ]

    offenders = [
        fragment
        for fragment in forbidden_fragments
        if fragment in text
    ]

    assert offenders == []