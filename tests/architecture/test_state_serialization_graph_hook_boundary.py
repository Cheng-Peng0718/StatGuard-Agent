from pathlib import Path


def test_graph_uses_state_serialization_audit_observe_only():
    audit_text = Path("core/workflow/audit_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "audit_state_serialization" in audit_text
    assert "def attach_state_serialization_audit" in audit_text
    assert "audit_state_serialization" not in graph_text


def test_graph_does_not_store_full_safe_state_in_serialization_audit():
    audit_text = Path("core/workflow/audit_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    start = audit_text.index("def compact_state_serialization_audit")
    rest = audit_text[start + 1:]
    next_def_offset = rest.find("\ndef ")
    body = (
        audit_text[start:]
        if next_def_offset == -1
        else audit_text[start:start + 1 + next_def_offset]
    )

    forbidden_storage_patterns = [
        '"safe_state"',
        "'safe_state'",
        "safe_state:",
        "safe_state =",
        "safe_state=",
    ]

    for pattern in forbidden_storage_patterns:
        assert pattern not in body

    assert '"status": audit_result.status' in body
    assert '"n_issues": len(audit_result.issues)' in body
    assert '"issues": [' in body