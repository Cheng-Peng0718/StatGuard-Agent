from pathlib import Path


def test_graph_uses_state_serialization_audit_observe_only():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    assert "audit_state_serialization" in graph_text
    assert "def _attach_state_serialization_audit" in graph_text
    assert "state_serialization_audit" in graph_text


def test_graph_does_not_store_full_safe_state_in_serialization_audit():
    graph_text = Path("core/graph.py").read_text(encoding="utf-8")

    start = graph_text.index("def _compact_state_serialization_audit")
    rest = graph_text[start + 1:]
    next_def_offset = rest.find("\ndef ")
    body = graph_text[start:] if next_def_offset == -1 else graph_text[start:start + 1 + next_def_offset]

    assert '"safe_state"' not in body
    assert "'safe_state'" not in body