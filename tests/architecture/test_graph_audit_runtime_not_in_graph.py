from pathlib import Path


def test_execution_audit_runtime_lives_outside_core_graph():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    audit_text = Path("core/workflow/audit_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "_attach_execution_audit" not in graph_text
    assert "attach_execution_audit" not in graph_text

    assert "def attach_execution_audit" in audit_text