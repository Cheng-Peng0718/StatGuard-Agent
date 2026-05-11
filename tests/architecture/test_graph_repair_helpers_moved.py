from pathlib import Path


def test_repair_runtime_helpers_live_outside_core_graph():
    graph_text = Path("core/workflow/repair_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    repair_text = Path("core/workflow/repair_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def _attach_repair_decision",
        "def _attach_repair_proposal",
        "def _attach_repair_attempt_if_allowed",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    required_defs = [
        "def attach_repair_decision",
        "def attach_repair_proposal",
        "def attach_repair_attempt_if_allowed",
        "def attach_repair_after_summarize",
    ]

    for required in required_defs:
        assert required in repair_text


def test_core_graph_does_not_import_audit_runtime_helpers():
    graph_text = Path("core/graph.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )
    audit_text = Path("core/workflow/audit_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    assert "from core.workflow.audit_runtime import attach_execution_audit" not in graph_text
    assert "attach_execution_audit" not in graph_text

    assert "def attach_execution_audit" in audit_text