from pathlib import Path


def test_audit_runtime_helpers_live_outside_core_graph():
    graph_text = Path("core/workflow/audit_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    audit_text = Path("core/workflow/audit_runtime.py").read_text(
        encoding="utf-8",
        errors="ignore",
    )

    forbidden_defs = [
        "def _as_plain_dict",
        "def _merge_state_for_audit",
        "def _compact_state_serialization_audit",
        "def _attach_state_serialization_audit",
        "def _attach_execution_audit",
    ]

    for forbidden in forbidden_defs:
        assert forbidden not in graph_text

    required_defs = [
        "def as_plain_dict",
        "def merge_state_for_audit",
        "def compact_state_serialization_audit",
        "def attach_state_serialization_audit",
        "def attach_execution_audit",
    ]

    for required in required_defs:
        assert required in audit_text


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